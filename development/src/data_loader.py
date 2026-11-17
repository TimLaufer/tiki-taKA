"""Load the match tracking (ATD) into numpy arrays on the video timeline.

The data is sensitive: it is read where it lives (config ``data.dir``, or the
environment variable ``MTAH_DATA_DIR``) and never copied into this repo. The
parsed arrays are cached as .npz in ``data.cache_dir``, next to the data.

    python -m src.data_loader              # parse (or read the cache), print a summary
    python -m src.data_loader --no-cache   # force a fresh parse

Parsing uses kloppy's EPTS metadata parser and line patterns, but reads the
lines itself instead of ``kloppy.metrica.load_tracking_epts``: that one builds
a Python object per player per frame (several GB of RAM for a full match) and
drops every ball channel except the position, e.g. the ball speed. Both pieces
are kloppy internals, which is one more reason kloppy is pinned to 3.18.0.

What every field means: docs/DATEN_LEGENDE.md.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
import warnings
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.phases import graded_mask
from src.progress import Progress, say

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"
CACHE_VERSION = 5  # content-addressed sources and parser, strict frame/coordinate validation


def load_config(path=DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve(path) -> Path:
    path = Path(path)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def get_data_dir(config: dict) -> Path:
    """MTAH_DATA_DIR if set; else data.dir; if that does not exist, the first of
    data.fallback_dirs that does (a local layout other than the shared one)."""
    if os.environ.get("MTAH_DATA_DIR"):
        return _resolve(os.environ["MTAH_DATA_DIR"])
    configured = _resolve(config["data"]["dir"])
    if not configured.is_dir():
        for fallback in config["data"].get("fallback_dirs") or []:
            if _resolve(fallback).is_dir():
                return _resolve(fallback)
    return configured


def atd_paths(config: dict, source: str) -> tuple[Path, Path]:
    """Metadata .xml and raw .txt of the tracking for one camera source.

    Each file is taken from data.metadata_file / data.raw_file if set and present
    (tactical only), else from the layout of DATA.md (<data dir>/<source>/<name>),
    else from a search of the whole data dir, which also finds it inside
    unmerged download folders.
    """
    data_dir = get_data_dir(config)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    data = config["data"]
    stem = f"{data['game']}_{source}_atd"
    paths = []
    for key, suffix in (("metadata_file", ".xml"), ("raw_file", ".txt")):
        configured = data.get(key) if source == "tactical" else None
        paths.append(_find_file(data_dir, f"{stem}{suffix}", source, configured))
    return paths[0], paths[1]


def atd_frame_rate(config: dict, source: str) -> float:
    """Frames per second of a recording, as its tracking metadata states."""
    meta_path, _ = atd_paths(config, source)
    for element in ET.parse(meta_path).getroot().iter():
        if element.tag.rsplit("}", 1)[-1] == "FrameRate":
            return float(element.text)
    raise ValueError(f"No FrameRate in {meta_path}")


def load_homography_quality(config: dict, source: str) -> pd.DataFrame:
    """Per video frame of a recording: is there a homography (valid), and how well does it
    fit (reprojection_error, lower is better). The matrices themselves are not loaded."""
    data_dir = get_data_dir(config)
    path = _find_file(data_dir, f"{config['data']['game']}_{source}_homography.csv", source, None)
    table = pd.read_csv(path, usecols=["frame", "reprojection_error", "valid"])
    if not np.array_equal(table["frame"].to_numpy(), np.arange(len(table))):
        raise ValueError(f"{path}: expected one row per video frame, counting from 0")
    if not table["valid"].isin([0, 1]).all():
        raise ValueError(f"{path}: homography valid must be 0 or 1")
    return pd.DataFrame(
        {"valid": table["valid"].astype(bool).to_numpy(),
         "reprojection_error": table["reprojection_error"].astype(float).to_numpy()},
        index=pd.RangeIndex(len(table), name="frame"),
    )


def _find_file(data_dir: Path, name: str, source: str, configured: str | None) -> Path:
    candidates = ([data_dir / configured] if configured else []) + [data_dir / source / name]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    found = sorted(data_dir.rglob(name))
    if len(found) == 1:
        return found[0].resolve()
    if found:
        listing = "\n  ".join(map(str, found))
        raise RuntimeError(f"{name} exists more than once, keep one or set its path in the config:\n  {listing}")
    tried = "\n  ".join(map(str, candidates))
    raise FileNotFoundError(f"{name} not found anywhere under {data_dir}. Looked first at:\n  {tried}")

@dataclass
class Tracking:
    """Tracking of one camera on its video timeline.

    Index i is video frame i; for the tactical camera that is submission row i.
    Positions are normalised to 0-1 as delivered (x along the pitch, y across
    it, origin top-left). Speeds are as delivered, in `speed_unit`. NaN means
    not observed.
    """

    ball: np.ndarray  # (n_frames, 2)
    ball_z: np.ndarray  # (n_frames,), all NaN if the export has no ball height
    ball_speed: np.ndarray  # (n_frames,), all NaN if not exported
    players: np.ndarray  # (n_frames, n_slots, 2)
    speed: np.ndarray  # (n_frames, n_slots), all NaN if not exported
    speed_unit: str  # as declared in the metadata, e.g. "kph"; "" if no speed
    team: np.ndarray  # (n_slots,) "H" or "A"
    slot_ids: np.ndarray  # (n_slots,) track slot ids from the metadata
    team_names: np.ndarray  # [home name, away name]
    home_attacks_first_half: str  # "left_to_right", "right_to_left" or "unknown"
    first_frame: int = 0  # video frame of index 0: 0 for the whole match, more for a part

    @property
    def n_frames(self) -> int:
        return len(self.ball)

    @property
    def frames(self) -> np.ndarray:
        """The video frame of each index."""
        return np.arange(self.first_frame, self.first_frame + self.n_frames)

    def slice(self, first: int, last: int) -> Tracking:
        """Video frames first..last (inclusive) as a Tracking of their own (views, no copy)."""
        if not self.first_frame <= first <= last < self.first_frame + self.n_frames:
            raise IndexError(f"Frames {first}-{last} are outside {self.first_frame}-{self.frames[-1]}")
        part = slice(first - self.first_frame, last - self.first_frame + 1)
        return replace(
            self,
            ball=self.ball[part],
            ball_z=self.ball_z[part],
            ball_speed=self.ball_speed[part],
            players=self.players[part],
            speed=self.speed[part],
            first_frame=first,
        )


def _read_rows(raw_path: Path, meta):
    """One {field: value} dict per line of the raw .txt, with "frame_id".

    Fields are named like kloppy names them ("player_<slot>_<channel>",
    "ball_<channel>"), but every ball channel of every sensor is kept.
    """
    player_map = {pc.player_channel_id: pc for pc in meta.player_channels}
    ball_map = {ch.channel_id: ch for sensor in meta.sensors for ch in sensor.channels}
    specs = sorted(meta.data_format_specifications, key=lambda s: s.start_frame)
    patterns = [re.compile(s.to_regex(player_channel_map=player_map, ball_channel_map=ball_map)) for s in specs]
    frame_fields = [s.split_register.children[0].name for s in specs]

    k = 0  # current layout; layouts cover consecutive frame ranges
    with open(raw_path, encoding="ascii") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = patterns[k].match(line)
            while k + 1 < len(specs) and (match is None or int(match[frame_fields[k]]) > specs[k].end_frame):
                k += 1
                match = patterns[k].match(line)
            if match is None:
                raise ValueError(f"Line does not fit the layout in the metadata: {line[:80]}")
            if match.end() != len(line):
                raise ValueError("ATD row contains trailing fields outside its declared layout")
            row = {key: float(value) if value else np.nan for key, value in match.groupdict().items()}
            frame = row.pop(frame_fields[k])
            if not np.isfinite(frame) or frame != int(frame):
                raise ValueError("ATD frame IDs must be integers")
            row["frame_id"] = int(frame)
            yield row


def parse_atd(meta_path: Path, raw_path: Path, progress: bool = False) -> Tracking:
    from kloppy.infra.serializers.tracking.metrica_epts.metadata import load_metadata

    with open(meta_path, "rb") as f:
        meta = load_metadata(f)

    slot_ids, team, team_names = [], [], {}
    for t in meta.teams:
        if t.ground.name not in ("HOME", "AWAY"):
            continue
        side = "H" if t.ground.name == "HOME" else "A"
        team_names[side] = t.name
        for p in t.players:
            slot_ids.append(p.player_id)
            team.append(side)
    slot_index = {sid: i for i, sid in enumerate(slot_ids)}

    # Field names in kloppy's parsed rows are "player_<slot>_<channel>".
    position_fields, speed_fields, ignored = {}, {}, set()
    for pc in meta.player_channels:
        sid, channel = pc.player.player_id, pc.channel
        if sid not in slot_index:
            continue
        field = f"player_{sid}_{channel.channel_id}"
        if channel.sensor.sensor_id == "position" and channel.channel_id in ("x", "y"):
            position_fields[field] = (slot_index[sid], "xy".index(channel.channel_id))
        elif channel.sensor.sensor_id == "speed":
            speed_fields[field] = slot_index[sid]
        else:
            ignored.add(f"{channel.sensor.sensor_id}/{channel.channel_id}")
    if ignored:
        warnings.warn(f"ATD channels not loaded: {sorted(ignored)}")

    # Ball columns: 0-2 position x/y/z, 3 speed.
    ball_fields, speed_unit = {}, ""
    for sensor in meta.sensors:
        for channel in sensor.channels:
            if sensor.sensor_id == "position" and channel.channel_id in ("x", "y", "z"):
                ball_fields[f"ball_{channel.channel_id}"] = "xyz".index(channel.channel_id)
            elif sensor.sensor_id == "speed":
                ball_fields[f"ball_{channel.channel_id}"] = 3
                speed_unit = str(channel.unit or "")

    n_frames = max(spec.end_frame for spec in meta.data_format_specifications)
    n_slots = len(slot_ids)
    ball = np.full((n_frames, 4), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)

    bar = Progress(n_frames, f"reading {raw_path.name}") if progress else None
    previous = -1
    for row in _read_rows(raw_path, meta):
        i = row["frame_id"] - 1  # ATD frame_id 1..N -> video frame 0..N-1
        if not previous < i < n_frames:
            raise ValueError(f"ATD frame {i + 1} is repeated, unordered or outside 1..{n_frames}")
        previous = i
        for field, value in row.items():
            if field in position_fields:
                players[(i, *position_fields[field])] = value
            elif field in speed_fields:
                speed[i, speed_fields[field]] = value
            elif field in ball_fields:
                ball[i, ball_fields[field]] = value
        if bar:
            bar.update()
    if bar:
        bar.close()

    # Partial pairs and infinities cannot define a position. Keep both coordinates missing.
    players[~np.isfinite(players).all(axis=2)] = np.nan
    ball[~np.isfinite(ball[:, :2]).all(axis=1), :2] = np.nan

    orientation = getattr(getattr(meta, "orientation", None), "name", "")
    home_attacks = {"HOME_AWAY": "left_to_right", "AWAY_HOME": "right_to_left"}
    return Tracking(
        ball=ball[:, :2],
        ball_z=ball[:, 2],
        ball_speed=ball[:, 3],
        players=players,
        speed=speed,
        speed_unit=speed_unit,
        team=np.array(team),
        slot_ids=np.array(slot_ids),
        team_names=np.array([team_names.get("H", ""), team_names.get("A", "")]),
        home_attacks_first_half=home_attacks.get(orientation, "unknown"),
    )


def _cache_path(config: dict, source: str) -> Path:
    name = f"{config['data']['game']}_{source}_tracking.npz"
    return _resolve(config["data"]["cache_dir"]) / name


_CACHED_FIELDS = [f.name for f in fields(Tracking) if f.name != "first_frame"]  # the cache is the whole match


def _parser_identity() -> list[str]:
    """The code that turns an export into arrays - the Tracking fields, _read_rows, parse_atd,
    check_frame_ids, the cache version - and kloppy's metadata parser. A change to any of them
    must invalidate cached arrays, not only a change to the data. Other code of this module
    (finding the files, the summary) does not: editing it must not force a fresh parse."""
    import inspect
    from importlib.metadata import version

    code = "".join(inspect.getsource(part) for part in _PARSER_CODE)
    digest = hashlib.sha256(f"{CACHE_VERSION}|{code}".encode()).hexdigest()
    return [f"parser|{digest}", f"kloppy|{version('kloppy')}"]


def _source_signature(sources: tuple[Path, ...]) -> np.ndarray:
    """Hash contents as well as paths; copied/restored timestamps cannot hide edits."""
    return np.array([f"{p.resolve()}|{file_sha256(p)}" for p in sources] + _parser_identity())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_frame_ids(raw_path: Path, expected: int) -> None:
    """Submission preflight: the delivered export must contain every frame once."""
    count = 0
    with raw_path.open(encoding="ascii") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            try:
                frame = int(line.split(":", 1)[0])
            except ValueError as error:
                raise ValueError(f"{raw_path.name}: malformed ATD frame ID") from error
            if frame != count:
                raise ValueError(f"{raw_path.name}: expected ATD frame {count}, found {frame}")
    if count != expected:
        raise ValueError(f"{raw_path.name}: {count} records, expected {expected}")


# the parsing code as this module defines it (not as a test may patch it later), for _parser_identity
_PARSER_CODE = (Tracking, _read_rows, parse_atd, check_frame_ids)


def _write_cache(tracking: Tracking, path: Path, *sources: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(getattr(tracking, name)) for name in _CACHED_FIELDS}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, version=CACHE_VERSION,
                                source_signature=_source_signature(sources), **arrays)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_cache(path: Path, *sources: Path) -> Tracking | None:
    """The cached tracking, or None if it is missing or stale."""
    if not path.exists() or any(s.stat().st_mtime > path.stat().st_mtime for s in sources):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if int(z["version"]) != CACHE_VERSION:
                return None
            if not np.array_equal(z["source_signature"], _source_signature(sources)):
                return None
            values = {name: z[name] for name in _CACHED_FIELDS}
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        warnings.warn(f"Ignoring unreadable tracking cache: {path}")
        return None
    for name in ("home_attacks_first_half", "speed_unit"):
        values[name] = str(values[name])
    return Tracking(**values)


def load_tracking(config: dict, source: str | None = None, use_cache: bool = True,
                  verbose: bool = True) -> Tracking:
    """The tracking of one recording: from the cache if it is up to date, else parsed and cached.

    verbose: say in the terminal what happens, with a progress bar while parsing.
    """
    source = source or config["data"]["source"]
    meta_path, raw_path = atd_paths(config, source)
    cache = _cache_path(config, source)
    if use_cache:
        cached = _read_cache(cache, meta_path, raw_path)
        if cached is not None:
            if verbose:
                say(f"{source} tracking from the cache ({cache.name})")
            return cached
    if verbose:
        say(f"{source} tracking: reading {raw_path.name}, once - after that it comes from the cache")
    tracking = parse_atd(meta_path, raw_path, progress=verbose)
    if use_cache:
        if verbose:
            say(f"writing the cache {cache} (a few seconds) ...")
        _write_cache(tracking, cache, meta_path, raw_path)
    return tracking


def _speed_stats(values: np.ndarray, unit: str) -> str:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return "not exported"
    median, p99 = np.percentile(values, [50, 99])
    return f"median {median:.1f}, 99th percentile {p99:.1f}, max {values.max():.1f} {unit}"


def summary(tracking: Tracking, config: dict) -> str:
    graded = graded_mask(tracking.n_frames, config)
    ball_seen = ~np.isnan(tracking.ball[:, 0])
    tracked = (~np.isnan(tracking.players[:, :, 0])).sum(axis=1)
    home, away = tracking.team_names
    unit = tracking.speed_unit
    return "\n".join([
        f"frames                  {tracking.n_frames}",
        f"home (H)                {home}: {(tracking.team == 'H').sum()} slots",
        f"away (A)                {away}: {(tracking.team == 'A').sum()} slots",
        f"home attacks 1st half   {tracking.home_attacks_first_half}",
        f"ball visible            {ball_seen.mean():.1%} of all, {ball_seen[graded].mean():.1%} of graded frames",
        f"tracked objects/frame   {tracked[graded].mean():.1f} on graded frames",
        f"ball height exported    {'yes' if (~np.isnan(tracking.ball_z)).any() else 'no'}",
        f"speed unit (metadata)   {unit or '-'}",
        f"player speed            {_speed_stats(tracking.speed[graded].ravel(), unit)}",
        f"ball speed              {_speed_stats(tracking.ball_speed[graded], unit)}",
    ])


def main():
    parser = argparse.ArgumentParser(description="Parse the tracking and print a summary.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source", help="tactical or broadcast (default: config)")
    parser.add_argument("--no-cache", action="store_true", help="parse again, ignore the cache")
    args = parser.parse_args()
    config = load_config(args.config)
    tracking = load_tracking(config, args.source, use_cache=not args.no_cache)
    print(summary(tracking, config))


if __name__ == "__main__":
    main()
