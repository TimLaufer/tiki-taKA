"""Labelled matches from other games, for validating the possession model (RULES.md allows
labelled data from other games; never anything about the provided match).

Metrica Sports' open sample data (https://github.com/metrica-sports/sample-data; same provider
and conventions as the challenge: 25 fps, 0-1 coordinates from the top left, 105 x 68 m):

    Sample Game 1, 2   tracking as CSV (one file per team), events as CSV
    Sample Game 3      tracking as FIFA EPTS (metadata XML + raw TXT, like the challenge ATD),
                       events as JSON

Labels (`possession_labels`) are derived from Metrica's human-annotated events, following its
event definitions (documentation/events-definitions.pdf in that repository):

* A player is in possession from his first touch until another player takes over or the ball goes
  out of play. So the team of the latest on-ball event (SET PIECE, PASS, CARRY, RECOVERY, SHOT,
  BALL LOST, BALL OUT; a BALL LOST keeps its team until the opponent's next event) owns the ball.
  CHALLENGE and CARD events do not change possession by themselves.
* The ball is dead from the end of a BALL OUT, a SHOT that goes OUT or results in a GOAL, a pass
  or loss flagged OFFSIDE, or an intervention (FORCED, VOLUNTARY, END HALF), and from the start
  of a FAULT RECEIVED (the free kick is awarded immediately), until the next on-ball event, which
  restarts play.
* Each period is graded from its first to its last event; other frames are X.

The labels are an event-based proxy: they say who touched the ball last, which is close to, but
not the same as, the challenge's "controlled possession" from corrected tracking.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import Tracking, parse_atd

ON_BALL = ("SET PIECE", "PASS", "CARRY", "RECOVERY", "SHOT", "BALL LOST", "BALL OUT")
STOP_AT_END = ("OUT", "GOAL", "OFFSIDE", "FORCED", "VOLUNTARY", "END HALF")
EVENT_COLUMNS = ["team", "type", "subtype", "period", "start", "end"]


def _game_dir(root: Path, game: int) -> Path:
    path = Path(root) / "data" / f"Sample_Game_{game}"
    if not path.is_dir():
        raise FileNotFoundError(f"{path} not found: download metrica-sports/sample-data there")
    return path


def _read_team_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """One team's tracking CSV: frame numbers, and the x/y column pairs per player and the ball."""
    header = pd.read_csv(path, nrows=3, header=None)
    names = header.iloc[2].tolist()
    table = pd.read_csv(path, skiprows=3, header=None)
    columns = []
    for i, name in enumerate(names):
        if isinstance(name, str) and name not in ("Period", "Frame", "Time [s]"):
            columns.append((name, i))  # x in column i, y in column i + 1
    return table, columns


def load_csv_tracking(root: Path, game: int) -> Tracking:
    """Sample Game 1 or 2 as a Tracking on its frame timeline (index = frame - 1)."""
    folder = _game_dir(root, game)
    tables = {side: _read_team_csv(folder / f"Sample_Game_{game}_RawTrackingData_{name}_Team.csv")
              for side, name in (("H", "Home"), ("A", "Away"))}
    frames = tables["H"][0][1].to_numpy(dtype=int)
    if not np.array_equal(frames, tables["A"][0][1].to_numpy(dtype=int)):
        raise ValueError(f"Game {game}: home and away tracking cover different frames")
    if not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError(f"Game {game}: frames are not 1..n in order")
    players, team, slots = [], [], []
    ball = None
    for side, (table, columns) in tables.items():
        for name, i in columns:
            xy = np.array(table[[i, i + 1]].to_numpy(dtype=np.float32), copy=True)
            if name == "Ball":
                ball = xy if ball is None else ball
                continue
            players.append(xy)
            team.append(side)
            slots.append(f"{side}{name.removeprefix('Player')}")
    n = len(frames)
    players = np.stack(players, axis=1)
    players[~np.isfinite(players).all(axis=2)] = np.nan
    ball[~np.isfinite(ball).all(axis=1)] = np.nan
    return Tracking(ball=ball, ball_z=np.full(n, np.nan, np.float32), ball_speed=np.full(n, np.nan, np.float32),
                    players=players, speed=np.full(players.shape[:2], np.nan, np.float32), speed_unit="",
                    team=np.array(team), slot_ids=np.array(slots), team_names=np.array(["Home", "Away"]),
                    home_attacks_first_half="unknown")


def load_epts_tracking(root: Path, game: int = 3) -> tuple[Tracking, dict[str, str]]:
    """Sample Game 3 through the challenge's own EPTS parser; also team id -> "H"/"A"."""
    folder = _game_dir(root, game)
    meta = folder / f"Sample_Game_{game}_metadata.xml"
    tracking = parse_atd(meta, folder / f"Sample_Game_{game}_tracking.txt")
    from kloppy.infra.serializers.tracking.metrica_epts.metadata import load_metadata

    with open(meta, "rb") as handle:
        teams = load_metadata(handle).teams
    sides = {t.team_id: ("H" if t.ground.name == "HOME" else "A") for t in teams}
    return tracking, sides


def load_events(root: Path, game: int, team_sides: dict[str, str] | None = None) -> pd.DataFrame:
    """Events as one table: team (H/A), type, subtype, period, start and end frame index."""
    folder = _game_dir(root, game)
    if game in (1, 2):
        raw = pd.read_csv(folder / f"Sample_Game_{game}_RawEventsData.csv")
        events = pd.DataFrame({
            "team": raw["Team"].map({"Home": "H", "Away": "A"}).fillna(""),
            "type": raw["Type"].astype(str).str.upper(),
            "subtype": raw["Subtype"].fillna("").astype(str).str.upper(),
            "period": raw["Period"].astype(int),
            "start": raw["Start Frame"].astype(int) - 1,
            "end": raw["End Frame"].astype(int) - 1,
        })
    else:
        data = json.loads((folder / f"Sample_Game_{game}_events.json").read_text())["data"]
        rows = []
        for e in data:
            sub = e.get("subtypes")
            names = [s["name"] for s in sub] if isinstance(sub, list) else [sub["name"]] if sub else []
            rows.append({"team": (team_sides or {}).get(e["team"]["id"], ""), "type": e["type"]["name"].upper(),
                         "subtype": "-".join(names).upper(), "period": int(e["period"]),
                         "start": int(e["start"]["frame"]) - 1, "end": int(e["end"]["frame"]) - 1})
        events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    events["end"] = np.maximum(events["end"], events["start"])  # a few events end "before" they start
    return events.sort_values(["start", "end"], kind="stable", ignore_index=True)


def periods(events: pd.DataFrame) -> list[tuple[int, int]]:
    """(first, last) frame index of each period: from its first event to its last event's end."""
    return [(int(part["start"].min()), int(part["end"].max())) for _, part in events.groupby("period")]


def possession_labels(events: pd.DataFrame, n_frames: int) -> np.ndarray:
    """Per frame H/A/D inside the periods, X outside; see the module docstring."""
    labels = np.full(n_frames, "X", dtype="<U1")
    on_ball = events[events["type"].isin(ON_BALL) & events["team"].isin(["H", "A"])]
    for first, last in periods(events):
        inside = on_ball[(on_ball["start"] >= first) & (on_ball["start"] <= last)]
        part = np.full(last - first + 1, "D", dtype="<U1")  # before the kickoff event: dead
        starts = inside["start"].to_numpy() - first
        for team, start, stop in zip(inside["team"], starts, np.r_[starts[1:], len(part)]):
            part[start:stop] = team
        dead = np.zeros(len(part), dtype=bool)
        stops = events[(events["start"] >= first) & (events["start"] <= last)]
        for row in stops.itertuples():
            # whole subtype tokens: "ON TARGET-GOAL" is a goal, a "GOAL KICK" pass is not
            tokens = set(str(row.subtype).split("-"))
            at_end = row.type == "BALL OUT" or (row.type in ("SHOT", "BALL LOST", "PASS")
                                                and bool(tokens & set(STOP_AT_END)))
            if row.type == "FAULT RECEIVED":
                begin = row.start
            elif at_end:
                begin = row.end
            else:
                continue
            following = inside["start"].to_numpy()
            restart = following[following > begin]
            stop = int(restart[0]) if len(restart) else last + 1
            dead[begin - first : stop - first] = True
        part[dead] = "D"
        labels[first : last + 1] = part
    return labels


# The sample games' corrected ball sits about 0.1 m from the player who has it; in the challenge's
# automatic tracking that distance has a median of 1.8 m on live frames. A slowly drifting error of
# this size gives the sample games the same live-play statistics (nearest-player distance and ball
# speed, both measured on the challenge input without labels).
CHALLENGE_LIKE_NOISE = (1.15, 3.0)  # per-axis standard deviation (m), time constant (s)


def degrade_ball(tracking: Tracking, sigma_m: float, tau_s: float, seed: int, fps: float = 25,
                 pitch=(105.0, 68.0)) -> Tracking:
    """The tracking with a smooth Gaussian error (AR(1), stationary sigma_m per axis, time constant
    tau_s) added to every ball position; reproducible for a given seed."""
    from dataclasses import replace

    rng = np.random.default_rng(seed)
    phi = np.exp(-1.0 / (tau_s * fps))
    steps = rng.normal(0.0, sigma_m * np.sqrt(1 - phi**2), (tracking.n_frames, 2))
    error = np.empty_like(steps)
    error[0] = rng.normal(0.0, sigma_m, 2)
    for i in range(1, len(error)):
        error[i] = phi * error[i - 1] + steps[i]
    ball = tracking.ball.astype(float) + error / np.asarray(pitch, dtype=float)
    return replace(tracking, ball=ball.astype(np.float32))


def controlled_labels(labels: np.ndarray, events: pd.DataFrame, min_s: float = 2.0, fps: float = 25) -> np.ndarray:
    """The labels with CHALLENGE.md's notion of control: a spell won from the opponent in open play
    counts only if the team keeps the ball for min_s or completes a pass in it (one of the proxies
    the challenge names); otherwise it was a touch without control and the previous team keeps the
    ball. Spells after a stoppage always count: a restart has a clear possessor."""
    out = labels.copy()
    passes = events[(events["type"] == "PASS") & events["team"].isin(["H", "A"])]
    i = 1
    while i < len(out):
        team, before = out[i], out[i - 1]
        if team not in ("H", "A") or before not in ("H", "A") or before == team:
            i += 1
            continue
        stop = i
        while stop < len(out) and out[stop] == team:
            stop += 1
        completed = passes[(passes["team"] == team) & (passes["start"] >= i) & (passes["end"] < stop)]
        if stop - i < min_s * fps and completed.empty:
            out[i:stop] = before
        i = stop
    return out


def game_config(base: dict, spans: list[tuple[int, int]], n_frames: int) -> dict:
    """The base configuration with the phase table of an external game: its periods graded,
    everything around them not, the match clock starting at each period's first event."""
    phases, cursor = [], 0
    for number, (first, last) in enumerate(spans, start=1):
        if first > cursor:
            phases.append({"name": "before" if number == 1 else "break", "first": cursor, "last": first - 1,
                           "half": 0, "graded": False})
        phases.append({"name": ("first_half", "second_half")[number - 1], "first": first, "last": last,
                       "half": number, "graded": True, "clock_start_min": 45 * (number - 1)})
        cursor = last + 1
    if cursor < n_frames:
        phases.append({"name": "after", "first": cursor, "last": n_frames - 1, "half": 0, "graded": False})
    config = {**base, "phases": phases, "pitch": [105.0, 68.0], "fps": 25}
    config["sync"] = {**base.get("sync", {}), "enabled": False}
    config["smart_tagging"] = {**base.get("smart_tagging", {}), "enabled": False}
    return config


def load_game(root: Path, game: int, base_config: dict) -> tuple[Tracking, np.ndarray, dict, pd.DataFrame]:
    """(tracking, labels, config, events) of one sample game, on its frame-index timeline."""
    if game in (1, 2):
        tracking, sides = load_csv_tracking(root, game), None
    else:
        tracking, sides = load_epts_tracking(root, game)
    events = load_events(root, game, sides)
    spans = periods(events)
    if len(spans) != 2:
        raise ValueError(f"Game {game}: expected two periods, found {len(spans)}")
    labels = possession_labels(events, tracking.n_frames)
    return tracking, labels, game_config(base_config, spans, tracking.n_frames), events
