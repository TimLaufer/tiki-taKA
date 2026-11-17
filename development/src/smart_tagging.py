"""Metrica's smart tagging, checked against the tracking: is what a tag claims really there?

    python -m src.smart_tagging                  # the set-piece tags, confirmed or not
    python -m src.visualization --tags           # look at them on the pitch (keys n / p)
    python -m src.visualization --tag 12         # ... starting at tag 12

The smart tagging (<game>_tactical_smart_tagging.xml, a SportsCode file) is Metrica's
automatic annotation of the tactical video: phases of play and set pieces, each an interval
(seconds from the first frame of the tactical video, so frame = round(seconds x fps)) with a
code and tags (Team, Half, Type, ...). It is inferred, not ground truth: it has false
positives and false negatives, and every throw-in has team N/A. So each set-piece tag is
checked twice against the tracking (settings under `smart_tagging` in config.yaml):

    ours        a still ball of ours (src/restarts.py) of the same kind, played within
                margin_s of the tag's interval: its verdict and team
    signature   is the mark of that kind unusually strong in the tag's interval? The
                signature, a rolling mean over smooth_s, at its peak in the interval (with
                margin_s around it), against the peaks of every interval of the same length
                in the halves - the share it beats (fit). The marks:
                    throw-in    players within line_band_m of a touchline, or beyond it
                    corner      players in a penalty box (the fuller one)
                    goal kick   the ball in a goal area
                    kickoff     players in their own half
                    free kick   the players slow (their median speed, negated)
                Also, for every kind, how slow the players are (stoppage).

    verdict     confirmed     our restart of that kind, or a fit of at least confirm_pct
                contradicted  no still ball of ours of that kind and a fit below contradict_pct
                unclear       in between

The team: the tag's, or ours where the tag has none. The public ground truth plays no part.

Also, for the prediction (Stefan): `load_set_pieces` gives the SET PIECES intervals as one
fallible cue among several that a stretch without the ball is a stoppage (src/stoppages.py).
The prediction never copies an interval into the labels and never lets a tag decide alone;
the team tag is kept for diagnostics only (it is "N/A" for every throw-in). Time base, as
DATA.md documents: <start> and <end> are seconds from the first frame of the tactical video.
"""
from __future__ import annotations

import argparse
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import Tracking, _find_file, get_data_dir
from src.dead_ball import (
    GOAL_AREA_DEPTH,
    GOAL_AREA_HALF_WIDTH,
    PENALTY_BOX_DEPTH,
    PENALTY_BOX_HALF_WIDTH,
    median_speed,
)
from src.goalkeeper import defended_goals
from src.phases import match_clock, video_time
from src.progress import Steps

SET_PIECES = "SET PIECES"
KINDS = {"throw-in": "throw-in", "corner kick": "corner", "goal kick": "goal kick", "free kick": "free kick",
         "kick off (after goal)": "kickoff", "kick off (start)": "kickoff", "penalty": "penalty"}
MARKS = {"throw-in": "touchline", "corner": "box", "goal kick": "goal_area", "kickoff": "own_half",
         "free kick": "slow"}
CONFIRMED, CONTRADICTED, UNCLEAR = "confirmed", "contradicted", "unclear"
COLUMNS = ["first", "last", "type", "team"]  # the set pieces of load_set_pieces


def smart_tagging_path(config: dict) -> Path:
    """The smart-tagging file: data.smart_tagging_file if set, else <dir>/tactical/<game>_tactical_smart_tagging.xml,
    else searched for below the data dir."""
    data = config["data"]
    return _find_file(get_data_dir(config), f"{data['game']}_tactical_smart_tagging.xml", "tactical",
                      data.get("smart_tagging_file"))


def load_tags(path, config: dict, team_names) -> pd.DataFrame:
    """One row per instance, in time order: tag (its ID), code (upper case), first and last video
    frame, a column per tag group (lower case, e.g. team, type), team_side H / A / "" (N/A)."""
    root = ET.parse(path).getroot()
    fps = config["fps"]
    rows = []
    for instance in root.iter("instance"):
        start, end = float(instance.findtext("start", "nan")), float(instance.findtext("end", "nan"))
        row = {"tag": (instance.findtext("ID") or "").strip(), "code": (instance.findtext("code") or "").strip().upper(),
               "first": int(round(start * fps)), "last": int(round(end * fps))}
        for label in instance.iter("label"):
            group, text = label.findtext("group"), (label.findtext("text") or "").strip()
            if group:
                row[group.strip().lower().replace(" ", "_")] = text
        rows.append(row)
    tags = pd.DataFrame(rows)
    if tags.empty:
        return tags
    tags = tags.sort_values(["first", "last"], kind="stable").reset_index(drop=True)
    tags["team_side"] = [_side(name, team_names) for name in tags.get("team", pd.Series([""] * len(tags)))]
    return tags


def _side(name, team_names) -> str:
    """H / A for a team name of the tags, "" for N/A or an unknown name."""
    if not isinstance(name, str) or not name.strip():
        return ""
    name = name.strip().lower()
    for side, team in zip("HA", team_names):
        team = str(team).strip().lower()
        if name == team or name in team or team in name:
            return side
    return ""


def signatures(tracking: Tracking, ball: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Per video frame inside the halves (NaN outside): the marks of the set pieces, see the
    module docstring. tracking: the players without the keepers held at their last position."""
    settings, pitch, fps = config["smart_tagging"], config["pitch"], config["fps"]
    length, width = pitch
    out, band = config["restarts"]["max_player_out_m"], settings["line_band_m"]
    frames = np.arange(tracking.first_frame, tracking.first_frame + tracking.n_frames)
    xy = tracking.players.astype(np.float32) * np.asarray(pitch, dtype=np.float32)
    x, y = xy[:, :, 0], xy[:, :, 1]
    with np.errstate(invalid="ignore"):  # NaN: no player, counts nowhere
        along = (x >= 0) & (x <= length)
        touchline = along & (((y <= band) & (y >= -out)) | ((y >= width - band) & (y <= width + out)))
        box_band = np.abs(y - width / 2) <= PENALTY_BOX_HALF_WIDTH
        box = np.maximum((box_band & (x >= 0) & (x <= PENALTY_BOX_DEPTH)).sum(axis=1),
                         (box_band & (x <= length) & (x >= length - PENALTY_BOX_DEPTH)).sum(axis=1))
        bx, by = (ball[c].reindex(frames).to_numpy(dtype=float) for c in ("x_m", "y_m"))
        goal_area = ((np.minimum(bx, length - bx) <= GOAL_AREA_DEPTH) & (np.abs(by - width / 2) <= GOAL_AREA_HALF_WIDTH))
    goal_area = np.where(np.isnan(bx), np.nan, goal_area.astype(float))
    own_half = np.full(len(frames), np.nan)
    goals = defended_goals(tracking, config)
    halves = [p for p in config["phases"] if "clock_start_min" in p]
    in_half = np.zeros(len(frames), dtype=bool)
    for half in halves:
        rows = (frames >= half["first"]) & (frames <= half["last"])
        in_half |= rows
        own = np.zeros((rows.sum(), xy.shape[1]), dtype=bool)
        for side in "HA":
            left = goals[(half["name"], side)] == "left"
            with np.errstate(invalid="ignore"):
                own |= (tracking.team == side)[None, :] & ((x[rows] < length / 2) if left else (x[rows] > length / 2))
        seen = (~np.isnan(x[rows])).sum(axis=1)
        own_half[rows] = np.where(seen > 0, own.sum(axis=1) / np.maximum(seen, 1), np.nan)
    marks = pd.DataFrame({
        "touchline": touchline.sum(axis=1).astype(float), "box": box.astype(float), "goal_area": goal_area,
        "own_half": own_half, "slow": -median_speed(tracking, pitch, fps, config["dead_ball"]["speed_window_s"]),
    }, index=pd.RangeIndex(frames[0], frames[-1] + 1, name="frame"))
    marks.loc[~in_half, :] = np.nan
    window = max(int(round(settings["smooth_s"] * fps)), 1)
    return marks.rolling(window, center=True, min_periods=1).mean()


class _Fit:
    """The share of all intervals of the halves of the same length whose peak a peak beats."""

    def __init__(self, marks: pd.DataFrame, fps: int):
        self.marks, self.fps, self.cache = marks, fps, {}

    def __call__(self, mark: str, first: int, last: int) -> float:
        part = self.marks[mark].loc[first:last]
        if part.isna().all():
            return np.nan
        peak = part.max()
        length = max(self.fps, int(round((last - first + 1) / self.fps)) * self.fps)
        key = (mark, length)
        if key not in self.cache:
            peaks = self.marks[mark].rolling(length, min_periods=max(length // 2, 1)).max().to_numpy()
            self.cache[key] = np.sort(peaks[~np.isnan(peaks)])
        peaks = self.cache[key]
        return float(np.searchsorted(peaks, peak, side="left") / len(peaks)) if len(peaks) else np.nan


def check_set_pieces(tags: pd.DataFrame, still_balls: pd.DataFrame, marks: pd.DataFrame, config: dict) -> pd.DataFrame:
    """The set-piece tags inside the halves, checked (see the module docstring), in time order.
    still_balls: src.restarts.candidates (every still ball with its verdict)."""
    settings, fps = config["smart_tagging"], config["fps"]
    margin = int(round(settings["margin_s"] * fps))
    halves = [p for p in config["phases"] if "clock_start_min" in p]
    fit = _Fit(marks, fps)
    set_pieces = tags[tags["code"] == SET_PIECES] if len(tags) else tags
    rows = []
    for tag in set_pieces.itertuples():
        half = next((h for h in halves if tag.first <= h["last"] and tag.last >= h["first"] - margin), None)
        if half is None:
            continue
        kind = _kind(getattr(tag, "type", ""))
        a, b = max(tag.first - margin, half["first"]), min(tag.last + margin, half["last"])
        same = still_balls[(still_balls["kind"] == kind) & (still_balls["last"] >= a) & (still_balls["last"] <= b)]
        ours = None
        if len(same):
            rank = (same["verdict"] != "restart").astype(int) * 10**9 + (same["last"] - tag.first).abs()
            ours = same.loc[rank.idxmin()]
        mark = MARKS.get(kind)
        kind_fit = fit(mark, a, b) if mark else np.nan
        stoppage = fit("slow", a, b)
        if ours is not None and ours["verdict"] == "restart" or kind_fit >= settings["confirm_pct"]:
            verdict = CONFIRMED
        elif ours is None and kind_fit < settings["contradict_pct"]:
            verdict = CONTRADICTED
        else:
            verdict = UNCLEAR
        our_team = ours["team"] if ours is not None else ""
        tag_team = tag.team_side
        rows.append({
            "tag": tag.tag, "half": half["name"], "match": match_clock(max(tag.first, half["first"]), config),
            "video": video_time(tag.first, config), "first": int(tag.first), "last": int(tag.last),
            "duration_s": round((tag.last - tag.first + 1) / fps, 1), "kind": kind, "tag_team": tag_team,
            "ours": f"{ours['verdict']} ({ours['kind']} {ours['team'] or '-'})" if ours is not None else "-",
            "our_team": our_team, "team": tag_team or our_team,
            "team_agrees": ("yes" if tag_team == our_team else "no") if tag_team and our_team else "-",
            "fit": round(kind_fit, 2) if not np.isnan(kind_fit) else np.nan,
            "stoppage": round(stoppage, 2) if not np.isnan(stoppage) else np.nan, "verdict": verdict,
        })
    columns = ["tag", "half", "match", "video", "first", "last", "duration_s", "kind", "tag_team", "ours", "our_team",
               "team", "team_agrees", "fit", "stoppage", "verdict"]
    table = pd.DataFrame(rows, columns=columns)
    table.insert(0, "id", np.arange(1, len(table) + 1))
    table["summary"] = [_summary(row) for row in table.itertuples()]
    return table


def _kind(tag_type) -> str:
    """Our kind for a smart-tagging Type, e.g. "Corner Kick" -> "corner"; unknown ones lower case."""
    tag_type = tag_type.strip().lower() if isinstance(tag_type, str) else ""
    return KINDS.get(tag_type, tag_type)


def _summary(row) -> str:
    team = row.tag_team or (f"{row.our_team} (ours, the tag says N/A)" if row.our_team else "N/A")
    fit = "-" if pd.isna(row.fit) else f"{row.fit:.0%}"
    stoppage = "-" if pd.isna(row.stoppage) else f"{row.stoppage:.0%}"
    text = (f"tag {row.tag}: {row.kind} {team} - {row.verdict}. ours: {row.ours}; signature beats {fit} "
            f"of the moments, players slower than at {stoppage}")
    return text + (" - the team differs from ours" if row.team_agrees == "no" else "")


def untagged(still_balls: pd.DataFrame, tags: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Our restarts (verdict restart) with no set-piece tag of their kind within margin_s."""
    margin = int(round(config["smart_tagging"]["margin_s"] * config["fps"]))
    restarts = still_balls[still_balls["verdict"] == "restart"]
    set_pieces = tags[tags["code"] == SET_PIECES] if len(tags) else tags
    if set_pieces.empty:
        return restarts
    kinds = [_kind(t) for t in set_pieces.get("type", pd.Series([""] * len(set_pieces)))]
    lonely = []
    for row in restarts.itertuples():
        near = [(k == row.kind) and (f - margin <= row.last <= l + margin)
                for k, f, l in zip(kinds, set_pieces["first"], set_pieces["last"])]
        if not any(near):
            lonely.append(row.Index)
    return restarts.loc[lonely]



def parse_set_pieces(path: Path, fps: float, team_names=("", "")) -> pd.DataFrame:
    """SET PIECES instances: first/last video frame (inclusive), type, team ("H", "A" or "")."""
    home, away = (str(name).strip().lower() for name in team_names)
    rows = []
    for instance in ET.parse(path).getroot().iter("instance"):
        if (instance.findtext("code") or "").strip().upper() != SET_PIECES:
            continue
        start, end = float(instance.findtext("start")), float(instance.findtext("end"))
        if not 0 <= start <= end:
            raise ValueError(f"{path.name}: set piece with invalid interval {start}-{end} s")
        tags = {(label.findtext("group") or "").strip().lower(): (label.findtext("text") or "").strip()
                for label in instance.findall("label")}
        team = tags.get("team", "").lower()
        rows.append({"first": int(round(start * fps)), "last": int(round(end * fps)),
                     "type": tags.get("type", ""),
                     "team": "H" if home and team == home else "A" if away and team == away else ""})
    return pd.DataFrame(rows, columns=COLUMNS).sort_values("first", ignore_index=True)


def load_set_pieces(config: dict, team_names=("", "")) -> tuple[pd.DataFrame, Path | None]:
    """The set pieces and the file they came from; empty (and None) when disabled or absent.

    The tagging only corroborates, so a missing file weakens the evidence instead of stopping
    the run; the run report records whether it was used.
    """
    if not config.get("smart_tagging", {}).get("enabled", True):
        return pd.DataFrame(columns=COLUMNS), None
    try:
        path = smart_tagging_path(config)
    except FileNotFoundError as error:
        warnings.warn(f"Smart Tagging not found, stoppages are judged without it: {error}")
        return pd.DataFrame(columns=COLUMNS), None
    return parse_set_pieces(path, config["fps"], team_names), path


def main():
    from src.data_loader import DEFAULT_CONFIG, load_config
    from src.pipeline import load_match, measured
    from src.restarts import candidates

    parser = argparse.ArgumentParser(description="Smart tagging, checked against the tracking.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    steps = Steps(9)

    data = load_match(config, steps)
    with steps("Our still balls (src/restarts.py) and the marks of the set pieces, frame by frame"):
        tracking = measured(data)
        still_balls = candidates(tracking, data.ball, config)
        marks = signatures(tracking, data.ball, config)
    with steps("The smart tagging: every set-piece tag checked"):
        tags = load_tags(smart_tagging_path(config), config, data.tracking.team_names)
        table = check_set_pieces(tags, still_balls, marks, config)

    print(f"\n{len(tags)} smart-tagging instances, per code:")
    print(tags["code"].value_counts().to_string() if len(tags) else "  none")
    print(f"\n{len(table)} set-piece tags in the halves, per kind and verdict:")
    print(pd.crosstab(table["kind"], table["verdict"], margins=True, margins_name="all").to_string()
          if len(table) else "  none")
    if len(table):
        both = table[table["team_agrees"] != "-"]
        found = table[table["ours"] != "-"]
        print(f"\nwith a still ball of ours of that kind: {len(found)} of {len(table)}; where both name a team, "
              f"they agree in {(both['team_agrees'] == 'yes').mean():.0%} of {len(both)}" if len(both) else "")
        filled = table[(table["tag_team"] == "") & (table["our_team"] != "")]
        print(f"tags without a team that ours fills: {len(filled)} of {(table['tag_team'] == '').sum()}")
    lonely = untagged(still_balls, tags, config)
    print(f"\nour restarts with no tag of their kind nearby: {len(lonely)}, per kind:")
    print(lonely["kind"].value_counts().to_string() if len(lonely) else "  none")
    print("\nThe set-piece tags (match data - keep this list to yourself):")
    print(table.drop(columns="summary").to_string(index=False))
    print("\nlook at them on the pitch: .venv/bin/python -m src.visualization --tag <id>   (then n / p)")
    return table


if __name__ == "__main__":
    table = main()
