"""Match phases on the video timeline, splitting the data by them, and frame <-> time.

    python -m src.phases                   # the phases: frame ranges, duration, what the data has in each
    python -m src.phases --frame 20000     # video time, phase and match clock of a frame (no data needed)
    python -m src.phases --clock 45+2:30   # the frame of a match clock reading (no data needed)

Frames are frames of the tactical video (tactical/BAR-ATH_tactical.mp4), 25 per
second: video time = frame / 25. The match clock starts at each kickoff
(clock_start_min in config.yaml): frame 11185 is 0:00, frame 113742 is 45:00.
Time after 45:00 in the first half reads 45+m:ss, after 90:00 in the second 90+m:ss.

The boundaries live in config.yaml (from DATA.md and FAQ.md), in video frames,
first and last inclusive:

    pre_match                    0 -  11184   not graded
    first_half               11185 -  90651   graded
    halftime                 90652 - 110807   not graded
    second_half_pre_kickoff 110808 - 113741   graded, ground truth is all D
    second_half             113742 - 190197   graded
    post_match              190198 - 199318   not graded

In Python:

    parts = split_tracking(tracking, config)   # {"first_half": Tracking, ...}
    parts["first_half"].players                # (79467, 46, 2): x, y of every track
    parts["first_half"].frames                 # video frame of each index: 11185 ...
    dfs = split_dataframe(df, config)          # {"first_half": DataFrame, ...}
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.data_loader import Tracking


def check_phases(config: dict, n_frames: int) -> None:
    """The phases must cover video frames 0..n_frames-1 in order, without gaps or overlaps."""
    expected = 0
    for p in config["phases"]:
        if p["first"] != expected or p["last"] < p["first"]:
            raise ValueError(f"Phase {p['name']} should start at frame {expected}: check config.yaml")
        expected = p["last"] + 1
    if expected != n_frames:
        raise ValueError(f"The phases end at frame {expected - 1}, the data at frame {n_frames - 1}")


def _per_frame(n_frames: int, config: dict, key: str, fill, dtype) -> np.ndarray:
    out = np.full(n_frames, fill, dtype=dtype)
    for p in config["phases"]:
        out[p["first"] : p["last"] + 1] = p[key]
    return out


def phase_labels(n_frames: int, config: dict) -> pd.Categorical:
    """The phase name of every frame, ordered as in the match."""
    names = [p["name"] for p in config["phases"]]
    return pd.Categorical(_per_frame(n_frames, config, "name", None, object), categories=names, ordered=True)


def half_of_frames(n_frames: int, config: dict) -> np.ndarray:
    """1 or 2 during a half (graded part), 0 elsewhere."""
    return _per_frame(n_frames, config, "half", 0, np.int8)


def graded_mask(n_frames: int, config: dict) -> np.ndarray:
    return _per_frame(n_frames, config, "graded", False, bool)


def phase_at(frame: int, config: dict) -> str:
    return next((p["name"] for p in config["phases"] if p["first"] <= frame <= p["last"]), "")


def first_graded_frame(config: dict) -> int:
    """The first kickoff."""
    return next(p["first"] for p in config["phases"] if p["graded"])


HALF_MINUTES = 45  # regular length of a half; the clock shows later time as stoppage time


def video_time(frame: int, config: dict) -> str:
    """Position in the video as h:mm:ss.ss, e.g. to jump there in a video player."""
    seconds = frame / config["fps"]
    return f"{int(seconds // 3600)}:{int(seconds % 3600 // 60):02d}:{seconds % 60:05.2f}"


def clock_phase(frame: int, config: dict) -> dict | None:
    """The half whose clock is running at this frame, or None (before kickoff, halftime, ...)."""
    return next((p for p in config["phases"] if "clock_start_min" in p and p["first"] <= frame <= p["last"]), None)


def match_seconds(frames, config: dict) -> np.ndarray:
    """The match clock in seconds for many frames at once: 0 at the first kickoff, 2700 at
    the second (45:00). NaN where the clock is not running."""
    frames = np.asarray(frames)
    out = np.full(frames.shape, np.nan)
    for p in config["phases"]:
        if "clock_start_min" in p:
            inside = (frames >= p["first"]) & (frames <= p["last"])
            out[inside] = p["clock_start_min"] * 60 + (frames[inside] - p["first"]) / config["fps"]
    return out


def match_clock(frame: int, config: dict) -> str:
    """The match clock at a frame, as on TV: 23:10, 45+2:30, 67:05, 90+1:00.

    Empty outside the halves (before kickoff, halftime, after the final whistle).
    """
    for p in config["phases"]:
        if p["first"] <= frame <= p["last"] and "clock_start_min" in p:
            start = p["clock_start_min"] * 60
            seconds = start + (frame - p["first"]) / config["fps"]
            regular_end = start + HALF_MINUTES * 60
            if seconds < regular_end:
                return f"{int(seconds // 60)}:{int(seconds % 60):02d}"
            extra = seconds - regular_end
            return f"{regular_end // 60}+{int(extra // 60)}:{int(extra % 60):02d}"
    return ""


def _seconds(text: str) -> float:
    minutes, _, seconds = text.strip().partition(":")
    return int(minutes) * 60 + float(seconds or 0)


def clock_to_frame(clock: str, config: dict) -> int:
    """The frame of a match clock reading: "23:10", "45+2:30" (first half), "67:05", "90+1:00".

    Seconds may have decimals ("23:10.5"); minutes alone ("23") mean :00.
    """
    halves = [p for p in config["phases"] if "clock_start_min" in p]
    base, plus, extra = clock.partition("+")
    if plus:  # stoppage time: the half whose regular end is the base minute
        end_minute = int(base)
        half = next((p for p in halves if p["clock_start_min"] + HALF_MINUTES == end_minute), None)
        if half is None:
            raise ValueError(f"{clock}: stoppage time only after {[p['clock_start_min'] + HALF_MINUTES for p in halves]}")
        seconds = end_minute * 60 + _seconds(extra)
    else:
        seconds = _seconds(base)
        started = [p for p in halves if p["clock_start_min"] * 60 <= seconds]
        if not started:
            raise ValueError(f"{clock}: before the first kickoff")
        half = max(started, key=lambda p: p["clock_start_min"])
        if seconds >= (half["clock_start_min"] + HALF_MINUTES) * 60 and half is not halves[-1]:
            raise ValueError(f"{clock}: write stoppage time as {half['clock_start_min'] + HALF_MINUTES}+m:ss")
    frame = half["first"] + round((seconds - half["clock_start_min"] * 60) * config["fps"])
    if not half["first"] <= frame <= half["last"]:
        raise ValueError(f"{clock} is after the end of {half['name']} at {match_clock(half['last'], config)}")
    return frame


def split_tracking(tracking: Tracking, config: dict) -> dict[str, Tracking]:
    """One Tracking per phase. Same fields as the whole match; `frames` gives the
    video frame of each index. The arrays are views, so this costs no memory."""
    check_phases(config, tracking.n_frames)
    return {p["name"]: tracking.slice(p["first"], p["last"]) for p in config["phases"]}


def split_dataframe(df: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """One DataFrame per phase; the index stays the video frame."""
    check_phases(config, len(df))
    return {p["name"]: df.loc[p["first"] : p["last"]] for p in config["phases"]}


def overview(tracking: Tracking, config: dict) -> pd.DataFrame:
    """One row per phase: where it is, how long, and what the data has in it."""
    rows = []
    for name, part in split_tracking(tracking, config).items():
        p = next(p for p in config["phases"] if p["name"] == name)
        seconds = part.n_frames / config["fps"]
        rows.append({
            "phase": name,
            "first": p["first"],
            "last": p["last"],
            "frames": part.n_frames,
            "duration": f"{int(seconds // 60)}:{int(seconds % 60):02d}",
            "graded": "yes" if p["graded"] else "no",
            "ball_visible": f"{(~np.isnan(part.ball[:, 0])).mean():.1%}",
            "tracked_per_frame": round(float((~np.isnan(part.players[:, :, 0])).sum(axis=1).mean()), 1),
            "note": p.get("note", ""),
        })
    return pd.DataFrame(rows).set_index("phase")


def main():
    from src.data_loader import DEFAULT_CONFIG, load_config, load_tracking

    parser = argparse.ArgumentParser(description="Show the match phases, or convert between frame and time.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source", help="tactical or broadcast (default: config)")
    parser.add_argument("--frame", type=int, help="show video time, phase and match clock of this frame")
    parser.add_argument("--clock", help='show the frame of this match clock, e.g. "23:10" or "45+2:30"')
    args = parser.parse_args()
    config = load_config(args.config)

    if args.clock is not None:
        args.frame = clock_to_frame(args.clock, config)
    if args.frame is not None:
        print(f"frame        {args.frame}")
        print(f"video time   {video_time(args.frame, config)}   (frame / {config['fps']}, in BAR-ATH_tactical.mp4)")
        print(f"phase        {phase_at(args.frame, config) or '-'}")
        print(f"match clock  {match_clock(args.frame, config) or '- (no play)'}")
        return

    tracking = load_tracking(config, args.source)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(overview(tracking, config))


if __name__ == "__main__":
    main()
