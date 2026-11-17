"""Ball analysis: clean the ball track, place the ball where it is not seen, derive its motion.

    python -m src.ball_analysis     # how much of the ball track is seen, filled, missing; gap lengths

In Python (main.py builds this as data.ball):

    ball = analyze_ball(ball_dataframe(df), config)   # df: the match DataFrame (to_dataframe)
    ball.loc[20000]                                   # position, source, velocity of frame 20000

How long the ball goes unseen, on the data as delivered (main.py does this per half):

    frames_without_ball(ball)                # per frame: frames without the ball so far, 0 when seen
    gap_stats(ball, config)                  # number of gaps, longest gap and where it is, ...
    plot_frames_without_ball(ball, config)                   # the counter over the video frames
    plot_frames_without_ball(ball, config, x_axis="minute")  # over the match minute, video time on top

The steps run separately inside each match phase, so nothing is carried across
a phase boundary such as halftime. Settings: `ball` in config.yaml.

  1. False detections: a single sighting that jumps away from both neighbouring
     sightings faster than ball.max_speed_ms, while those two agree with each
     other, is dropped (column outlier). Runs of several false frames are not caught.
  2. Gaps: a stretch without the ball between two sightings, at most
     ball.max_gap_s long, is filled on the straight line between them at constant
     speed (source = interpolated). On the pitch plane a ball in flight moves
     exactly like that, a ball at a player's feet roughly. Longer gaps, and gaps at
     the start or end of a phase, stay empty (source = missing).
  3. Motion: velocity from the positions ball.velocity_window frames before and
     after, in m/s.

Output, one row per frame, same index as the input (the video frame):

    phase           copied from the input, if there
    x, y            position 0-1, observed or interpolated, NaN where missing
    x_m, y_m        the same in metres
    source          observed | interpolated | missing
    gap_frames      length of the gap the frame lies in, 0 when observed
    outlier         the detection in this frame was dropped as false
    vx, vy          velocity in m/s along x and y (y points down the screen)
    speed_ms        speed in m/s, from the positions
    speed_exported  the exported ball speed (ball_s), unit as in the metadata
    in_pitch        position inside the pitch lines (False where missing)
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

from src.phases import HALF_MINUTES, clock_phase, match_clock, match_seconds, video_time

SOURCES = ["observed", "interpolated", "missing"]
BALL_COLUMNS = ["phase", "ball_x", "ball_y", "ball_s"]


def ball_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """The ball part of the match DataFrame: phase, ball_x, ball_y, ball_s."""
    ball = df[[c for c in BALL_COLUMNS if c in df]].copy()
    ball.attrs["speed_unit"] = df.attrs.get("speed_unit", "")
    return ball


def _segments(df: pd.DataFrame) -> list[tuple[int, int]]:
    """(start, stop) row positions of each run of one phase; one run if there is no phase column."""
    if "phase" not in df:
        return [(0, len(df))]
    phase = df["phase"].astype(object).to_numpy()
    starts = np.flatnonzero(np.r_[True, phase[1:] != phase[:-1]])
    return list(zip(starts, np.r_[starts[1:], len(df)]))


def _false_detections(x_m: np.ndarray, y_m: np.ndarray, fps: float, max_speed: float) -> np.ndarray:
    """Single sightings that jump away from both neighbouring sightings, which agree with each other."""
    idx = np.flatnonzero(~np.isnan(x_m))
    false = np.zeros(len(x_m), dtype=bool)
    if len(idx) < 3:
        return false
    points, times = np.column_stack([x_m[idx], y_m[idx]]), idx / fps

    def speed(a, b):
        return np.hypot(*(points[b] - points[a]).T) / (times[b] - times[a])

    i = np.arange(1, len(idx) - 1)
    jump = (speed(i - 1, i) > max_speed) & (speed(i, i + 1) > max_speed) & (speed(i - 1, i + 1) <= max_speed)
    false[idx[i[jump]]] = True
    return false


def _neighbours(seen: np.ndarray):
    """Per frame: the last sighting at or before it (-1 if none) and the next at or after it (n if none)."""
    n = len(seen)
    t = np.arange(n)
    previous = np.maximum.accumulate(np.where(seen, t, -1))
    following = np.minimum.accumulate(np.where(seen, t, n)[::-1])[::-1]
    return previous, following


def gap_length(seen: np.ndarray) -> np.ndarray:
    """Per frame: the length of the stretch without the ball it lies in; 0 where the ball is seen."""
    previous, following = _neighbours(seen)
    return np.where(seen, 0, following - previous - 1)


def _fill_gaps(x: np.ndarray, y: np.ndarray, max_gap: int):
    """Fill gaps of at most max_gap frames between two sightings on the straight line.

    Returns x, y, the mask of filled frames and the gap length of every frame.
    """
    n = len(x)
    t = np.arange(n)
    seen = ~np.isnan(x)
    previous, following = _neighbours(seen)
    gap = np.where(seen, 0, following - previous - 1)
    fill = ~seen & (previous >= 0) & (following < n) & (gap <= max_gap)
    if fill.any():
        x, y = x.copy(), y.copy()
        x[fill] = np.interp(t[fill], t[seen], x[seen])
        y[fill] = np.interp(t[fill], t[seen], y[seen])
    return x, y, fill, gap


def _velocity(x_m: np.ndarray, y_m: np.ndarray, fps: float, k: int):
    vx, vy = np.full(len(x_m), np.nan), np.full(len(x_m), np.nan)
    if len(x_m) > 2 * k:
        dt = 2 * k / fps
        vx[k:-k] = (x_m[2 * k :] - x_m[: -2 * k]) / dt
        vy[k:-k] = (y_m[2 * k :] - y_m[: -2 * k]) / dt
    return vx, vy


def analyze_ball(ball: pd.DataFrame, config: dict) -> pd.DataFrame:
    """The ball track cleaned, gap-filled and with its motion; see the module docstring.

    `ball` needs ball_x and ball_y (0-1); phase and ball_s are used if present.
    """
    settings = config["ball"]
    fps = config["fps"]
    length, width = config["pitch"]
    max_gap = int(round(settings["max_gap_s"] * fps))
    window = int(settings["velocity_window"])
    if window < 1:
        raise ValueError("ball.velocity_window must be at least 1")

    x = ball["ball_x"].to_numpy(dtype=float).copy()
    y = ball["ball_y"].to_numpy(dtype=float).copy()
    unseen = np.isnan(x) | np.isnan(y)  # a half position counts as not seen
    x[unseen] = np.nan
    y[unseen] = np.nan

    n = len(ball)
    columns = {name: np.full(n, np.nan) for name in ("x", "y", "vx", "vy")}
    gap = np.zeros(n, dtype=np.int64)
    filled = np.zeros(n, dtype=bool)
    outlier = np.zeros(n, dtype=bool)
    for start, stop in _segments(ball):
        part = slice(start, stop)
        xs, ys = x[part].copy(), y[part].copy()
        false = _false_detections(xs * length, ys * width, fps, settings["max_speed_ms"])
        xs[false] = ys[false] = np.nan
        xs, ys, fill, gaps = _fill_gaps(xs, ys, max_gap)
        vx, vy = _velocity(xs * length, ys * width, fps, window)
        columns["x"][part], columns["y"][part] = xs, ys
        columns["vx"][part], columns["vy"][part] = vx, vy
        gap[part], filled[part], outlier[part] = gaps, fill, false

    observed = ~np.isnan(columns["x"]) & ~filled
    source = np.where(observed, "observed", np.where(filled, "interpolated", "missing"))
    out = pd.DataFrame(index=ball.index)
    if "phase" in ball:
        out["phase"] = ball["phase"]
    out["x"], out["y"] = columns["x"], columns["y"]
    out["x_m"], out["y_m"] = columns["x"] * length, columns["y"] * width
    out["source"] = pd.Categorical(source, categories=SOURCES)
    out["gap_frames"] = gap
    out["outlier"] = outlier
    out["vx"], out["vy"] = columns["vx"], columns["vy"]
    out["speed_ms"] = np.hypot(columns["vx"], columns["vy"])
    if "ball_s" in ball:
        out["speed_exported"] = ball["ball_s"].to_numpy(dtype=float)
    out["in_pitch"] = (out["x"] >= 0) & (out["x"] <= 1) & (out["y"] >= 0) & (out["y"] <= 1)
    if "ball_from" in ball:  # a merged ball: which video each sighting came from
        out["ball_from"] = ball["ball_from"].to_numpy()
    out.attrs["speed_unit"] = ball.attrs.get("speed_unit", "")
    return out


def phase_table(ball: pd.DataFrame) -> pd.DataFrame:
    """Per phase: share of frames with the ball observed, interpolated and missing."""
    groups = ball.groupby("phase", observed=True, sort=False) if "phase" in ball else [("all", ball)]
    rows = []
    for name, part in groups:
        share = part["source"].value_counts(normalize=True)
        rows.append({
            "phase": name,
            "frames": len(part),
            "observed": f"{share.get('observed', 0):.1%}",
            "interpolated": f"{share.get('interpolated', 0):.1%}",
            "missing": f"{share.get('missing', 0):.1%}",
            "false_detections": int(part["outlier"].sum()),
        })
        if "ball_from" in part:
            rows[-1]["from_video_2"] = f"{(part['ball_from'] == 'second').mean():.1%}"
    return pd.DataFrame(rows).set_index("phase")


def gap_table(ball: pd.DataFrame, config: dict) -> pd.DataFrame:
    """The gaps in the graded phases, by length: how many, and how many frames they cover."""
    fps = config["fps"]
    graded = [p["name"] for p in config["phases"] if p["graded"]]
    phase = ball["phase"].astype(object).to_numpy() if "phase" in ball else np.full(len(ball), "all", dtype=object)
    in_graded = np.isin(phase, graded) if "phase" in ball else np.ones(len(ball), dtype=bool)
    without = (ball["source"] != "observed").to_numpy()
    first_of_gap = without & (np.r_[True, phase[1:] != phase[:-1]] | ~np.r_[False, without[:-1]])
    lengths = ball["gap_frames"].to_numpy()[first_of_gap & in_graded]

    labels = ["up to 0.2 s", "0.2-1 s", "1-2 s", "2-5 s", "5-10 s", "over 10 s"]
    bucket = pd.cut(lengths / fps, [0, 0.2, 1, 2, 5, 10, np.inf], labels=labels)
    table = (
        pd.DataFrame({"length": bucket, "frames": lengths})
        .groupby("length", observed=False)
        .agg(gaps=("frames", "size"), frames=("frames", "sum"))
    )
    table["share_of_graded_frames"] = (table["frames"] / max(int(in_graded.sum()), 1)).map("{:.1%}".format)
    return table


def speed_check(ball: pd.DataFrame) -> str:
    """Compare the exported ball speed with the speed from the positions: settles its unit."""
    unit = ball.attrs.get("speed_unit") or "?"
    if "speed_exported" not in ball:
        return "exported ball speed: not in the data"
    moving = (ball["source"] == "observed") & (ball["speed_ms"] > 2) & (ball["speed_exported"] > 0)
    if not moving.any():
        return "exported ball speed: no moving ball to compare with"
    ratio = (ball.loc[moving, "speed_exported"] / ball.loc[moving, "speed_ms"]).median()
    return (
        f"exported ball speed / speed from positions: median {ratio:.2f} "
        f"(3.6 means the export is km/h, 1.0 means m/s; the metadata says '{unit}')"
    )


def frames_without_ball(ball: pd.DataFrame) -> pd.Series:
    """Per frame: how many frames in a row the ball has not been seen, up to and including this one.

    0 where the ball is seen. Seen in frame 4, unseen in 5-9, seen in 10 gives
    1, 2, 3, 4, 5 for frames 5-9 and 0 in frame 10: each peak is the length of a gap.
    Works on the delivered positions (ball_x, ball_y), before anything is filled.
    """
    seen = (ball["ball_x"].notna() & ball["ball_y"].notna()).to_numpy()
    t = np.arange(len(ball))
    last_seen = np.maximum.accumulate(np.where(seen, t, -1))
    return pd.Series(t - last_seen, index=ball.index, name="frames_without_ball")


def gap_stats(ball: pd.DataFrame, config: dict) -> dict:
    """Gaps in the delivered ball track: how many, how long, and the longest one."""
    fps = config["fps"]
    max_gap_s = config["ball"]["max_gap_s"]
    counter = frames_without_ball(ball)
    values = counter.to_numpy()
    gaps = values[(values > 0) & (np.r_[values[1:], 0] == 0)]  # the peaks
    longest = int(values.max()) if len(values) else 0
    end = counter.idxmax() if longest else None
    return {
        "frames": len(values),
        "ball_seen": f"{(values == 0).mean():.1%}",
        "gaps": len(gaps),
        "frames_without_ball": int(gaps.sum()),
        "median_gap_frames": int(np.median(gaps)) if len(gaps) else 0,
        f"gaps_longer_than_{max_gap_s}_s": int((gaps > round(max_gap_s * fps)).sum()),
        "longest_gap_frames": longest,
        "longest_gap_s": round(longest / fps, 1),
        "longest_gap_at_frames": f"{end - longest + 1}-{end}" if longest else "-",
    }


def _minute_label(minute: float, regular_end: int) -> str:
    """A match minute as on TV: 23, 45, 45+3 (first half), 90+2 (second half)."""
    if minute <= regular_end + 1e-6:
        return f"{minute:.0f}"
    return f"{regular_end}+{minute - regular_end:.0f}"


def minute_axes(ax, frames: np.ndarray, minutes: np.ndarray, config: dict,
                 video: str = "tactical", video_offset: int = 0) -> None:
    """Match minutes along the bottom (stoppage time shaded), the time in `video` along the top.

    video_offset: tactical frame - frame of that video (0 for the tactical video itself).
    """
    clocked = np.flatnonzero(~np.isnan(minutes))
    if clocked.size == 0:
        raise ValueError("No match clock in these frames: they are outside the halves")
    half = clock_phase(int(frames[clocked[0]]), config)
    if clock_phase(int(frames[clocked[-1]]), config) is not half:
        raise ValueError("x_axis='minute' works on one half at a time")
    regular_end = half["clock_start_min"] + HALF_MINUTES
    # minute in the video - match minute
    offset = (half["first"] - video_offset) / config["fps"] / 60 - half["clock_start_min"]

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda m, _: _minute_label(m, regular_end)))
    ax.set_xlabel("match minute")
    first, last = np.nanmin(minutes), np.nanmax(minutes)
    if last > first:
        ax.set_xlim(first, last)  # no margin: no minute -2 before the kickoff
    if last > regular_end:
        ax.axvspan(regular_end, last, color="#9ca3af", alpha=0.18, lw=0)
        ax.text(regular_end, 0.97, " stoppage time", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8, color="#4b5563")
    top = ax.secondary_xaxis("top", functions=(lambda m: m + offset, lambda v: v - offset))
    top.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v // 60)}:{int(v % 60):02d}"))
    top.set_xlabel(f"time in the {video} video (h:mm)", fontsize=9)


def plot_frames_without_ball(ball: pd.DataFrame, config: dict, ax=None, title: str = "", x_axis: str = "frame",
                             compact: bool = False, video: str = "tactical", video_offset: int = 0):
    """Frames without the ball so far (see frames_without_ball), over one half.

    x_axis="frame": tactical video frames. x_axis="minute": the match minute as on TV,
    with stoppage time as 45+m; the top axis gives the time in the video, which pre-match
    and halftime shift away from the match minute.
    video, video_offset: which recording the ball comes from and its offset (tactical
    frame - frame of that video), for the top axis and the title. A recording other than
    the tactical one also gets its own frame numbers along the top of the frame plot.
    compact: shorter titles and labels, for a grid of many panels.
    """
    fps = config["fps"]
    counter = frames_without_ball(ball)
    frames, values = counter.index.to_numpy(), counter.to_numpy()
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4))
    if x_axis == "minute":
        x = match_seconds(frames, config) / 60
        minute_axes(ax, frames, x, config, video, video_offset)
    elif x_axis == "frame":
        x = frames
        ax.set_xlabel("video frame" if not video_offset else "tactical video frame")
        if video_offset:
            top = ax.secondary_xaxis("top", functions=(lambda t: t - video_offset, lambda f: f + video_offset))
            top.set_xlabel(f"{video} video frame", fontsize=9)
    else:
        raise ValueError(f"x_axis must be 'frame' or 'minute', not {x_axis!r}")

    ax.plot(x, values, lw=0.6, color="#1d4ed8")
    if "ball_from" in ball:  # a merged ball: show where it came from the second video
        from_second = (ball["ball_from"] == "second").to_numpy()
        if from_second.any():
            ax.fill_between(x, 0, 1, where=from_second, transform=ax.get_xaxis_transform(),
                            color="#f59e0b", alpha=0.35, lw=0, step="mid")
    if "verdict" in ball:  # the long gaps with their dead-ball verdict (src/dead_ball.py)
        for verdict, color in (("dead", "#dc2626"), ("unclear", "#6b7280")):
            where = (ball["verdict"] == verdict).to_numpy()
            if where.any():
                ax.fill_between(x, 0, 1, where=where, transform=ax.get_xaxis_transform(),
                                color=color, alpha=0.3, lw=0, step="mid")
    max_gap_s = config["ball"]["max_gap_s"]
    ax.axhline(max_gap_s * fps, color="#374151", ls="--", lw=1)
    if not compact:
        ax.text(0.005, max_gap_s * fps, f" gaps up to {max_gap_s} s are filled", transform=ax.get_yaxis_transform(),
                va="bottom", fontsize=8, color="#374151")

    if len(values) and values.max() > 0:
        peak = int(np.argmax(values))
        longest, start = int(values[peak]), int(frames[peak]) - int(values[peak]) + 1
        ax.plot([x[peak]], [longest], "o", ms=4, color="#111827")
        clock = match_clock(start, config)
        if compact:
            ax.set_title(f"longest {longest / fps:.1f} s, from {clock or f'frame {start}'}", loc="right", fontsize=8)
        else:
            where = [f"frames {start}-{start + longest - 1}"]
            if clock:
                where.append(f"match {clock}")
            where.append(f"{video} video {video_time(start - video_offset, config)}")
            ax.set_title(f"longest gap: {longest} frames = {longest / fps:.1f} s  |  from " + ", ".join(where),
                         loc="right", fontsize=9)
    ax.set_title(title, loc="left", fontsize=9 if compact else None)
    ax.set_ylabel("frames without ball")
    seconds = ax.secondary_yaxis("right", functions=(lambda f: f / fps, lambda s: s * fps))
    seconds.set_ylabel("seconds")
    ax.set_ylim(bottom=0)
    return ax


def _seen(ball: pd.DataFrame) -> np.ndarray:
    return (ball["ball_x"].notna() & ball["ball_y"].notna()).to_numpy()


def gaps_to_fill(first: pd.DataFrame, second: pd.DataFrame, min_gap_frames: int = 0) -> np.ndarray:
    """The frames where `first` has no ball, lie in a gap longer than min_gap_frames, and
    `second` has one: where merge_ball would take the ball of `second`."""
    seen_first = _seen(first)
    return ~seen_first & _seen(second) & (gap_length(seen_first) > min_gap_frames)


def merge_ball(first: pd.DataFrame, second: pd.DataFrame, min_gap_frames: int = 0,
               usable: np.ndarray | None = None) -> pd.DataFrame:
    """The ball of `first`, completed with the ball of `second` in the gaps of `first` that
    are longer than min_gap_frames. Both cover the same frames, e.g. one half of the tactical
    ball and of the broadcast ball moved onto the tactical timeline.

    usable: optional mask of the frames whose `second` ball may be taken (e.g. the broadcast
    shows that moment live). Column ball_from: first, second, rejected (second had a ball in
    such a gap, but `usable` said no) or none.
    """
    out = first[[c for c in BALL_COLUMNS if c in first]].copy()
    wanted = gaps_to_fill(first, second, min_gap_frames)
    take = wanted if usable is None else wanted & usable
    for column in ("ball_x", "ball_y", "ball_s"):
        if column in out and column in second:
            out[column] = out[column].astype(float)  # video 1 is float32, video 2 float64
            out.loc[take, column] = second[column].to_numpy(dtype=float)[take]
    out["ball_from"] = np.select([_seen(first), take, wanted], ["first", "second", "rejected"], "none")
    return out


def insert_halves(ball: pd.DataFrame, halves: dict[str, pd.DataFrame] | None) -> pd.DataFrame:
    """The ball of the whole match with the rows of the given halves put in, e.g. the merged
    ball per half. Column ball_from: from the halves; elsewhere first where the ball is seen."""
    out = ball.copy()
    for column in ("ball_x", "ball_y", "ball_s"):
        if column in out:
            out[column] = out[column].astype(float)
    out["ball_from"] = np.where(_seen(out), "first", "none").astype(object)
    for half in (halves or {}).values():
        for column in ("ball_x", "ball_y", "ball_s", "ball_from"):
            if column in half and column in out:
                out.loc[half.index, column] = half[column].to_numpy()
    return out


def plot_gap_grid(sources: dict, config: dict, halves: list[str] | None = None):
    """One figure: a row per half and source, a column per x axis (video frame, match minute).

    sources: {label: (ball per half, recording, offset per half)}, for example
        {"video 1 (tactical)": (tactical_halves, "tactical", {}),
         "video 2 (broadcast)": (broadcast_halves, "broadcast", data.video_2.offsets)}
    (the offsets are computed by src.synchronization, never typed in)
    halves: only these halves (default: all of the first source).
    """
    labels = list(sources)
    halves = halves or list(sources[labels[0]][0])
    rows = [(half, label) for half in halves for label in labels]
    height = 2.3 if len(rows) <= 4 else 1.6 if len(rows) <= 6 else 1.2  # still fits a 1080p screen
    fig, axes = plt.subplots(len(rows), 2, figsize=(16, height * len(rows) + 0.8), squeeze=False,
                             layout="constrained")
    for (half, label), row in zip(rows, axes):
        balls, video, offsets = sources[label]
        for ax, x_axis in zip(row, ("frame", "minute")):
            plot_frames_without_ball(balls[half], config, ax=ax, title=f"{half} | {label}", x_axis=x_axis,
                                     compact=True, video=video, video_offset=offsets.get(half, 0))
    fig.suptitle("Frames without the ball, per half and video - left over the video frame, right over the match "
                 f"minute. Dashed: {config['ball']['max_gap_s']} s, gaps up to this are filled; dot: the longest gap. "
                 "Orange: ball taken from video 2. Red / grey: long gap judged dead ball / unclear.", fontsize=10)
    return fig


def summary(ball: pd.DataFrame, config: dict) -> str:
    return "\n".join([
        "Ball per phase:",
        phase_table(ball).to_string(),
        "",
        f"Gaps in the graded phases by length (filled up to ball.max_gap_s = {config['ball']['max_gap_s']} s,"
        " if there is a sighting on both sides):",
        gap_table(ball, config).to_string(),
        "",
        speed_check(ball),
    ])


def main():
    from src.data_loader import DEFAULT_CONFIG, load_config, load_tracking
    from src.inspect_data import to_dataframe

    parser = argparse.ArgumentParser(description="Analyse the ball track and show how much of it is seen.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    df = to_dataframe(load_tracking(config), config)
    print(summary(analyze_ball(ball_dataframe(df), config), config))


if __name__ == "__main__":
    main()
