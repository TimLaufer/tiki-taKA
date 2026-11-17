"""Dead-ball indicators from the players: how crowded they stand, and how fast they move.

    python -m src.dead_ball                       # the densest moments of the match, a plot per half
    python -m src.dead_ball --save-plot crowd.png

At many stoppages the players crowd together - corner, free kick near the box with its wall,
goal celebration, cooling break at the bench - and at nearly all of them they slow down to a
walk. Crowding alone misses throw-ins, goal kicks and quick free kicks; the speed is there for
those. Per video frame (`indicators`):

    crowd        the most players within dead_ball.radius_m of one player, that player included
    crowd_share  crowd / players on the pitch
    crowd_x, _y  where that densest spot is, in metres
    players      players tracked on the pitch
    speed_ms     median speed of the players on the pitch, in m/s, from their positions

Episodes (`episodes`): stretches inside the halves where the crowd (smoothed) reaches
crowd_min_players for at least min_duration_s - numbered densest first (id), with match clock,
video time and zone. Look at them on the pitch with `python -m src.visualization --episode <id>`
(keys n / p for the next / previous one), or in the video at the video time.

The thresholds are a first guess to look at, not tuned. The public ground truth is drawn only
to understand what the labels mean: tuning on it is against the rules (RULES.md).
"""
from __future__ import annotations

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ball_analysis import minute_axes
from src.data_loader import Tracking
from src.phases import match_clock, match_seconds, phase_labels, video_time
from src.positions import pack_filled, players_on_pitch_m
from src.progress import Steps, say

BLOCK = 4096  # frames per vectorised step: bounds the memory of the pairwise distances
PENALTY_BOX_DEPTH, PENALTY_BOX_HALF_WIDTH, CENTRE_CIRCLE_RADIUS = 16.5, 20.16, 9.15
GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH, PENALTY_SPOT_DISTANCE = 5.5, 9.16, 11.0
TOUCHLINE_BAND_M = 5.0
# where a ball last seen before a long gap says it went out, and where a ball seen again says restart
OUT_ZONES = ("out of play", "touchline", "goal line", "corner")
RESTART_ZONES = ("corner", "goal area", "centre spot", "penalty spot", "touchline")


def crowding(tracking: Tracking, pitch, radius_m: float) -> pd.DataFrame:
    """Per frame: the most players within radius_m of one player (crowd, that player included),
    where that densest spot is (crowd_x, crowd_y in metres), and the players on the pitch."""
    xy = pack_filled(players_on_pitch_m(tracking, pitch))
    n = len(xy)
    crowd = np.zeros(n, dtype=np.int16)
    centre = np.full((n, 2), np.nan, dtype=np.float32)
    for start in range(0, n, BLOCK):
        part = xy[start : start + BLOCK]  # (C, K, 2), NaN for empty slots
        d2 = ((part[:, :, None, :] - part[:, None, :, :]) ** 2).sum(axis=-1)  # (C, K, K)
        near = (d2 <= radius_m**2).sum(axis=2)  # per player: players within the radius (NaN never counts)
        best = near.argmax(axis=1)
        rows = np.arange(len(part))
        crowd[start : start + len(part)] = near[rows, best]
        centre[start : start + len(part)] = part[rows, best]
    return pd.DataFrame(
        {"crowd": crowd, "crowd_x": centre[:, 0], "crowd_y": centre[:, 1],
         "players": (~np.isnan(xy[:, :, 0])).sum(axis=1)},
        index=pd.RangeIndex(tracking.first_frame, tracking.first_frame + n, name="frame"),
    )


def median_speed(tracking: Tracking, pitch, fps: float, window_s: float) -> np.ndarray:
    """Per frame: the median speed (m/s) of the players on the pitch, each from its positions
    window_s before and after. The median keeps single tracks that jump to another person out."""
    xy = players_on_pitch_m(tracking, pitch)
    k = max(int(round(window_s * fps)), 1)
    speed = np.full(xy.shape[:2], np.nan, dtype=np.float32)
    if len(xy) > 2 * k:
        speed[k:-k] = np.hypot(*(xy[2 * k :] - xy[: -2 * k]).transpose(2, 0, 1)) / (2 * k / fps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames without any player: NaN
        return np.nanmedian(speed, axis=1)


def indicators(tracking: Tracking, config: dict) -> pd.DataFrame:
    """The dead-ball indicators of every frame, see the module docstring."""
    settings, pitch, fps = config["dead_ball"], config["pitch"], config["fps"]
    ind = crowding(tracking, pitch, settings["radius_m"])
    ind.insert(0, "phase", phase_labels(tracking.first_frame + tracking.n_frames, config)[tracking.first_frame :])
    ind.insert(2, "crowd_share", ind["crowd"] / ind["players"].where(ind["players"] > 0))
    ind["speed_ms"] = median_speed(tracking, pitch, fps, settings["speed_window_s"])
    return ind


def smoothed(ind: pd.DataFrame, column: str, config: dict) -> pd.Series:
    window = max(int(round(config["dead_ball"]["smooth_s"] * config["fps"])), 1)
    return ind[column].rolling(window, center=True, min_periods=1).median()


def zone(x: float, y: float, half: dict, home_attacks_first_half: str, pitch) -> str:
    """Where on the pitch, in words. "penalty box H" is the box H defends."""
    length, width = pitch
    if np.isnan(x) or np.isnan(y):
        return "-"
    if home_attacks_first_half in ("left_to_right", "right_to_left"):
        # attacking to the right means defending the left goal; the sides swap at halftime
        home_defends_left = (home_attacks_first_half == "left_to_right") == (half.get("half") == 1)
        left, right = ("H", "A") if home_defends_left else ("A", "H")
    else:
        left, right = "left", "right"
    in_box_band = abs(y - width / 2) <= PENALTY_BOX_HALF_WIDTH
    if in_box_band and x <= PENALTY_BOX_DEPTH:
        return f"penalty box {left}"
    if in_box_band and x >= length - PENALTY_BOX_DEPTH:
        return f"penalty box {right}"
    if y <= TOUCHLINE_BAND_M or y >= width - TOUCHLINE_BAND_M:
        return "touchline"
    if np.hypot(x - length / 2, y - width / 2) <= CENTRE_CIRCLE_RADIUS:
        return "centre circle"
    return f"half {left}" if x < length / 2 else f"half {right}"


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) positions of the stretches where mask is True, stop exclusive."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def episodes(ind: pd.DataFrame, tracking: Tracking, config: dict) -> pd.DataFrame:
    """Stretches inside the halves where the smoothed crowd reaches crowd_min_players for at
    least min_duration_s; densest first, then longest."""
    settings, fps = config["dead_ball"], config["fps"]
    crowd = smoothed(ind, "crowd", config).to_numpy()
    min_length = round(settings["min_duration_s"] * fps)
    frames = ind.index.to_numpy()
    rows = []
    for half in (p for p in config["phases"] if "clock_start_min" in p):
        inside = (frames >= half["first"]) & (frames <= half["last"])
        for start, stop in _runs(inside & (crowd >= settings["crowd_min_players"])):
            if stop - start < min_length:
                continue
            part = ind.iloc[start:stop]
            first, last = int(frames[start]), int(frames[stop - 1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                x, y = np.nanmedian(part["crowd_x"]), np.nanmedian(part["crowd_y"])
                speed = np.nanmedian(part["speed_ms"])
            rows.append({
                "half": half["name"],
                "match": match_clock(first, config),
                "video": video_time(first, config),
                "first": first,
                "last": last,
                "duration_s": round((stop - start) / fps, 1),
                "peak": int(part["crowd"].max()),
                "share": round(float(part["crowd_share"].mean()), 2),
                "zone": zone(x, y, half, tracking.home_attacks_first_half, config["pitch"]),
                "speed_ms": round(float(speed), 1),
            })
    columns = ["half", "match", "video", "first", "last", "duration_s", "peak", "share", "zone", "speed_ms"]
    table = pd.DataFrame(rows, columns=columns)
    # densest first, then longest, then earliest: the same order, so the same ids, on every run
    table = table.sort_values(["peak", "duration_s", "first"], ascending=[False, False, True], ignore_index=True)
    table.insert(0, "id", np.arange(1, len(table) + 1))
    return table


def ball_zone(x_m: float, y_m: float, pitch, margin_m: float) -> str:
    """Where a ball lies, in the words of the restarts: out of play, corner, touchline, goal
    line, goal area (goal kick), centre spot (kickoff), penalty spot - or in play."""
    length, width = pitch
    if np.isnan(x_m) or np.isnan(y_m):
        return "-"
    if x_m < -margin_m or x_m > length + margin_m or y_m < -margin_m or y_m > width + margin_m:
        return "out of play"
    on_goal_line = x_m <= margin_m or x_m >= length - margin_m
    on_touchline = y_m <= margin_m or y_m >= width - margin_m
    if on_goal_line and on_touchline:
        return "corner"
    if on_touchline:
        return "touchline"
    if on_goal_line:
        return "goal line"
    if abs(y_m - width / 2) <= GOAL_AREA_HALF_WIDTH and (x_m <= GOAL_AREA_DEPTH or x_m >= length - GOAL_AREA_DEPTH):
        return "goal area"
    if np.hypot(x_m - length / 2, y_m - width / 2) <= margin_m:
        return "centre spot"
    spots = (PENALTY_SPOT_DISTANCE, length - PENALTY_SPOT_DISTANCE)
    if any(np.hypot(x_m - spot, y_m - width / 2) <= margin_m for spot in spots):
        return "penalty spot"
    return "in play"


def long_gaps(ball: pd.DataFrame, ind: pd.DataFrame, crowd_episodes: pd.DataFrame, config: dict) -> pd.DataFrame:
    """The stretches inside the halves where the analysed ball (data.ball) is still missing and
    longer than ball.max_gap_s, in time order, with the evidence for a dead ball and a verdict.

    Evidence: the players walk (median speed at most dead_ball.slow_speed_ms), they crowd (a crowd
    episode overlaps), the ball was last seen out of play or on a line, the ball is seen again at
    a restart spot. Any of them: "dead"; none: "unclear" - play probably goes on unseen.
    """
    settings, fps, pitch = config["dead_ball"], config["fps"], config["pitch"]
    margin = settings["line_margin_m"]
    min_length = round(config["ball"]["max_gap_s"] * fps)
    frames = ball.index.to_numpy()
    missing = (ball["source"] == "missing").to_numpy()
    x_m, y_m = ball["x_m"].to_numpy(dtype=float), ball["y_m"].to_numpy(dtype=float)
    crowd = ind["crowd"].reindex(ball.index).to_numpy(dtype=float)
    speed = ind["speed_ms"].reindex(ball.index).to_numpy(dtype=float)
    episode_of = np.zeros(len(frames), dtype=int)
    for episode in crowd_episodes.itertuples():
        episode_of[np.searchsorted(frames, episode.first) : np.searchsorted(frames, episode.last, "right")] = episode.id

    rows = []
    for half in (p for p in config["phases"] if "clock_start_min" in p):
        inside = (frames >= half["first"]) & (frames <= half["last"])
        for start, stop in _runs(inside & missing):
            if stop - start <= min_length:
                continue
            seen_before = start > 0 and inside[start - 1]
            seen_after = stop < len(frames) and inside[stop]
            before = ball_zone(x_m[start - 1], y_m[start - 1], pitch, margin) if seen_before else "-"
            after = ball_zone(x_m[stop], y_m[stop], pitch, margin) if seen_after else "-"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                walking = float(np.nanmedian(speed[start:stop]))
                peak = np.nanmax(crowd[start:stop])
            crowded = sorted(set(episode_of[start:stop]) - {0})
            reasons = []
            if walking <= settings["slow_speed_ms"]:
                reasons.append(f"players walk ({walking:.1f} m/s)")
            if crowded:
                reasons.append(f"players crowd (episode {', '.join(map(str, crowded))})")
            if before in OUT_ZONES:
                reasons.append(f"ball went out ({before})")
            if after in RESTART_ZONES:
                reasons.append(f"restart at {after}")
            first, last = int(frames[start]), int(frames[stop - 1])
            rows.append({
                "half": half["name"],
                "match": match_clock(first, config),
                "video": video_time(first, config),
                "first": first,
                "last": last,
                "duration_s": round((stop - start) / fps, 1),
                "speed_ms": round(walking, 1),
                "crowd_peak": int(peak) if not np.isnan(peak) else 0,
                "crowded": bool(crowded),
                "ball_before": before,
                "ball_after": after,
                "verdict": "dead" if reasons else "unclear",
                "reasons": "; ".join(reasons),
            })
    columns = ["half", "match", "video", "first", "last", "duration_s", "speed_ms", "crowd_peak", "crowded",
               "ball_before", "ball_after", "verdict", "reasons"]
    table = pd.DataFrame(rows, columns=columns)
    table.insert(0, "id", np.arange(1, len(table) + 1))
    table["summary"] = [f"{v}: {r}" if r else f"{v}: no sign of a stoppage" for v, r in zip(table["verdict"],
                                                                                               table["reasons"])]
    return table


def gap_summary(gaps: pd.DataFrame) -> pd.DataFrame:
    """Per half and verdict: how many long gaps, and how long together."""
    if gaps.empty:
        return pd.DataFrame(columns=["half", "verdict", "gaps", "seconds"])
    return (gaps.groupby(["half", "verdict"], sort=False)
            .agg(gaps=("id", "size"), seconds=("duration_s", "sum")).round(1).reset_index())


def ground_truth_share(table: pd.DataFrame, gt: np.ndarray) -> list[str]:
    """Per episode: the share of its graded frames the public ground truth calls D ("" outside it)."""
    shares = []
    for first, last in zip(table["first"], table["last"]):
        labels = gt[first : last + 1]
        graded = np.isin(labels, ["H", "A", "D"])
        shares.append(f"{(labels[graded] == 'D').mean():.0%}" if graded.any() else "")
    return shares


def plot_indicators(ind: pd.DataFrame, table: pd.DataFrame, config: dict, gt: np.ndarray | None = None):
    """Per half: the crowd at the densest spot and the players' median speed over the match minute,
    the episodes shaded red, and (first minutes only) where the public ground truth says D."""
    settings = config["dead_ball"]
    halves = [p for p in config["phases"] if "clock_start_min" in p]
    crowd, speed = smoothed(ind, "crowd", config), smoothed(ind, "speed_ms", config)
    fig, axes = plt.subplots(len(halves), 1, figsize=(15, 3.8 * len(halves)), squeeze=False, layout="constrained")
    for ax, half in zip(axes[:, 0], halves):
        frames = ind.loc[half["first"] : half["last"]].index.to_numpy()
        minutes = match_seconds(frames, config) / 60
        minute_axes(ax, frames, minutes, config)
        lines = ax.plot(minutes, crowd.loc[frames], color="#1d4ed8", lw=0.8,
                        label=f"players within {settings['radius_m']:g} m of the densest spot")
        ax.axhline(settings["crowd_min_players"], color="#1d4ed8", ls="--", lw=0.8, alpha=0.6)
        ax.set_ylabel(f"players within {settings['radius_m']:g} m")
        ax.set_ylim(bottom=0)
        twin = ax.twinx()
        lines += twin.plot(minutes, speed.loc[frames], color="#f59e0b", lw=0.8, label="median player speed (m/s)")
        twin.set_ylabel("median speed (m/s)")
        twin.set_ylim(bottom=0)
        for episode in table[table["half"] == half["name"]].itertuples():
            span = match_seconds(np.array([episode.first, episode.last]), config) / 60
            ax.axvspan(*span, color="#dc2626", alpha=0.15, lw=0)
        if gt is not None:
            dead = gt[frames] == "D"
            if dead.any():
                lines.append(ax.fill_between(minutes, 0, 1, where=dead, transform=ax.get_xaxis_transform(),
                                             color="#6b7280", alpha=0.35, lw=0, step="mid",
                                             label="public ground truth: D"))
        n = int((table["half"] == half["name"]).sum())
        ax.set_title(half["name"], loc="left")
        ax.set_title(f"{n} episode{'' if n == 1 else 's'} (red): >= {settings['crowd_min_players']} players within "
                     f"{settings['radius_m']:g} m for >= {settings['min_duration_s']:g} s", loc="right", fontsize=9)
        twin.legend(handles=lines, loc="upper left", fontsize=8, framealpha=0.9)  # on the twin: drawn on top
    return fig


def main():
    from src.data_loader import DEFAULT_CONFIG, PROJECT_ROOT, load_config, load_tracking
    from src.visualization import read_labels

    parser = argparse.ArgumentParser(description="Dead-ball indicators: crowding and speed of the players.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--save-plot", help="write the plot to this file instead of a window")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["dead_ball"]
    steps = Steps(3)

    with steps("Loading the tracking"):
        tracking = load_tracking(config)
    with steps(f"Crowding within {settings['radius_m']:g} m and the players' speed, frame by frame"):
        ind = indicators(tracking, config)
    with steps("The densest moments inside the halves"):
        table = episodes(ind, tracking, config)

    gt = None
    gt_path = PROJECT_ROOT / config.get("public_gt", "")
    if config.get("public_gt") and gt_path.is_file():
        gt = read_labels(gt_path, tracking.n_frames)
        table["public_gt_D"] = ground_truth_share(table, gt)
    else:
        say(f"no public ground truth at {gt_path}: drawn without it")

    in_halves = ind[ind["phase"].isin([p["name"] for p in config["phases"] if "clock_start_min" in p])]
    crowd, speed = in_halves["crowd"], in_halves["speed_ms"]
    print(f"\nIn the halves: players within {settings['radius_m']:g} m of the densest spot: median "
          f"{crowd.median():.0f}, 90th percentile {crowd.quantile(0.9):.0f}, max {crowd.max()}; "
          f"median player speed: median {speed.median():.1f} m/s, 10th percentile {speed.quantile(0.1):.1f} m/s")
    print(f"\n{len(table)} episodes with >= {settings['crowd_min_players']} players within {settings['radius_m']:g} m "
          f"for >= {settings['min_duration_s']:g} s; the {min(settings['top'], len(table))} densest "
          "(zone 'penalty box H' = the box H defends):")
    print(table.head(settings["top"]).to_string(index=False))
    print("\nlook at them on the pitch: .venv/bin/python -m src.visualization --episode <id>"
          "   (then n / p: next / previous episode)")

    fig = plot_indicators(ind, table, config, gt)
    if args.save_plot:
        fig.savefig(args.save_plot, dpi=120)
        print(f"wrote {args.save_plot}")
    else:
        say("the plot window is open - close it to finish")
        plt.show()
    return ind, table


if __name__ == "__main__":
    ind, table = main()
