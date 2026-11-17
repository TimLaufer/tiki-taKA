"""Ordered shifts: does a team move as a block, P_i(t) = P_i(t0) + c(t)?

    python -m src.block_shift                    # block shifts per team, a plot per half
    python -m src.block_shift --save-plot blocks.png
    python -m src.visualization --blocks         # look at them on the pitch, with the shift arrows

In build-up play the ball circulates and the defending team shifts with it as one block:
everyone moves by the same c, the shape stays. So the team shifting with the ball usually
defends, and the ball is with the other team - a hint even where the ball is not seen.

Per frame t and team (`block_indicators`), comparing the positions block.window_s apart,
centred on t, for the track slots tracked at both ends:

    shift_x, shift_y    c, the common displacement: the median of the players' displacements (m)
    shift_m             |c|
    spread_m            median distance of a player's own displacement from c: 0 for a perfect block
    rigidity            shift_m / (shift_m + spread_m): 1 = a pure translation, near 0 = no common move
    players             players compared
    centre_x, centre_y  the team's centre at t (m), where the viewer draws the arrow
    block               a block shift: shift_m >= min_shift_m, rigidity >= min_rigidity,
                        players >= min_players
    follows_ball        cosine between c and the ball's displacement over the same window,
                        where the ball moved at least min_ball_move_m: 1 = with the ball

The medians make a slot that jumps to another player harmless: it is outvoted. Columns are
prefixed with the team, e.g. A_rigidity. Episodes (`block_episodes`): a team in block shift
for at least min_duration_s, numbered in time order. First guesses, not tuned; the public
ground truth is shown for understanding only (RULES.md).
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
from src.positions import players_on_pitch_m
from src.progress import Steps, say

TEAM_COLOURS = {"H": "#1d4ed8", "A": "#dc2626"}
SHIFT_COLUMNS = ["shift_x", "shift_y", "shift_m", "spread_m", "rigidity", "players", "centre_x", "centre_y"]


def _window_half(config: dict) -> int:
    return max(int(round(config["block"]["window_s"] * config["fps"] / 2)), 1)


def _displacement(values: np.ndarray, half: int) -> np.ndarray:
    """Per frame t: values[t + half] - values[t - half]; NaN where the window leaves the data."""
    out = np.full(values.shape, np.nan, dtype=np.float32)
    n = len(values)
    if n > 2 * half:
        out[half : n - half] = values[2 * half :] - values[: n - 2 * half]
    return out


def team_shifts(tracking: Tracking, config: dict) -> dict[str, pd.DataFrame]:
    """Per team: the SHIFT_COLUMNS of every frame (see the module docstring)."""
    xy = players_on_pitch_m(tracking, config["pitch"])
    half = _window_half(config)
    index = pd.RangeIndex(tracking.first_frame, tracking.first_frame + tracking.n_frames, name="frame")
    shifts = {}
    for side in "HA":
        team = xy[:, tracking.team == side, :]  # (T, S, 2)
        moved = _displacement(team, half)
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)  # frames without players: NaN
            common = np.nanmedian(moved, axis=1)  # (T, 2): c
            own = np.hypot(*(moved - common[:, None, :]).transpose(2, 0, 1))  # (T, S)
            spread = np.nanmedian(own, axis=1)
            centre = np.nanmean(team, axis=1)
            shift = np.hypot(common[:, 0], common[:, 1])
            rigidity = shift / (shift + spread)
        shifts[side] = pd.DataFrame(
            {"shift_x": common[:, 0], "shift_y": common[:, 1], "shift_m": shift, "spread_m": spread,
             "rigidity": rigidity, "players": (~np.isnan(moved[:, :, 0])).sum(axis=1),
             "centre_x": centre[:, 0], "centre_y": centre[:, 1]},
            index=index,
        )
    return shifts


def block_indicators(tracking: Tracking, ball: pd.DataFrame, config: dict) -> pd.DataFrame:
    """The block-shift indicators of every frame for both teams; ball: the analysed ball
    (data.ball, columns x_m, y_m) for follows_ball."""
    settings = config["block"]
    shifts = team_shifts(tracking, config)
    index = shifts["H"].index
    ball_xy = ball[["x_m", "y_m"]].reindex(index).to_numpy(dtype=np.float32)
    ball_moved = _displacement(ball_xy, _window_half(config))
    ball_move = np.hypot(ball_moved[:, 0], ball_moved[:, 1])
    columns = {"phase": phase_labels(tracking.first_frame + tracking.n_frames, config)[tracking.first_frame :],
               "ball_move_m": ball_move}
    for side, shift in shifts.items():
        for name in SHIFT_COLUMNS:
            columns[f"{side}_{name}"] = shift[name].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            block = ((shift["shift_m"] >= settings["min_shift_m"]) & (shift["rigidity"] >= settings["min_rigidity"])
                     & (shift["players"] >= settings["min_players"]))
            dot = shift["shift_x"].to_numpy() * ball_moved[:, 0] + shift["shift_y"].to_numpy() * ball_moved[:, 1]
            cosine = dot / (shift["shift_m"].to_numpy() * ball_move)
        moved_enough = (ball_move >= settings["min_ball_move_m"]) & (shift["shift_m"].to_numpy() > 0)
        columns[f"{side}_block"] = block.to_numpy()
        columns[f"{side}_follows_ball"] = np.where(moved_enough, cosine, np.nan)
    return pd.DataFrame(columns, index=index)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def _direction(shift_x: float, shift_y: float, team: str, half: dict, home_attacks_first_half: str) -> str:
    """across, forward or back - forward meaning towards the goal the team attacks."""
    if np.isnan(shift_x) or np.isnan(shift_y):
        return "-"
    if abs(shift_y) >= abs(shift_x):
        return "across"
    if home_attacks_first_half not in ("left_to_right", "right_to_left"):
        return "along"
    home_attacks_right = (home_attacks_first_half == "left_to_right") == (half.get("half") == 1)
    attacks_right = home_attacks_right if team == "H" else not home_attacks_right
    return "forward" if (shift_x > 0) == attacks_right else "back"


def block_episodes(ind: pd.DataFrame, tracking: Tracking, config: dict) -> pd.DataFrame:
    """Per team, the stretches inside the halves in block shift for at least min_duration_s
    (in at least half of the frames around, block.smooth_s), in time order, with the hint."""
    settings, fps = config["block"], config["fps"]
    window = max(int(round(settings["smooth_s"] * fps)), 1)
    min_length = round(settings["min_duration_s"] * fps)
    frames = ind.index.to_numpy()
    rows = []
    for half in (p for p in config["phases"] if "clock_start_min" in p):
        inside = (frames >= half["first"]) & (frames <= half["last"])
        for side in "HA":
            other = "A" if side == "H" else "H"
            share = ind[f"{side}_block"].astype(float).rolling(window, center=True, min_periods=1).mean()
            for start, stop in _runs(inside & (share >= 0.5).to_numpy()):
                if stop - start < min_length:
                    continue
                part = ind.iloc[start:stop]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    shift_x, shift_y = np.nanmedian(part[f"{side}_shift_x"]), np.nanmedian(part[f"{side}_shift_y"])
                    follows = np.nanmedian(part[f"{side}_follows_ball"])
                if np.isnan(follows):
                    hint = "ball not seen moving"
                elif follows >= 0.5:
                    hint = f"follows the ball: {other} has it"
                elif follows <= -0.5:
                    hint = "moves against the ball"
                else:
                    hint = "-"
                first, last = int(frames[start]), int(frames[stop - 1])
                rows.append({
                    "half": half["name"],
                    "team": side,
                    "match": match_clock(first, config),
                    "video": video_time(first, config),
                    "first": first,
                    "last": last,
                    "duration_s": round((stop - start) / fps, 1),
                    "direction": _direction(shift_x, shift_y, side, half, tracking.home_attacks_first_half),
                    "shift_m": round(float(np.nanmedian(part[f"{side}_shift_m"])), 1),
                    "rigidity": round(float(np.nanmedian(part[f"{side}_rigidity"])), 2),
                    "follows_ball": round(float(follows), 2) if not np.isnan(follows) else np.nan,
                    "hint": hint,
                })
    columns = ["half", "team", "match", "video", "first", "last", "duration_s", "direction", "shift_m",
               "rigidity", "follows_ball", "hint"]
    table = pd.DataFrame(rows, columns=columns).sort_values(["first", "team"], ignore_index=True)
    table.insert(0, "id", np.arange(1, len(table) + 1))
    table["summary"] = [f"{t} shifts as a block {d}, {s} m per {settings['window_s']:g} s, rigidity {r}; {h}"
                        for t, d, s, r, h in zip(table["team"], table["direction"], table["shift_m"],
                                                 table["rigidity"], table["hint"])]
    return table


def ground_truth_majority(table: pd.DataFrame, gt: np.ndarray) -> list[str]:
    """Per stretch: the most common public ground-truth label and its share ("" outside it)."""
    out = []
    for first, last in zip(table["first"], table["last"]):
        labels = gt[first : last + 1]
        labels = labels[np.isin(labels, ["H", "A", "D"])]
        if len(labels) == 0:
            out.append("")
            continue
        values, counts = np.unique(labels, return_counts=True)
        out.append(f"{values[counts.argmax()]} {counts.max() / len(labels):.0%}")
    return out


def ground_truth_crosstab(ind: pd.DataFrame, gt: np.ndarray) -> pd.DataFrame:
    """In the frames the public ground truth labels: what it says while each team shifts as a
    block (rows) - is the block team the one without the ball? For understanding only."""
    labels = gt[ind.index.to_numpy()]
    graded = np.isin(labels, ["H", "A", "D"])
    rows = {}
    for side in "HA":
        frames = graded & ind[f"{side}_block"].to_numpy()
        counts = pd.Series(labels[frames]).value_counts().reindex(["H", "A", "D"], fill_value=0)
        rows[f"{side} in block shift"] = {"frames": int(frames.sum()),
                                          **{f"gt {k}": f"{v / max(frames.sum(), 1):.0%}" for k, v in counts.items()}}
    return pd.DataFrame(rows).T


def plot_blocks(ind: pd.DataFrame, table: pd.DataFrame, config: dict, gt: np.ndarray | None = None):
    """Per half two panels: the block's shift (m per window) and its rigidity, per team, with
    the episodes shaded in the team's colour and (first minutes) the public ground truth."""
    settings = config["block"]
    halves = [p for p in config["phases"] if "clock_start_min" in p]
    window = max(int(round(settings["smooth_s"] * config["fps"])), 1)
    fig, axes = plt.subplots(2 * len(halves), 1, figsize=(15, 2.4 * 2 * len(halves) + 0.6), squeeze=False,
                             layout="constrained")
    for h, half in enumerate(halves):
        frames = ind.loc[half["first"] : half["last"]].index.to_numpy()
        minutes = match_seconds(frames, config) / 60
        for ax, column, threshold, label in ((axes[2 * h, 0], "shift_m", settings["min_shift_m"],
                                              f"shift (m per {settings['window_s']:g} s)"),
                                             (axes[2 * h + 1, 0], "rigidity", settings["min_rigidity"], "rigidity")):
            minute_axes(ax, frames, minutes, config)
            for side, colour in TEAM_COLOURS.items():
                values = ind.loc[frames, f"{side}_{column}"].rolling(window, center=True, min_periods=1).median()
                ax.plot(minutes, values, color=colour, lw=0.7, label=side)
                for episode in table[(table["half"] == half["name"]) & (table["team"] == side)].itertuples():
                    span = match_seconds(np.array([episode.first, episode.last]), config) / 60
                    ax.axvspan(*span, color=colour, alpha=0.15, lw=0)
            ax.axhline(threshold, color="#374151", ls="--", lw=0.8)
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0)
            ax.set_title(f"{half['name']} | {label}", loc="left", fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
        if gt is not None:  # a strip at the bottom of the shift panel: what the ground truth says
            ax = axes[2 * h, 0]
            labels = gt[frames]
            for label, colour in (("H", TEAM_COLOURS["H"]), ("A", TEAM_COLOURS["A"]), ("D", "#6b7280")):
                where = labels == label
                if where.any():
                    ax.fill_between(minutes, 0, 0.06, where=where, transform=ax.get_xaxis_transform(),
                                    color=colour, lw=0, step="mid")
    fig.suptitle("Block shifts per team: shaded where a team shifts as a block (in its colour). "
                 "Strip at the bottom: public ground truth H / A / D (for understanding only).", fontsize=10)
    return fig


def main():
    from src.data_loader import DEFAULT_CONFIG, load_config
    from src.pipeline import load_match, load_public_gt, measured

    parser = argparse.ArgumentParser(description="Ordered shifts: teams moving as a block.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--save-plot", help="write the plot to this file instead of a window")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["block"]
    steps = Steps(9)

    data = load_match(config, steps)
    with steps(f"Block shifts: each team's common shift over {settings['window_s']:g} s, frame by frame"):
        ind = block_indicators(measured(data), data.ball, config)
    with steps("The block-shift episodes inside the halves"):
        table = block_episodes(ind, data.tracking, config)
    gt = load_public_gt(config, data.tracking.n_frames)

    halves = [p["name"] for p in config["phases"] if "clock_start_min" in p]
    in_halves = ind[ind["phase"].isin(halves)]
    print("\nShare of the time in the halves a team shifts as a block: "
          + ", ".join(f"{side} {in_halves[f'{side}_block'].mean():.1%}" for side in "HA"))
    if gt is not None:
        table["public_gt"] = ground_truth_majority(table, gt)
        print("\nWhile a team shifts as a block, the public ground truth says (understanding only, never tune on it):")
        print(ground_truth_crosstab(in_halves, gt).to_string())
    print(f"\n{len(table)} block-shift episodes (>= {settings['min_shift_m']:g} m per {settings['window_s']:g} s, "
          f"rigidity >= {settings['min_rigidity']:g}, >= {settings['min_duration_s']:g} s), in time order:")
    print(table.drop(columns="summary").to_string(index=False))
    print("\nlook at them on the pitch: .venv/bin/python -m src.visualization --block <id>   (then n / p)")

    fig = plot_blocks(ind, table, config, gt)
    if args.save_plot:
        fig.savefig(args.save_plot, dpi=120)
        print(f"wrote {args.save_plot}")
    else:
        say("the plot window is open - close it to finish")
        plt.show()
    return ind, table


if __name__ == "__main__":
    ind, table = main()
