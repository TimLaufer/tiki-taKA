"""The dead balls of both logics over time: Stefan's (src/stoppages.py, as run_pipeline.py uses
it) against Tim's (src/segments.py, as src.baseline uses it), and the public ground truth's D
where it has one.

    python -m src.dead_ball_compare                        # one window per half
    python -m src.dead_ball_compare --save-plot dead.png   # dead_first_half.png, dead_second_half.png

Per half a row per source, a column per x axis (video frame, match minute). Each dead ball is a
bar in the colour of its kind; live play stays white. The kind, the same way for both logics:

    1. a restart of src.restarts (verdict restart) played at the end of the dead ball: its kind
    2. else where the ball is first seen within kind_window_s after it: corner flag, goal
       area, touchline, centre spot or penalty spot (src.restarts.restart_kind)
    3. else "other" (a free kick no restart was found for, a drinks break, an injury ...)

The ground truth has no kind: its D is grey. The terminal gets per half how many dead balls and
minutes each source has, and what only one of them calls dead. The public ground truth is for
understanding only - never tune on it.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ball_analysis import minute_axes
from src.dead_ball import _runs
from src.phases import match_seconds
from src.progress import Steps, say

# one colour per kind, in the validated categorical order of the dataviz reference palette
KIND_COLOURS = {"throw-in": "#2a78d6", "corner": "#eb6834", "goal kick": "#1baf7a", "free kick": "#eda100",
                "kickoff": "#e87ba4", "penalty": "#008300", "other": "#6250d6", "D": "#6b7280"}
KIND_WINDOW_S = 1.0  # where the ball is first seen this long after a dead ball tells its kind
# the two dead-ball logics, named by how they work (Stefan: gap-based, Tim: restart-based)
STEFAN, TIM, TRUTH = "gap-based (stoppages.py)", "restart-based (segments.py)", "public ground truth"


def halves(config: dict) -> list[dict]:
    return [p for p in config["phases"] if "clock_start_min" in p]


def tim_dead(n_frames: int, events: pd.DataFrame, config: dict) -> np.ndarray:
    """Tim's dead balls as a frame mask: every event from where play stopped to where it is played."""
    dead = np.zeros(n_frames, dtype=bool)
    for event in events.itertuples():
        dead[event.start : event.played + 1] = True
    return dead


def dead_balls(mask: np.ndarray, restarts: pd.DataFrame, ball: pd.DataFrame, half: dict,
               config: dict) -> list[tuple[int, int, str]]:
    """(first, stop, kind) of every dead ball of a mask in a half; kind as the module docstring says."""
    from src.restarts import restart_kind

    fps = config["fps"]
    window, merge = int(round(KIND_WINDOW_S * fps)), int(round(config["segments"]["merge_s"] * fps))
    played = restarts.set_index("last")["kind"] if len(restarts) else pd.Series(dtype=str)
    x, y = ball["x_m"].to_numpy(dtype=float), ball["y_m"].to_numpy(dtype=float)
    out = []
    for a, b in _runs(mask[half["first"] : half["last"] + 1]):
        first, stop = half["first"] + a, half["first"] + b
        near = played[(played.index >= stop - 1 - merge) & (played.index <= stop - 1 + window)]
        kind = near.iloc[0] if len(near) else "other"
        if kind not in KIND_COLOURS or kind == "D":
            kind = "other"
        if not len(near):
            seen = [f for f in range(stop, min(stop + window, len(x))) if np.isfinite(x[f])]
            if seen:
                spot = restart_kind(x[seen[0]], y[seen[0]], config["pitch"], config["restarts"])
                # a penalty only from a restart that showed its scene, never from where the ball turns up
                kind = spot if spot in ("corner", "goal kick", "throw-in", "kickoff") else "other"
        out.append((first, stop, kind))
    return out


COMBINED = "combined"
TIM_KINDS = ("throw-in", "free kick", "corner", "penalty")  # where Tim's recognition is the better one


def combine(stefan: list[tuple[int, int, str]], tim: list[tuple[int, int, str]],
            n_frames: int) -> list[tuple[int, int, str, str]]:
    """Both recognitions judged, per stretch either of them calls dead: (first, stop, kind, source).

    1. Stefan says goal kick: a goal kick, Stefan's span (Tim takes many of them for penalties)
    2. Tim says throw-in, free kick, corner or penalty: Tim's kind and span
    3. both say the same: that kind, the union of their spans
    4. else the side with a kind beats "other"; with only one side, that one
    """
    any_dead = np.zeros(n_frames, dtype=bool)
    for first, stop, _ in stefan + tim:
        any_dead[first:stop] = True
    out = []
    for first, stop in _runs(any_dead):
        s = [bar for bar in stefan if bar[0] < stop and bar[1] > first]
        t = [bar for bar in tim if bar[0] < stop and bar[1] > first]
        s_kinds, t_kinds = {k for *_, k in s}, {k for *_, k in t}
        if "goal kick" in s_kinds:
            out += [(f, e, "goal kick", "Stefan") for f, e, _ in s]
        elif t_kinds & set(TIM_KINDS):
            out += [(f, e, k, "Tim") for f, e, k in t]
        elif s and t and s_kinds == t_kinds:
            out.append((int(first), int(stop), next(iter(s_kinds)), "both"))
        elif t and (t_kinds != {"other"} or not s):
            out += [(f, e, k, "Tim") for f, e, k in t]
        else:
            out += [(f, e, k, "Stefan") for f, e, k in s]
    return sorted(out)


def kinds_per_frame(bars: list[tuple[int, int, str]], n_frames: int) -> np.ndarray:
    """Per frame the kind of the dead ball it lies in, "" in live play."""
    out = np.full(n_frames, "", dtype=object)
    for first, stop, kind in bars:
        out[first:stop] = kind
    return out


def dead_ball_table(bars: dict[str, list[tuple[int, int, str]]], config: dict) -> pd.DataFrame:
    """The viewer's list: every stretch any source calls dead, in time order (id, first, last,
    duration_s, match, summary), the summary naming each source's kind and length there."""
    from src.phases import match_clock

    fps = config["fps"]
    n = max((stop for spans in bars.values() for _, stop, _ in spans), default=0)
    any_dead = np.zeros(n, dtype=bool)
    for spans in bars.values():
        for first, stop, _ in spans:
            any_dead[first:stop] = True
    rows = []
    for first, stop in _runs(any_dead):
        parts = []
        for name, spans in bars.items():
            mine = [(f, s, k) for f, s, k in spans if f < stop and s > first]
            parts.append(f"{name.split(' ')[0]}: " + (", ".join(f"{k} {(s - f) / fps:.1f} s" for f, s, k in mine)
                                                      if mine else "live"))
        rows.append({"first": int(first), "last": int(stop - 1), "duration_s": round((stop - first) / fps, 1),
                     "match": match_clock(int(first), config), "summary": " | ".join(parts)})
    table = pd.DataFrame(rows, columns=["first", "last", "duration_s", "match", "summary"])
    table.insert(0, "id", np.arange(1, len(table) + 1))
    return table


def judged_dead_balls(data, config: dict):
    """(bars, tim_events, stefan_seeds): Stefan's and Tim's dead balls with their kinds and the
    combined (judged) ones, as name -> [(first, stop, kind)]; Tim's events (src.segments, with
    after_team) and Stefan's restart seeds (src.stoppages: start of live play, team)."""
    from src.pipeline import dead_ball_analysis, measured
    from src.preprocessing import preprocess
    from src.restarts import candidates
    from src.segments import find_segments

    prepared = preprocess(data, config)
    events, _ = find_segments(data, config, dead_ball_analysis(data, config))
    tim = tim_dead(data.tracking.n_frames, events, config)
    still = candidates(measured(data), data.ball, config)
    restarts = still[still["verdict"] == "restart"]
    bars = {name: [bar for half in halves(config) for bar in dead_balls(mask, restarts, data.ball, half, config)]
            for name, mask in ((STEFAN, prepared.dead.copy()), (TIM, tim))}
    bars[COMBINED] = [(f, e, k) for f, e, k, _ in combine(bars[STEFAN], bars[TIM], data.tracking.n_frames)]
    return bars, events, prepared.seeds


def dead_balls_of_match(data, config: dict) -> dict[str, list[tuple[int, int, str]]]:
    """Stefan's, Tim's and the combined dead balls with their kinds, for the viewer and the plot."""
    return judged_dead_balls(data, config)[0]


def live_after(spans: list[tuple[int, int, str]], tim_events: pd.DataFrame, stefan_seeds: pd.DataFrame,
               config: dict) -> pd.DataFrame:
    """The live stretches between judged dead balls inside the halves (first, last, team): the
    team at a stretch's start is Tim's after_team where Tim's event is played right before it,
    else Stefan's restart seed starting there, else "" (the first confirmed control decides)."""
    tim_team = {int(e.played) + 1: e.after_team for e in tim_events.itertuples()}
    stefan_team = {int(s.start): s.team for s in stefan_seeds.itertuples()}
    rows = []
    for half in halves(config):
        begin = half["first"]
        for first, stop, _ in sorted(s for s in spans if s[0] <= half["last"] and s[1] > half["first"]):
            if first > begin:
                rows.append((begin, first - 1))
            begin = max(begin, stop)
        if begin <= half["last"]:
            rows.append((begin, half["last"]))
    return pd.DataFrame([{"first": a, "last": b, "team": tim_team.get(a) or stefan_team.get(a) or ""}
                         for a, b in rows], columns=["first", "last", "team"])


def compare(masks: dict[str, np.ndarray], config: dict) -> pd.DataFrame:
    """Per half and source: dead balls, minutes dead; and how much Stefan and Tim share."""
    fps, rows = config["fps"], []
    for half in halves(config):
        part = slice(half["first"], half["last"] + 1)
        for name, mask in masks.items():
            inside = mask[part]
            rows.append({"half": half["name"], "source": name, "dead balls": len(_runs(inside)),
                         "minutes dead": round(inside.sum() / fps / 60, 1),
                         "share of the half": f"{inside.mean():.1%}"})
        if STEFAN in masks and TIM in masks:
            stefan, tim = masks[STEFAN][part], masks[TIM][part]
            both, either = (stefan & tim).sum(), (stefan | tim).sum()
            rows.append({"half": half["name"], "source": "both (overlap / either)",
                         "dead balls": "", "minutes dead": round(both / fps / 60, 1),
                         "share of the half": f"{both / max(either, 1):.0%} of either"})
            for name, only in (("only Stefan", stefan & ~tim), ("only Tim", tim & ~stefan)):
                rows.append({"half": half["name"], "source": name, "dead balls": len(_runs(only)),
                             "minutes dead": round(only.sum() / fps / 60, 1), "share of the half": ""})
    return pd.DataFrame(rows)


def against_truth(masks: dict[str, np.ndarray], gt: np.ndarray) -> pd.DataFrame:
    """For understanding only: in the frames the public ground truth grades, how much of its D
    each source finds (recall) and how much of its own D is D there (precision)."""
    graded = np.isin(gt, ["H", "A", "D"])
    truth = gt[graded] == "D"
    rows = {}
    for name, mask in masks.items():
        if name == TRUTH:
            continue
        ours = mask[graded]
        rows[name] = {"D recall": f"{(ours & truth).sum() / max(truth.sum(), 1):.0%}",
                      "D precision": f"{(ours & truth).sum() / max(ours.sum(), 1):.0%}"}
    return pd.DataFrame(rows).T


def plot_half(bars: dict[str, list[tuple[int, int, str]]], half: dict, config: dict):
    """One figure for a half: a row per source (its dead balls as (first, stop, kind)), columns
    video frame and match minute."""
    from matplotlib.patches import Patch

    inside = {name: [(f, s, k) for f, s, k in spans if f <= half["last"] and s > half["first"]]
              for name, spans in bars.items()}
    shown = {name: spans for name, spans in inside.items() if spans or name != TRUTH}
    frames = np.arange(half["first"], half["last"] + 1)
    minutes = match_seconds(frames, config) / 60
    fig, axes = plt.subplots(len(shown), 2, figsize=(16, 1.9 * len(shown) + 1.6), squeeze=False,
                             gridspec_kw={"wspace": 0.08, "hspace": 1.6})
    for row, (name, spans) in zip(axes, shown.items()):
        for ax, x_axis in zip(row, ("frame", "minute")):
            for first, stop, kind in spans:
                first, stop = max(first, half["first"]), min(stop, half["last"] + 1)
                if x_axis == "frame":
                    x, width = first, stop - first
                else:
                    x, width = minutes[first - half["first"]], (stop - first) / config["fps"] / 60
                ax.broken_barh([(x, width)], (0.15, 0.7), facecolors=KIND_COLOURS.get(kind, KIND_COLOURS["other"]),
                               linewidth=0)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            for side in ("left", "right", "top"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{half['name']} | {name}", loc="left", fontsize=10)
            if x_axis == "frame":
                ax.set_xlim(half["first"], half["last"])
                ax.set_xlabel("video frame")
            else:
                minute_axes(ax, frames, minutes, config)
    handles = [Patch(facecolor=colour, label=kind if kind != "D" else "D (ground truth, no kind)")
               for kind, colour in KIND_COLOURS.items()] + [Patch(facecolor="white", edgecolor="#9ca3af",
                                                                  label="live play")]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Dead balls, {half['name']}, coloured by kind - white: live play", fontsize=11, x=0.01, ha="left")
    return fig


def _mask(spans: list[tuple[int, int, str]], n_frames: int) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for first, stop, _ in spans:
        mask[first:stop] = True
    return mask


def main():
    from src.data_loader import DEFAULT_CONFIG, load_config
    from src.pipeline import load_match, load_public_gt

    parser = argparse.ArgumentParser(description="The dead balls of Stefan's and Tim's logic over time.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--save-plot", help="write the plots to <stem>_<half>.png instead of windows")
    args = parser.parse_args()
    config = load_config(args.config)
    steps = Steps(8)

    data = load_match(config, steps)
    n = data.tracking.n_frames
    with steps("Stefan's (src/stoppages.py) and Tim's (src/segments.py) dead balls, their kinds, both judged"):
        bars = dead_balls_of_match(data, config)
    gt = load_public_gt(config, n)
    if gt is not None:
        inside = np.zeros(n, dtype=bool)
        for half in halves(config):
            inside[half["first"] : half["last"] + 1] = True
        bars[TRUTH] = [(int(f), int(s), "D") for f, s in _runs((gt == "D") & inside)]
    masks = {name: _mask(spans, n) for name, spans in bars.items()}

    print("\nDead balls per half and source:")
    print(compare(masks, config).to_string(index=False))
    combined = pd.Series([k for *_, k in bars[COMBINED]]).value_counts()
    print("\nThe judged ones (combined), per kind:")
    print(combined.to_string() if len(combined) else "  none")
    if gt is not None:
        print("\nAgainst the public ground truth's D (understanding only, never tune on it):")
        print(against_truth(masks, gt).to_string())
    figures = {half["name"]: plot_half(bars, half, config) for half in halves(config)}
    if args.save_plot:
        stem, dot, suffix = args.save_plot.rpartition(".")
        for name, fig in figures.items():
            path = f"{stem}_{name}.{suffix}" if dot else f"{args.save_plot}_{name}.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            print(f"wrote {path} - match data, do not commit or share it")
    else:
        say("one plot window per half - close them to finish")
        plt.show()
    return bars


if __name__ == "__main__":
    bars = main()
