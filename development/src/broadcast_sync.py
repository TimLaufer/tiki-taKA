"""Synchronise the broadcast video with the tactical one, from the players' positions.

    python -m src.broadcast_sync                      # match, estimate the offset(s), print, plot
    python -m src.broadcast_sync --save-plot sync.png

Both recordings come with tracking in the same pitch coordinates (0-1, each through its
own camera's homography). Wherever the broadcast shows live play, the players it sees
stand where the tactical tracking has players of the same team at that moment.

  1. Pick `sync.samples` broadcast frames, spread evenly over the usable ones: homography
     valid, reprojection error at most `max_reprojection_error`, at least `min_players`
     players on the pitch, also in the frames `sequence_s` later.
  2. For each, find the tactical frame where the players match best. Track numbers are
     assigned independently in the two trackings (46 vs 51 slots), so track i in one is
     not track i in the other: each broadcast player is compared with the NEAREST tactical
     player of the same team, in metres, capped at `cap_m` against false detections.
     Comparing a short sequence instead of one frame makes the match unambiguous.
     Every `coarse_step`-th tactical frame first, then frame by frame around the best.
  3. offset = tactical frame - broadcast frame (in tactical frames). Drop the `drop_worst`
     share of the worst matches, take the median, count the matches within `tolerance_s`:
     once for one offset over the whole video, once for one offset per half (the broadcast
     is 23 min shorter, so halftime is probably cut and the offset jumps there).
     A match far earlier than the offset is most likely a replay.

Everything is computed, nothing is read off the video by hand (RULES.md), and the samples
are spread evenly instead of drawn at random, so a rerun gives the same result.
"""
from __future__ import annotations

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import Tracking
from src.phases import phase_at
from src.positions import pack_filled, players_on_pitch_m
from src.progress import Progress, Steps, say

BLOCK = 8192  # candidate frames per vectorised step: bounds the memory of the distance arrays
FAR = 1e4  # metres: stands in for an empty tactical slot, far beyond any cap
CANDIDATES = 5  # distinct best moments from the one-frame search, decided between by the whole sequence


def _metres_on_pitch(tracking: Tracking, pitch) -> dict[str, np.ndarray]:
    """Per team ("H", "A"): (frames, slots, 2) positions in metres, NaN for anyone off the pitch."""
    xy = players_on_pitch_m(tracking, pitch)
    return {side: xy[:, tracking.team == side, :] for side in "HA"}


def _visible(slots: np.ndarray) -> np.ndarray:
    """The (n, 2) positions of the slots that have one in this frame."""
    return slots[~np.isnan(slots[:, 0])]


def matching_arrays(tracking: Tracking, pitch) -> dict[str, np.ndarray]:
    """Like _metres_on_pitch, packed for speed: per frame the filled slots come first and
    the columns no frame needs are cut off (about half of the slots are empty at any
    moment), and empty slots stand FAR away instead of NaN, so frame_cost needs no NaN
    check. Which slot a player sits in does not matter to frame_cost anyway."""
    return {side: np.nan_to_num(pack_filled(xy), nan=FAR) for side, xy in _metres_on_pitch(tracking, pitch).items()}


def frame_cost(tactical: dict[str, np.ndarray], points: dict[str, np.ndarray],
               candidates: np.ndarray, cap_m: float) -> np.ndarray:
    """Per candidate tactical frame: the summed squared distance (m^2, capped at cap_m^2)
    from each given player to the nearest tactical player of the same team.

    Independent of track numbers: only the team and the position count.
    `tactical` comes from matching_arrays (no NaN).
    """
    total = np.zeros(len(candidates))
    for side, pts in points.items():
        if len(pts) == 0:
            continue
        for start in range(0, len(candidates), BLOCK):
            block = candidates[start : start + BLOCK]
            slots = tactical[side][block]  # (C, S, 2)
            d2 = ((slots[:, None, :, :] - pts[None, :, None, :]) ** 2).sum(axis=-1)  # (C, P, S)
            total[start : start + len(block)] += np.minimum(d2.min(axis=2), cap_m**2).sum(axis=1)
    return total


def _distinct_minima(frames: np.ndarray, costs: np.ndarray, min_gap: float, k: int) -> list[int]:
    """The k frames with the lowest cost that lie more than min_gap apart from each other."""
    chosen: list[int] = []
    for i in np.argsort(costs):
        if all(abs(int(frames[i]) - c) > min_gap for c in chosen):
            chosen.append(int(frames[i]))
            if len(chosen) == k:
                break
    return chosen


def usable_broadcast_frames(broadcast: Tracking, quality: pd.DataFrame, settings: dict,
                            steps: np.ndarray, pitch) -> np.ndarray:
    """Broadcast frames that may be matched: they and the frames `steps` later all have a
    valid, well-fitting homography and enough players on the pitch."""
    if len(quality) != broadcast.n_frames:
        raise ValueError(f"Homography has {len(quality)} rows, the broadcast tracking {broadcast.n_frames} frames")
    teams = _metres_on_pitch(broadcast, pitch)
    players = sum((~np.isnan(t[:, :, 0])).sum(axis=1) for t in teams.values())
    ok = (quality["valid"].to_numpy()
          & (quality["reprojection_error"].to_numpy() <= settings["max_reprojection_error"])
          & (players >= settings["min_players"]))
    usable = np.zeros_like(ok)
    end = len(ok) - steps.max()
    usable[:end] = np.logical_and.reduce([ok[s : s + end] for s in steps])
    return usable


def match_samples(tactical: Tracking, broadcast: Tracking, quality: pd.DataFrame, config: dict,
                  fps_tactical: float, fps_broadcast: float, progress: bool = False) -> pd.DataFrame:
    """Steps 1 and 2: one row per sampled broadcast frame with its best tactical frame.

    Columns: broadcast_frame, tactical_frame, offset (tactical frames), rms_m (typical
    distance per player at the best match), second_rms_m (best match more than 2 s away:
    close to rms_m means ambiguous), players (per frame of the sequence).
    progress: status lines and a progress bar in the terminal.
    """
    settings, pitch = config["sync"], config["pitch"]
    seconds = np.asarray(settings["sequence_s"], dtype=float)
    b_steps = np.round(seconds * fps_broadcast).astype(int)
    t_steps = np.round(seconds * fps_tactical).astype(int)
    usable = np.flatnonzero(usable_broadcast_frames(broadcast, quality, settings, b_steps, pitch))
    if usable.size == 0:
        raise ValueError("No usable broadcast frames: loosen sync.max_reprojection_error or sync.min_players")
    n = min(settings["samples"], usable.size)
    picks = np.unique(usable[np.linspace(0, usable.size - 1, n).round().astype(int)])
    if progress:
        say(f"usable broadcast frames: {usable.size} of {broadcast.n_frames} "
            f"(homography valid and fitting, >= {settings['min_players']} players); matching {len(picks)} of them")
        say("preparing the positions ...")

    tac_nan = _metres_on_pitch(tactical, pitch)
    in_play = sum((~np.isnan(t[:, :, 0])).sum(axis=1) for t in tac_nan.values()) >= settings["min_players"]
    del tac_nan
    tac = matching_arrays(tactical, pitch)
    bro = _metres_on_pitch(broadcast, pitch)
    last_start = tactical.n_frames - t_steps.max()  # a candidate needs its whole sequence
    step = int(settings["coarse_step"])
    coarse = np.arange(0, last_start, step)
    coarse = coarse[in_play[coarse]]  # no need to search halftime and the empty pitch
    away = 2 * fps_tactical  # "another moment": more than 2 s from the best match

    rows = []
    bar = Progress(len(picks), "matching") if progress else None
    for b in picks:
        sequence = [{side: _visible(bro[side][b + s]) for side in "HA"} for s in b_steps]
        n_players = sum(len(p) for frame in sequence for p in frame.values())

        def cost(candidates):
            return sum(frame_cost(tac, pts, candidates + s, settings["cap_m"])
                       for pts, s in zip(sequence, t_steps)) / n_players

        # coarse: the first frame alone, keep the best few distinct moments ...
        first_cost = frame_cost(tac, sequence[0], coarse, settings["cap_m"])
        results = []
        for rough in _distinct_minima(coarse, first_cost, away, CANDIDATES):
            # ... then the whole sequence, frame by frame around each, decides
            fine = np.arange(max(0, rough - step), min(last_start, rough + step + 1))
            fine_cost = cost(fine)
            best = int(np.argmin(fine_cost))
            results.append((float(fine_cost[best]), int(fine[best])))
        results.sort()
        rows.append({
            "broadcast_frame": int(b),
            "tactical_frame": results[0][1],
            "rms_m": float(np.sqrt(results[0][0])),
            "second_rms_m": float(np.sqrt(results[1][0])) if len(results) > 1 else np.inf,
            "players": n_players / len(b_steps),
        })
        if bar:
            bar.update()
    if bar:
        bar.close()

    matches = pd.DataFrame(rows)
    in_tactical_frames = np.round(matches["broadcast_frame"] * fps_tactical / fps_broadcast).astype(int)
    matches["offset"] = matches["tactical_frame"] - in_tactical_frames
    matches.attrs["usable_frames"] = int(usable.size)
    return matches


def estimate_offsets(matches: pd.DataFrame, config: dict, fps_tactical: float):
    """Step 3. Returns (matches with columns phase, dropped, status; one row per hypothesis).

    status: agrees (within the tolerance of its half's offset), earlier (matched a moment
    clearly before it: replay?), other (later, or outside the halves), dropped (worst matches).
    """
    settings = config["sync"]
    tolerance = round(settings["tolerance_s"] * fps_tactical)
    m = matches.copy()
    m["phase"] = [phase_at(int(t), config) for t in m["tactical_frame"]]
    m["dropped"] = m["rms_m"] > m["rms_m"].quantile(1 - settings["drop_worst"])
    kept = m[~m["dropped"]]

    halves = [p["name"] for p in config["phases"] if "clock_start_min" in p]
    rows = [{"hypothesis": "one offset", "part": "whole video", **_agreement(kept, tolerance)}]
    medians = {}
    for name in halves:
        part = kept[kept["phase"] == name]
        row = _agreement(part, tolerance)
        medians[name] = row["offset"]
        if len(part) >= 2 and row["share"] > 0:
            agree = part[(part["offset"] - row["offset"]).abs() <= tolerance]
            row["slope"] = float(np.polyfit(agree["broadcast_frame"], agree["tactical_frame"], 1)[0]) \
                if agree["broadcast_frame"].nunique() >= 2 else np.nan
        rows.append({"hypothesis": "one offset per half", "part": name, **row})

    status = np.full(len(m), "other", dtype=object)
    for i, (phase, offset, dropped) in enumerate(zip(m["phase"], m["offset"], m["dropped"])):
        if dropped:
            status[i] = "dropped"
        elif phase in medians and medians[phase] is not None:
            if abs(offset - medians[phase]) <= tolerance:
                status[i] = "agrees"
            elif offset < medians[phase]:
                status[i] = "earlier"
    m["status"] = status
    return m, pd.DataFrame(rows)


def _agreement(part: pd.DataFrame, tolerance: int) -> dict:
    """Median offset of these matches and the share of them within the tolerance of it."""
    if len(part) == 0:
        return {"matches": 0, "offset": None, "share": 0.0, "slope": np.nan}
    offset = int(np.median(part["offset"]))
    share = float(((part["offset"] - offset).abs() <= tolerance).mean())
    return {"matches": len(part), "offset": offset, "share": share, "slope": np.nan}


def broadcast_ball_halves(broadcast: Tracking, quality: pd.DataFrame, offsets: dict[str, int],
                          config: dict) -> dict[str, pd.DataFrame]:
    """The broadcast ball moved onto the tactical timeline, one DataFrame per half.

    Same columns as ball_dataframe (index = tactical video frame; phase, ball_x, ball_y,
    ball_s) plus broadcast_frame (-1 outside the broadcast video). offsets: per half,
    tactical frame = broadcast frame + offset (equal frame rates, as broadcast_sync checks).
    The ball only counts where that broadcast frame has a valid homography, i.e. shows the
    pitch; a replay shown in a wide shot still gets through.
    """
    valid = (quality["valid"].to_numpy() &
             (quality["reprojection_error"].to_numpy() <= config.get("sync", {}).get("max_reprojection_error", 3.0)))
    halves = {}
    for name, offset in offsets.items():
        phase = next(p for p in config["phases"] if p["name"] == name)
        frames = np.arange(phase["first"], phase["last"] + 1)
        source = frames - int(offset)
        inside = (source >= 0) & (source < broadcast.n_frames)
        usable = np.zeros(len(frames), dtype=bool)
        usable[inside] = valid[source[inside]]
        ball = np.full((len(frames), 3), np.nan)
        ball[usable, :2] = broadcast.ball[source[usable]]
        ball[usable, 2] = broadcast.ball_speed[source[usable]]
        half = pd.DataFrame({"phase": name, "ball_x": ball[:, 0], "ball_y": ball[:, 1], "ball_s": ball[:, 2],
                             "broadcast_frame": np.where(inside, source, -1)},
                            index=pd.Index(frames, name="frame"))
        half.attrs["speed_unit"] = broadcast.speed_unit
        halves[name] = half
    return halves


def broadcast_shows_live(tactical: Tracking, broadcast: Tracking, frames: np.ndarray, offset: int,
                         config: dict, block: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Does the broadcast show these tactical frames live? Per frame, the broadcast frame
    `frame - offset` is compared with the tactical frame like in the matching: each broadcast
    player against the nearest tactical player of the same team, capped at sync.cap_m.

    Returns (rms_m, players): the typical distance (about 1 m in live play, large in a replay
    or with a wrong homography; NaN without players) and how many players were compared.
    """
    pitch, cap = config["pitch"], config["sync"]["cap_m"]
    frames = np.asarray(frames, dtype=int)
    tac = matching_arrays(tactical, pitch)
    bro = _metres_on_pitch(broadcast, pitch)
    source = frames - int(offset)
    inside = (source >= 0) & (source < broadcast.n_frames)
    source = np.clip(source, 0, broadcast.n_frames - 1)
    total, count = np.zeros(len(frames)), np.zeros(len(frames))
    for side in "HA":
        for start in range(0, len(frames), block):
            part = slice(start, start + block)
            players = bro[side][source[part]]  # (F, Sb, 2), NaN for empty slots
            slots = tac[side][frames[part]]  # (F, S, 2), FAR for empty slots
            d2 = ((players[:, :, None, :] - slots[:, None, :, :]) ** 2).sum(axis=-1)  # (F, Sb, S)
            nearest = np.minimum(d2.min(axis=2), cap**2)  # NaN where the broadcast slot is empty
            compared = ~np.isnan(nearest) & inside[part, None]
            total[part] += np.where(compared, nearest, 0).sum(axis=1)
            count[part] += compared.sum(axis=1)
    rms = np.where(count > 0, np.sqrt(total / np.maximum(count, 1)), np.nan)
    return rms, count.astype(int)


def _signed_time(frames: float, fps: float) -> str:
    seconds = abs(frames) / fps
    return f"{'-' if frames < 0 else '+'}{int(seconds // 60)}:{seconds % 60:05.2f}"


def report(matches: pd.DataFrame, table: pd.DataFrame, config: dict, fps_tactical: float,
           fps_broadcast: float, n_broadcast: int) -> str:
    settings = config["sync"]
    dropped = matches.loc[matches["dropped"], "rms_m"]
    lines = [
        f"frame rates: tactical {fps_tactical:g} fps, broadcast {fps_broadcast:g} fps"
        + ("" if fps_tactical == fps_broadcast else "  -> different: offsets are in tactical frames"),
        f"usable broadcast frames: {matches.attrs.get('usable_frames', '?')} of {n_broadcast}",
        f"matched {len(matches)} samples; typical distance per player at the best match: "
        f"median {matches['rms_m'].median():.2f} m",
        f"dropped the worst {settings['drop_worst']:.0%}: {len(dropped)} matches"
        + (f" above {dropped.min():.2f} m" if len(dropped) else ""),
        "",
    ]
    shown = table.copy()
    shown["offset (time)"] = [None if o is None else _signed_time(o, fps_tactical) for o in shown["offset"]]
    shown["within"] = [f"{s:.0%}" for s in shown["share"]]
    shown["slope"] = [f"{s:.4f}" if pd.notna(s) else "" for s in shown["slope"]]
    shown = shown.rename(columns={"offset": "offset (frames)",
                                  "within": f"within ±{settings['tolerance_s']} s"})
    lines.append(shown[["hypothesis", "part", "matches", "offset (frames)", "offset (time)",
                        f"within ±{settings['tolerance_s']} s", "slope"]].to_string(index=False))
    counts = matches["status"].value_counts()
    lines += [
        "",
        f"per half: {counts.get('agrees', 0)} agree, {counts.get('earlier', 0)} matched an earlier moment "
        f"(replay?), {counts.get('other', 0)} other, {counts.get('dropped', 0)} dropped",
        f"slope 1.0000 = no stretching (expected {fps_tactical / fps_broadcast:.4f} from the frame rates)",
    ]
    return "\n".join(lines)


def plot_offsets(matches: pd.DataFrame, table: pd.DataFrame, config: dict, ax=None):
    """Offset over the broadcast video: one horizontal line per constant offset."""
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 5), layout="constrained")
    styles = {
        "agrees": dict(color="#1d4ed8", marker="o", s=14, label="agrees with its half"),
        "earlier": dict(color="#f59e0b", marker="o", s=18, label="earlier moment (replay?)"),
        "other": dict(color="#9ca3af", marker="o", s=14, label="other"),
        "dropped": dict(color="#dc2626", marker="x", s=22, label=f"worst {config['sync']['drop_worst']:.0%}, dropped"),
    }
    for status, style in styles.items():
        part = matches[matches["status"] == status]
        if len(part):
            ax.scatter(part["broadcast_frame"], part["offset"], zorder=3, **style)
    for row in table[table["hypothesis"] == "one offset per half"].itertuples():
        if row.offset is None:
            continue
        frames = matches.loc[matches["phase"] == row.part, "broadcast_frame"]
        ax.hlines(row.offset, frames.min(), frames.max(), color="#111827", lw=1, zorder=2)
        ax.annotate(f"{row.part}: {row.offset:+d} frames ({row.share:.0%} agree)", (frames.min(), row.offset),
                    xytext=(0, 6), textcoords="offset points", fontsize=9)
    ax.set_xlabel("broadcast frame")
    ax.set_ylabel("offset = tactical frame - broadcast frame")
    ax.set_title("Broadcast -> tactical: offset of every matched sample", loc="left")
    ax.legend(loc="best", fontsize=8)
    return ax


def main():
    from src.data_loader import (DEFAULT_CONFIG, atd_frame_rate, load_config,
                                 load_homography_quality, load_tracking)

    parser = argparse.ArgumentParser(description="Synchronise the broadcast with the tactical video.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--save-plot", help="write the plot to this file instead of a window")
    args = parser.parse_args()
    config = load_config(args.config)
    steps = Steps(5)

    with steps("Loading the tactical tracking"):
        tactical = load_tracking(config, "tactical")
    with steps("Loading the broadcast tracking"):
        broadcast = load_tracking(config, "broadcast")
        if list(tactical.team_names) != list(broadcast.team_names):
            warnings.warn(f"Teams differ: tactical {list(tactical.team_names)}, broadcast {list(broadcast.team_names)}")
        if tactical.home_attacks_first_half != broadcast.home_attacks_first_half:
            warnings.warn("The two trackings disagree on the playing direction: coordinates may be mirrored")
    with steps("Loading the broadcast homography quality and the frame rates"):
        quality = load_homography_quality(config, "broadcast")
        fps_t, fps_b = atd_frame_rate(config, "tactical"), atd_frame_rate(config, "broadcast")
        say(f"{int(quality['valid'].sum())} of {len(quality)} broadcast frames have a homography; "
            f"{fps_t:g} fps tactical, {fps_b:g} fps broadcast")
    with steps(f"Matching broadcast frames against the {tactical.n_frames} tactical frames (the long part)"):
        matches = match_samples(tactical, broadcast, quality, config, fps_t, fps_b, progress=True)
    with steps("Estimating the offsets"):
        matches, table = estimate_offsets(matches, config, fps_t)

    print()
    print(report(matches, table, config, fps_t, fps_b, broadcast.n_frames))
    ax = plot_offsets(matches, table, config)
    if args.save_plot:
        ax.figure.savefig(args.save_plot, dpi=120)
        print(f"\nwrote {args.save_plot}")
    else:
        say("the plot window is open - close it to finish")
        plt.show()
    return matches, table


if __name__ == "__main__":
    matches, table = main()
