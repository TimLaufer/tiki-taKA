"""Cleaning the player tracks: lost players bridged, players still missing taken from video 2.

Track slots are not people: when the tracker loses a player, his track often ends and a new
track, in another slot, starts when he is found again. So a gap is bridged across tracks too.
Both steps work inside the halves; settings under `players` in config.yaml.

1. Bridging (`bridge_tracks`): a track ends, and within max_gap_s a track of the same team
   starts where that player can have got to (max_speed_ms x the time + slack_m). The player is
   then moved on the straight line in between, written into a slot free over the gap. The same
   track picking up again is preferred (switch_penalty_m); each end and start is used once, the
   cheapest links (shortest way) first. Identities stay as they are: a bridge only closes the gap.
2. Video 2 (`fill_from_video_2`): where a team has fewer than per_team players on the pitch and
   the broadcast shows that moment live (its players match the tactical ones, as for the ball:
   merge.max_player_rms_m, merge.min_players), the broadcast players of that team farther than
   match_radius_m from every tactical one fill in, up to per_team, in extra slots v2H1, v2A1, ...
   (extra_slots per team). Their order in those slots is no identity.

Where each position comes from, per frame and slot (`source` arrays, SOURCE_NAMES):
0 none, 1 observed, 2 bridged on the same track, 3 bridged to another track, 4 from video 2;
for the keepers (src/goalkeeper.py) 5 the deepest player, 6 held where last seen.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.data_loader import Tracking
from src.positions import players_on_pitch_m

NONE, OBSERVED, SAME_TRACK, OTHER_TRACK, VIDEO_2, KEEPER_DEEPEST, KEEPER_HELD = 0, 1, 2, 3, 4, 5, 6
SOURCE_NAMES = {NONE: "none", OBSERVED: "observed", SAME_TRACK: "bridged (same track)",
                OTHER_TRACK: "bridged (to another track)", VIDEO_2: "from video 2",
                KEEPER_DEEPEST: "keeper: deepest player", KEEPER_HELD: "keeper: held at last position"}
FAR = 1e4  # metres: stands in for an empty slot
BLOCK = 8192


def _half_of_frames(n_frames: int, first_frame: int, config: dict) -> np.ndarray:
    """Per frame: 1 + the index of the half it lies in, 0 outside the halves."""
    half = np.zeros(n_frames, dtype=np.int8)
    frames = np.arange(first_frame, first_frame + n_frames)
    for number, phase in enumerate(p for p in config["phases"] if "clock_start_min" in p):
        half[(frames >= phase["first"]) & (frames <= phase["last"])] = number + 1
    return half


def bridge_tracks(tracking: Tracking, config: dict, alone: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    """Step 1. Returns the players (frames, slots, 2, 0-1) with the bridges in, and the source.

    alone: slots only ever bridged to themselves, e.g. the goalkeeper tracks: no other track
    may take them over, and they take over no other track.
    """
    settings, fps = config["players"], config["fps"]
    pitch = np.asarray(config["pitch"], dtype=float)
    players = tracking.players.astype(np.float32).copy()
    present = ~np.isnan(players[:, :, 0])
    source = np.where(present, OBSERVED, NONE).astype(np.uint8)
    n, slots = present.shape
    half = _half_of_frames(n, tracking.first_frame, config)
    max_gap = int(round(settings["max_gap_s"] * fps))

    change = np.diff(np.vstack([np.zeros((1, slots), np.int8), present.astype(np.int8),
                                np.zeros((1, slots), np.int8)]), axis=0)
    start_frame, start_slot = np.nonzero(change == 1)  # a track starts at this frame
    stop_frame, end_slot = np.nonzero(change == -1)
    end_frame = stop_frame - 1  # ... and ends at this one
    keep_start = (start_frame > 0) & (half[start_frame] > 0)
    keep_end = (stop_frame < n) & (half[end_frame] > 0)
    start_frame, start_slot = start_frame[keep_start], start_slot[keep_start]
    end_frame, end_slot = end_frame[keep_end], end_slot[keep_end]
    order = np.argsort(start_frame, kind="stable")
    start_frame, start_slot = start_frame[order], start_slot[order]
    start_pos = players[start_frame, start_slot] * pitch
    end_pos = players[end_frame, end_slot] * pitch

    candidates = []  # (cost, end index, start index)
    for e, (frame, slot) in enumerate(zip(end_frame, end_slot)):
        lo, hi = np.searchsorted(start_frame, [frame + 2, frame + max_gap + 2])  # 1 .. max_gap frames missing
        if lo == hi:
            continue
        idx = np.arange(lo, hi)
        idx = idx[(tracking.team[start_slot[idx]] == tracking.team[slot]) & (half[start_frame[idx]] == half[frame])]
        if idx.size == 0:
            continue
        if alone:  # a slot kept alone only links to itself
            isolated = np.isin(start_slot[idx], alone) | (slot in alone)
            idx = idx[~isolated | (start_slot[idx] == slot)]
            if idx.size == 0:
                continue
        distance = np.hypot(*(start_pos[idx] - end_pos[e]).T)
        reachable = distance <= settings["max_speed_ms"] * (start_frame[idx] - frame) / fps + settings["slack_m"]
        cost = distance + np.where(start_slot[idx] == slot, 0.0, settings["switch_penalty_m"])
        candidates += [(c, e, s) for c, s in zip(cost[reachable], idx[reachable])]

    used_end, used_start = set(), set()
    for _, e, s in sorted(candidates):
        if e in used_end or s in used_start:
            continue
        first, stop = end_frame[e] + 1, start_frame[s]  # the missing frames: first .. stop - 1
        target = next((slot for slot in (end_slot[e], start_slot[s])
                       if not (source[first:stop, slot] != NONE).any()), None)
        if target is None:
            continue  # neither slot is free over the gap
        used_end.add(e)
        used_start.add(s)
        weight = (np.arange(first, stop) - end_frame[e])[:, None] / (stop - end_frame[e])
        a, b = players[end_frame[e], end_slot[e]], players[stop, start_slot[s]]
        players[first:stop, target] = a + weight * (b - a)
        source[first:stop, target] = SAME_TRACK if end_slot[e] == start_slot[s] else OTHER_TRACK
    return players, source


def fill_from_video_2(tracking: Tracking, source: np.ndarray, video_2, config: dict) -> tuple[Tracking, np.ndarray]:
    """Step 2: tracking (already bridged) with extra_slots per team appended and filled from the
    broadcast where a team has fewer than per_team players and the broadcast shows the moment live.
    video_2: src.pipeline.Video2 (its tracking and offsets per half)."""
    from src.broadcast_sync import broadcast_shows_live

    settings, merge = config["players"], config["merge"]
    pitch = np.asarray(config["pitch"], dtype=np.float32)
    k, per_team = int(settings["extra_slots"]), int(settings["per_team"])
    n = tracking.n_frames
    extra = np.full((n, 2 * k, 2), np.nan, dtype=np.float32)
    tac = players_on_pitch_m(tracking, config["pitch"])
    bro = players_on_pitch_m(video_2.tracking, config["pitch"])
    count = {side: (~np.isnan(tac[:, tracking.team == side, 0])).sum(axis=1) for side in "HA"}

    for name, offset in video_2.offsets.items():
        phase = next(p for p in config["phases"] if p["name"] == name)
        frames = np.arange(phase["first"], phase["last"] + 1) - tracking.first_frame
        frames = frames[(frames >= 0) & (frames < n)]
        frames = frames[(count["H"][frames] < per_team) | (count["A"][frames] < per_team)]
        inside = (frames + tracking.first_frame - offset >= 0) & (frames + tracking.first_frame - offset
                                                                  < video_2.tracking.n_frames)
        frames = frames[inside]
        if frames.size == 0:
            continue
        rms, compared = broadcast_shows_live(tracking, video_2.tracking, frames + tracking.first_frame, offset, config)
        frames = frames[(compared >= merge["min_players"]) & (rms <= merge["max_player_rms_m"])]
        for s, side in enumerate("HA"):
            team_tac = np.nan_to_num(tac[:, tracking.team == side], nan=FAR)
            team_bro = bro[:, video_2.tracking.team == side]
            for start in range(0, len(frames), BLOCK):
                f = frames[start : start + BLOCK]
                b = f + tracking.first_frame - offset
                points, slots = team_bro[b], team_tac[f]  # (F, Sb, 2), (F, S, 2)
                nearest = np.sqrt(((points[:, :, None, :] - slots[:, None, :, :]) ** 2).sum(axis=-1).min(axis=2))
                isolated = ~np.isnan(points[:, :, 0]) & (nearest > settings["match_radius_m"])
                score = np.where(isolated, nearest, -np.inf)  # the most isolated first
                order = np.argsort(-score, axis=1)[:, :k]
                need = np.clip(per_team - count[side][f], 0, k)
                take = (np.arange(min(k, order.shape[1]))[None, :] < np.minimum(need, isolated.sum(axis=1))[:, None])
                chosen = np.take_along_axis(points, order[:, :, None], axis=1) / pitch
                block = np.full((len(f), k, 2), np.nan, dtype=np.float32)
                block[:, : chosen.shape[1]] = np.where(take[:, :, None], chosen, np.nan)
                extra[f, s * k : (s + 1) * k] = block

    extra_source = np.where(np.isnan(extra[:, :, 0]), NONE, VIDEO_2).astype(np.uint8)
    names = [f"v2{side}{i + 1}" for side in "HA" for i in range(k)]
    cleaned = replace(
        tracking,
        players=np.concatenate([tracking.players, extra], axis=1),
        speed=np.concatenate([tracking.speed, np.full((n, 2 * k), np.nan, dtype=np.float32)], axis=1),
        team=np.concatenate([tracking.team, np.array(["H"] * k + ["A"] * k)]),
        slot_ids=np.concatenate([tracking.slot_ids, np.array(names)]),
    )
    return cleaned, np.concatenate([source, extra_source], axis=1)


def clean_players(tracking: Tracking, video_2, config: dict,
                  alone: tuple[int, ...] = ()) -> tuple[Tracking, np.ndarray]:
    """Both steps: the cleaned tracking and, per frame and slot, where each position comes from.
    alone: slots bridged only to themselves (the goalkeeper tracks)."""
    players, source = bridge_tracks(tracking, config, alone)
    bridged = replace(tracking, players=players)
    if video_2 is None or not config["players"].get("from_video_2", True):
        return bridged, source
    return fill_from_video_2(bridged, source, video_2, config)


def player_table(raw: Tracking, cleaned: Tracking, source: np.ndarray, config: dict) -> pd.DataFrame:
    """Per half: players on the pitch per frame before and after, and where the positions come from."""
    rows = []
    raw_m, clean_m = players_on_pitch_m(raw, config["pitch"]), players_on_pitch_m(cleaned, config["pitch"])
    frames = np.arange(raw.first_frame, raw.first_frame + raw.n_frames)
    for phase in (p for p in config["phases"] if "clock_start_min" in p):
        inside = (frames >= phase["first"]) & (frames <= phase["last"])
        row = {"half": phase["name"]}
        for side in "HA":
            before = (~np.isnan(raw_m[inside][:, raw.team == side, 0])).sum(axis=1).mean()
            after = (~np.isnan(clean_m[inside][:, cleaned.team == side, 0])).sum(axis=1).mean()
            row[f"{side} players before"], row[f"{side} after"] = round(float(before), 2), round(float(after), 2)
        codes = source[inside]
        positions = max(int((codes != NONE).sum()), 1)
        for code in (OBSERVED, SAME_TRACK, OTHER_TRACK, VIDEO_2, KEEPER_DEEPEST, KEEPER_HELD):
            if code < KEEPER_DEEPEST or (codes == code).any():
                row[SOURCE_NAMES[code]] = f"{(codes == code).sum() / positions:.1%}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("half")
