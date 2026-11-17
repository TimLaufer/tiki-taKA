"""The goalkeeper model: one keeper per team in every frame of the halves, always identifiable.

The tactical camera pans with play, so the keeper at the far end is often out of the picture.
The model, per half and team (settings under `goalkeeper` in config.yaml):

1. Which goal the team defends: from the kickoff, when every player must stand in his own half -
   the team's centre in the second before the kickoff. Checked against the metadata's direction.
2. Seen: the keeper track the metadata marks (position_type Goalkeeper; the tracker probably
   tells keepers by their kit) wherever it stands in its own half. Otherwise the deepest player
   of the team - nearest his own goal line - if he stands in his own penalty box at least
   margin_m deeper than the next teammate; without that margin the deepest player in the picture
   may just be the last defender while the keeper is out of it. That player moves to the keeper slot.
3. Unseen: he stays where he was last seen in this half - the keeper is the deepest man and
   hardly moves while the play is far from him - and jumps to his new position when he is seen
   again. Before his first sighting in a half: where he is first seen. Never seen in a half:
   fallback_depth_m in front of his goal centre. Short gaps of the keeper track are bridged
   before (src/player_cleaning.py, the keeper track only ever bridged to itself).

The keepers sit in slots GKH and GKA (the marked keeper tracks, renamed, or new slots); where
their position comes from is in the source array: observed, bridged, KEEPER_DEEPEST, KEEPER_HELD.
"""
from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from dataclasses import replace

import numpy as np
import pandas as pd

from src.data_loader import Tracking
from src.player_cleaning import KEEPER_DEEPEST, KEEPER_HELD, NONE, OBSERVED
from src.progress import say

PENALTY_BOX_DEPTH, PENALTY_BOX_HALF_WIDTH = 16.5, 20.16
KICKOFF_WINDOW_S = 1.0  # the team's centre over this long before the kickoff


def keeper_slots(meta_path, tracking: Tracking) -> dict[str, int]:
    """Per team, the slot the metadata marks as goalkeeper (position_type Goalkeeper), if any."""
    root = ET.parse(meta_path).getroot()
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    slots = {}
    for player in root.iter("Player"):
        types = [p.findtext("Value") for p in player.iter("ProviderParameter") if p.findtext("Name") == "position_type"]
        if types and (types[0] or "").strip().lower() == "goalkeeper":
            index = np.flatnonzero(tracking.slot_ids == player.get("id"))
            if index.size:
                slots.setdefault(str(tracking.team[index[0]]), int(index[0]))
    return slots


def defended_goals(tracking: Tracking, config: dict) -> dict[tuple[str, str], str]:
    """(half, team) -> "left" or "right": the goal the team defends, from its centre at the kickoff."""
    length = config["pitch"][0]
    xy = tracking.players[:, :, 0] * length
    window = int(round(KICKOFF_WINDOW_S * config["fps"]))
    goals = {}
    for half in (p for p in config["phases"] if "clock_start_min" in p):
        kickoff = half["first"] - tracking.first_frame
        frames = slice(max(kickoff - window, 0), kickoff + 1)
        for side in "HA":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                centre = np.nanmean(xy[frames][:, tracking.team == side])
            goals[(half["name"], side)] = "left" if np.isnan(centre) or centre < length / 2 else "right"
    _check_against_metadata(goals, tracking, config)
    return goals


def _check_against_metadata(goals: dict, tracking: Tracking, config: dict) -> None:
    direction = tracking.home_attacks_first_half
    if direction not in ("left_to_right", "right_to_left"):
        return
    for (half_name, side), goal in goals.items():
        half = next(p for p in config["phases"] if p["name"] == half_name)
        home_defends_left = (direction == "left_to_right") == (half.get("half") == 1)
        expected = "left" if home_defends_left == (side == "H") else "right"
        if goal != expected:
            say(f"note: at the kickoff of {half_name}, {side} stands in the {goal} half, "
                f"the metadata says it defends the {expected} goal - the kickoff is taken")


def place_keepers(tracking: Tracking, source: np.ndarray, config: dict,
                  slots: dict[str, int]) -> tuple[Tracking, np.ndarray]:
    """Steps 2 and 3: a keeper for each team in every frame of the halves, in slot GKH / GKA."""
    settings = config["goalkeeper"]
    length, width = config["pitch"]
    players, source = tracking.players.copy(), source.copy()
    team, slot_ids = tracking.team.copy(), tracking.slot_ids.astype(object).copy()
    speed = tracking.speed
    n = tracking.n_frames
    for side in "HA":  # a slot for each keeper: the marked track, or a new one
        if side not in slots:
            players = np.concatenate([players, np.full((n, 1, 2), np.nan, np.float32)], axis=1)
            source = np.concatenate([source, np.zeros((n, 1), np.uint8)], axis=1)
            speed = np.concatenate([speed, np.full((n, 1), np.nan, np.float32)], axis=1)
            team, slot_ids = np.append(team, side), np.append(slot_ids, "")
            slots = {**slots, side: players.shape[1] - 1}
        slot_ids[slots[side]] = f"GK{side}"

    goals = defended_goals(tracking, config)
    metres = players.astype(np.float64) * [length, width]
    for half in (p for p in config["phases"] if "clock_start_min" in p):
        frames = np.arange(max(half["first"] - tracking.first_frame, 0), min(half["last"] - tracking.first_frame + 1, n))
        for side in "HA":
            keeper = slots[side]
            goal_x = 0.0 if goals[(half["name"], side)] == "left" else length
            depth = np.abs(metres[frames, :, 0] - goal_x)  # distance from his own goal line
            own_half = depth <= length / 2

            # 2a. the keeper track, where it stands in its own half
            seen = ~np.isnan(depth[:, keeper]) & own_half[:, keeper]
            players[frames[~seen], keeper] = np.nan
            source[frames[~seen], keeper] = NONE

            # 2b. else the deepest player, in his own box and clearly deepest
            others = (team == side) & (np.arange(players.shape[1]) != keeper)
            d = np.where(others & ~np.isnan(depth) & own_half, depth, np.inf)
            order = np.argsort(d, axis=1)
            first, second = np.take_along_axis(d, order[:, :1], 1)[:, 0], np.take_along_axis(d, order[:, 1:2], 1)[:, 0]
            deepest = order[:, 0]
            y = metres[frames, deepest, 1]
            in_box = (first <= PENALTY_BOX_DEPTH) & (np.abs(y - width / 2) <= PENALTY_BOX_HALF_WIDTH)
            with np.errstate(invalid="ignore"):  # inf - inf where nobody is there: no keeper anyway
                clearly_deepest = second - first >= settings["margin_m"]
            take = ~seen & in_box & clearly_deepest
            rows, columns = frames[take], deepest[take]
            players[rows, keeper] = players[rows, columns]
            source[rows, keeper] = KEEPER_DEEPEST
            players[rows, columns] = np.nan  # he is the keeper now, not a field player as well
            source[rows, columns] = NONE

            # 3. unseen: where he was last seen in this half (before the first sighting: the first)
            known = seen | take
            if known.all():
                continue
            if known.any():
                last = np.maximum.accumulate(np.where(known, np.arange(len(frames)), -1))
                last[last < 0] = np.flatnonzero(known)[0]
                held = players[frames[last], keeper]
            else:  # never seen in this half
                fallback = [abs(goal_x - settings["fallback_depth_m"]) / length, 0.5]
                held = np.tile(np.array(fallback, np.float32), (len(frames), 1))
            players[frames[~known], keeper] = held[~known]
            source[frames[~known], keeper] = KEEPER_HELD
    keepers = replace(tracking, players=players, speed=speed, team=team, slot_ids=slot_ids.astype(str))
    return keepers, source


def keeper_table(tracking: Tracking, source: np.ndarray, config: dict) -> pd.DataFrame:
    """Per half and keeper: where his position comes from."""
    rows = []
    frames = np.arange(tracking.first_frame, tracking.first_frame + tracking.n_frames)
    names = {OBSERVED: "seen (keeper track)", 2: "bridged", KEEPER_DEEPEST: "seen (deepest player)",
             KEEPER_HELD: "held (last seen)"}
    for half in (p for p in config["phases"] if "clock_start_min" in p):
        inside = (frames >= half["first"]) & (frames <= half["last"])
        for side in "HA":
            slot = np.flatnonzero(tracking.slot_ids == f"GK{side}")
            if slot.size == 0:
                continue
            codes = source[inside, slot[0]]
            rows.append({"half": half["name"], "keeper": f"GK{side}",
                         **{name: f"{(codes == code).mean():.1%}" for code, name in names.items()}})
    return pd.DataFrame(rows)
