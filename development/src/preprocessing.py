"""Combine the cleaning, dead-ball, restart and stoppage analyses for prediction.

No ground truth is loaded here or anywhere downstream of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import Tracking
from src.dead_ball import episodes, indicators, long_gaps
from src.phases import half_of_frames
from src.pipeline import MatchData, measured
from src.restarts import find_restarts
from src.smart_tagging import COLUMNS as SET_PIECE_COLUMNS, load_set_pieces
from src.stoppages import KEEPER, judge_gaps, restart_seeds, stoppage_mask


@dataclass
class PreparedData:
    tracking: Tracking  # players for control evidence: held goalkeeper positions removed
    ball: pd.DataFrame  # the analysed ball, broadcast-completed where the tactical one is missing
    dead: np.ndarray  # per frame: the ball is out of play
    gaps: pd.DataFrame  # long tactical ball gaps with their verdict (src.stoppages.judge_gaps)
    restarts: pd.DataFrame  # accepted restarts of a placed ball (src.restarts)
    seeds: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["start", "stop", "team", "how"]))
    holds: np.ndarray | None = None  # per frame: the team whose goalkeeper holds the unseen ball
    set_pieces: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=SET_PIECE_COLUMNS))
    smart_tagging: Path | None = None  # the Smart Tagging file used, if any


def accepted_restarts(tracking: Tracking, ball: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Placed-ball restarts (src.restarts) whose taker stands at the ball, whose team does not
    contradict the goal geometry, and whose ball is then seen moving in the same half."""
    candidates = find_restarts(tracking, ball, config)
    radius = config["model"]["control_radius_m"]
    lookahead = max(1, round(config["restarts"]["smooth_s"] * config["fps"]))
    half = half_of_frames(tracking.n_frames, config)
    speed = ball["speed_ms"].to_numpy()
    accepted = []
    for row in candidates.itertuples():
        after = np.arange(row.last + 1, min(row.last + 1 + lookahead, tracking.n_frames))
        moves = np.isfinite(speed[after]) & (speed[after] > config["restarts"]["still_speed_ms"])
        same_half = half[after] == half[row.last]
        accepted.append(np.isfinite(row.taker_m) and row.taker_m <= radius
                        and row.agrees != "no" and bool((moves & same_half).any()))
    return candidates.loc[np.asarray(accepted, dtype=bool)].copy()


def preprocess(raw: MatchData, config: dict, set_pieces: pd.DataFrame | None = None) -> PreparedData:
    """Dead ball, keeper holds and restart teams from algorithmic evidence only.

    set_pieces: Smart Tagging set pieces (src.smart_tagging); loaded from the data directory
    when None, as configured.
    """
    tracking = measured(raw)
    n = tracking.n_frames
    if tracking.first_frame != 0 or not np.array_equal(raw.ball.index.to_numpy(), np.arange(n)):
        raise ValueError("Preprocessing requires the full tactical timeline, starting at frame 0")
    tactical = raw.tactical_ball if raw.tactical_ball is not None else raw.ball
    tagging_path = None
    if set_pieces is None:
        set_pieces, tagging_path = load_set_pieces(config, tracking.team_names)
    ind = indicators(tracking, config)
    gaps = long_gaps(tactical, ind, episodes(ind, tracking, config), config)
    judged = judge_gaps(gaps, tracking, tactical, set_pieces, config)
    restarts = accepted_restarts(tracking, raw.ball, config)
    speed = ind["speed_ms"].to_numpy(dtype=float)
    dead = stoppage_mask(n, judged, raw.ball, restarts, speed, config)
    holds = np.full(n, "", dtype="<U1")
    for gap in judged[judged["verdict"] == KEEPER].itertuples():
        holds[int(gap.first) : int(gap.last) + 1] = gap.team
    holds[dead] = ""
    seeds = restart_seeds(dead, raw.ball, tracking, restarts, config)
    return PreparedData(tracking, raw.ball, dead, judged, restarts, seeds=seeds, holds=holds,
                        set_pieces=set_pieces, smart_tagging=tagging_path)
