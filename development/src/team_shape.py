"""Team shape: the team in possession spreads out, advances and runs less than the team chasing it.

Optional stage (config `team_shape.enabled`). Per frame, three differences between the outfield
players of the two teams (home minus away):

    spread   mean distance of the players to their team's centre (m)
    advance  speed of the team's centre towards the opponent goal (m/s), over +-0.5 s
    pace     median speed of the players (m/s), over +-0.5 s

A logistic model P(home has the ball | differences) is fitted per match on the pipeline's own
unambiguous frames - the owning team's player has the ball at his feet - so its labels are
pseudo-labels produced by the algorithm, not by a person (RULES.md). On the challenge match it
separates those frames with AUC 0.88. Team shape follows a change of possession only about 3 s
later (measured on the pipeline's own turnovers), so it never times a change; it judges one:

    veto  a possession won in open play that lasts at least lag_s + settle_s, but whose settled
          part (from lag_s after its start) the shape gives to the previous owner - its winner
          gets a probability below veto_below - is returned to the previous owner;
    fill  a run of frames no team could be given (X) takes the team the shape favours, when the
          shape is as clear as for a veto.
"""
from __future__ import annotations

import warnings

import numpy as np

from src.data_loader import Tracking
from src.dead_ball import _runs
from src.goalkeeper import defended_goals
from src.positions import players_on_pitch_m

TEAMS = ("H", "A")


def shape_differences(tracking: Tracking, config: dict) -> np.ndarray:
    """(frames, 3): spread, advance and pace of the home minus the away outfield players."""
    fps, n = config["fps"], tracking.n_frames
    k = max(1, round(0.5 * fps))
    xy = players_on_pitch_m(tracking, config["pitch"]).astype(float)
    outfield = ~np.char.startswith(tracking.slot_ids.astype(str), "GK")
    goals = defended_goals(tracking, config)
    frames = np.arange(tracking.first_frame, tracking.first_frame + n)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames without any player: NaN
        for side in TEAMS:
            players = xy[:, (tracking.team == side) & outfield]
            centre = np.nanmean(players, axis=1)
            towards = np.zeros(n)  # +1 where the team attacks the right goal, 0 outside the halves
            for phase in (p for p in config["phases"] if "clock_start_min" in p):
                inside = (frames >= phase["first"]) & (frames <= phase["last"])
                towards[inside] = 1.0 if goals[(phase["name"], side)] == "left" else -1.0
            advance = np.full(n, np.nan)
            pace = np.full(players.shape[:2], np.nan)
            if n > 2 * k:
                advance[k:-k] = (centre[2 * k :, 0] - centre[: -2 * k, 0]) / (2 * k / fps)
                pace[k:-k] = np.hypot(*(players[2 * k :] - players[: -2 * k]).transpose(2, 0, 1)) / (2 * k / fps)
            spread = np.nanmean(np.hypot(*(players - centre[:, None, :]).transpose(2, 0, 1)), axis=1)
            out[side] = np.c_[spread, advance * towards, np.nanmedian(pace, axis=1)]
    return out["H"] - out["A"]


def fit(differences: np.ndarray, labels: np.ndarray, evidence: np.ndarray, min_frames: int = 1000):
    """Logistic weights (intercept first) on standardized differences, from the frames whose
    label has control evidence of its own team (the pseudo-labels); None if too few of either
    team. Iteratively reweighted least squares: deterministic."""
    usable = np.isfinite(differences).all(axis=1) & np.isin(labels, TEAMS) & (evidence == labels)
    home = labels[usable] == "H"
    if min(home.sum(), (~home).sum()) < min_frames:
        return None
    mean, std = differences[usable].mean(axis=0), differences[usable].std(axis=0)
    std[std == 0] = 1.0
    z = np.c_[np.ones(usable.sum()), (differences[usable] - mean) / std]
    y = home.astype(float)
    weights = np.zeros(z.shape[1])
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-z @ weights))
        hessian = z.T @ (z * (p * (1 - p))[:, None]) + 1e-6 * np.eye(z.shape[1])
        step = np.linalg.solve(hessian, z.T @ (y - p))
        weights += step
        if np.abs(step).max() < 1e-9:
            break
    return {"weights": weights, "mean": mean, "std": std}


def home_logit(differences: np.ndarray, model: dict) -> np.ndarray:
    """Per frame: log-odds that home has the ball by team shape alone (0 where unknown)."""
    ok = np.isfinite(differences).all(axis=1)
    z = np.c_[np.ones(len(differences)), (np.nan_to_num(differences) - model["mean"]) / model["std"]]
    return np.where(ok, z @ model["weights"], 0.0)


def _probability(logit: np.ndarray, team: str) -> float:
    mean = float(np.mean(logit)) * (1.0 if team == "H" else -1.0)
    return 1.0 / (1.0 + np.exp(-mean))


def apply(labels: np.ndarray, logit: np.ndarray, graded: np.ndarray, config: dict) -> tuple[np.ndarray, dict]:
    """The veto and the fill (module docstring) on finished labels; returns labels and counts.
    Only graded frames are filled."""
    settings = config["team_shape"]
    fps = config["fps"]
    lag, settle = round(settings["lag_s"] * fps), round(settings["settle_s"] * fps)
    threshold = settings["veto_below"]
    out = labels.copy()
    vetoed = filled = 0
    position = 1
    while position < len(out):
        team = out[position]
        if team not in TEAMS or out[position - 1] not in TEAMS or out[position - 1] == team:
            position += 1
            continue
        stop = position
        while stop < len(out) and out[stop] == team:
            stop += 1
        if stop - position >= lag + settle and _probability(logit[position + lag : stop], team) < threshold:
            out[position:stop] = out[position - 1]  # won in open play, but the teams never reacted
            vetoed += 1
        position = stop
    for start, stop in _runs((out == "X") & graded):
        p_home = _probability(logit[start:stop], "H")
        if p_home >= 1 - threshold or p_home <= threshold:
            out[start:stop] = "H" if p_home >= 0.5 else "A"
            filled += 1
    return out, {"vetoed": vetoed, "filled": filled}


def refine(labels: np.ndarray, tracking: Tracking, evidence: np.ndarray, graded: np.ndarray,
           config: dict) -> tuple[np.ndarray, dict]:
    """The whole stage: differences, the per-match fit, then veto and fill."""
    differences = shape_differences(tracking, config)
    model = fit(differences, labels, evidence)
    if model is None:
        return labels, {"vetoed": 0, "filled": 0, "fitted": False}
    out, counts = apply(labels, home_logit(differences, model), graded, config)
    return out, {**counts, "fitted": True, "weights": [round(float(w), 4) for w in model["weights"]]}
