"""Stoppages: when the ball is out of play, and which team restarts play.

The ground truth's D is the referee's dead ball. What the inputs show of it, measured on this
match without any labels (development/README.md has the numbers):

* The tactical ball track has two kinds of gaps: a few under 1 s, and the rest 5 s or longer,
  with practically nothing in between. The long ones cover about 40% of the graded frames,
  close to the share of a professional match during which the ball is out of play.
* Inside the long gaps the players' median speed is about 1 m/s, mostly walking pace, against
  1.6 m/s while the ball is seen; the broadcast mostly cuts away to replays and close-ups; and
  Smart Tagging's set pieces begin shortly before the gaps end.

So each long gap (`judge_gaps`) is judged mainly by the players, who stop running when the
referee stops play:

  keeper  the same goalkeeper controls the ball where it vanishes and where it reappears, and
          it was not last seen at a line: he holds it, which is play (Law 12). His team has it.
  dead    the ball was last seen out of play; or the players walk (median speed at most
          walking_speed_ms); or they are slow (at most slow_speed_ms) and one more cue agrees:
          the ball last seen at a line, reappearing at a restart spot, a crowd episode, or a
          Smart Tagging set piece around the restart (fallible: it never decides alone).
  live    otherwise: play goes on unseen, and the possession model keeps the owner.

Further dead ball (`stoppage_mask`):
  * the ball observed clearly beyond a line for at least model.out_min_s
  * the waiting before an accepted restart of a placed ball (src.restarts)
  * graded frames before a half's kickoff: the ball is in play from the kickoff (Law 8), and
    the kickoff is where the half's match clock starts in the published phase table
  * a visible interruption shorter than bridge_s between two stoppages while the players still
    walk: play has not restarted, the ball was only glimpsed

Each live stretch starts with a restart (`restart_seeds`). Its team:
  1. a detected restart of a placed ball, released at the start of the stretch
  2. the first ball seen within seed_window_s: at a corner flag the attacking team, in a goal
     area the defending team (the laws); elsewhere the player in control at the ball, by the
     possession model's own control test (the taker)
  3. none: the possession model gives the stretch to the first team whose control it confirms

Every threshold is a physical or football quantity in config.yaml (`stoppages`), fixed before
any check against labels.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.data_loader import Tracking
from src.dead_ball import OUT_ZONES, RESTART_ZONES, _runs
from src.goalkeeper import defended_goals
from src.phases import clock_phase, graded_mask, half_of_frames, match_seconds
from src.restarts import _nearest, restart_kind

DEAD, LIVE, KEEPER = "dead", "live", "keeper"
JUDGED_COLUMNS = ["verdict", "team", "cues", "why"]


def control_team(tracking: Tracking, frame: int, x_m: float, y_m: float, config: dict) -> tuple[str, int]:
    """(team, slot) of the player in control at the ball: within model.control_radius_m and
    more than model.control_margin_m nearer than every opponent; ("", -1) if nobody is.

    The same test the possession model applies to every frame, at a single frame."""
    if not (np.isfinite(x_m) and np.isfinite(y_m)):
        return "", -1
    model = config["model"]
    near = _nearest(tracking, frame - tracking.first_frame, x_m, y_m, config)
    for side, other in (("H", "A"), ("A", "H")):
        distance, slot = near[side]
        if distance <= model["control_radius_m"] and distance + model["control_margin_m"] < near[other][0]:
            return side, slot
    return "", -1


def _xy(ball: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    first = int(ball.index[0]) if len(ball) else 0
    if not np.array_equal(ball.index.to_numpy(), np.arange(first, first + len(ball))):
        raise ValueError("The ball must have one row per consecutive video frame")
    return ball["x_m"].to_numpy(dtype=float), ball["y_m"].to_numpy(dtype=float), first


def keeper_holds(gap, tracking: Tracking, ball: pd.DataFrame, config: dict) -> str:
    """The keeper's team if the same goalkeeper is in control where the ball vanishes and where
    it reappears, and the ball was not last seen at a line; "" otherwise."""
    if gap.ball_before in OUT_ZONES or "-" in (gap.ball_before, gap.ball_after):
        return ""
    x, y, first = _xy(ball)
    ends = []
    for frame in (int(gap.first) - 1, int(gap.last) + 1):
        if not 0 <= frame - first < len(x):
            return ""
        ends.append(control_team(tracking, frame, x[frame - first], y[frame - first], config))
    (team, slot), (_, slot_after) = ends
    if team and slot == slot_after and str(tracking.slot_ids[slot]).startswith("GK"):
        return team
    return ""


def set_piece_at(frame: int, set_pieces: pd.DataFrame, tolerance: int) -> str:
    """The type of a tagged set piece whose interval contains this frame (+- tolerance), or ""."""
    if set_pieces is None or set_pieces.empty:
        return ""
    hit = set_pieces[(set_pieces["first"] - tolerance <= frame) & (frame <= set_pieces["last"] + tolerance)]
    return str(hit["type"].iloc[0] or "set piece") if len(hit) else ""


def judge_gaps(gaps: pd.DataFrame, tracking: Tracking, ball: pd.DataFrame, set_pieces: pd.DataFrame | None,
               config: dict) -> pd.DataFrame:
    """The long gaps of the tactical ball (src.dead_ball.long_gaps) with verdict (dead, live,
    keeper), team (the keeper's), the corroborating cues and why; see the module docstring."""
    settings = config["stoppages"]
    tolerance = round(config.get("smart_tagging", {}).get("tolerance_s", 1.0) * config["fps"])
    rows = []
    for gap in gaps.itertuples():
        cues = []
        if gap.ball_before in OUT_ZONES:
            cues.append(f"last seen: {gap.ball_before}")
        if gap.ball_after in RESTART_ZONES:
            cues.append(f"reappears: {gap.ball_after}")
        if getattr(gap, "crowded", False):
            cues.append("crowd")
        tagged = set_piece_at(int(gap.last) + 1, set_pieces, tolerance)
        if tagged:
            cues.append(f"tagged: {tagged}")
        speed = gap.speed_ms
        keeper = keeper_holds(gap, tracking, ball, config)
        if keeper:
            verdict, why = KEEPER, "the goalkeeper holds the ball"
        elif gap.ball_before == "out of play":
            verdict, why = DEAD, "ball last seen out of play"
        elif speed <= settings["walking_speed_ms"]:  # NaN (nobody tracked) is never walking
            verdict, why = DEAD, f"players walk ({speed:.1f} m/s)"
        elif speed <= settings["slow_speed_ms"] and cues:
            verdict, why = DEAD, f"players slow ({speed:.1f} m/s) and corroborated"
        elif speed <= settings["slow_speed_ms"]:
            verdict, why = LIVE, f"players slow ({speed:.1f} m/s), uncorroborated"
        else:
            verdict, why = LIVE, f"players run ({speed:.1f} m/s)" if np.isfinite(speed) else "no players tracked"
        rows.append({"verdict": verdict, "team": keeper, "cues": "; ".join(cues), "why": why})
    # long_gaps' own verdict only lists evidence ("dead" if any); this verdict replaces it.
    evidence = gaps.drop(columns=["verdict", "summary"], errors="ignore").rename(columns={"reasons": "evidence"})
    return pd.concat([evidence.reset_index(drop=True), pd.DataFrame(rows, columns=JUDGED_COLUMNS)], axis=1)


def _outside_runs(ball: pd.DataFrame, half: np.ndarray, config: dict) -> list[tuple[int, int]]:
    """Runs of the observed ball clearly beyond a line, at least model.out_min_s long."""
    x, y, _ = _xy(ball)
    margin = config["dead_ball"]["line_margin_m"]
    length, width = config["pitch"]
    with np.errstate(invalid="ignore"):
        outside = ((ball["source"].to_numpy() == "observed") & np.isfinite(x) & np.isfinite(y)
                   & ((x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)))
    shortest = max(1, round(config["model"].get("out_min_s", 0.4) * config["fps"]))
    runs = []
    for side in np.unique(half[half > 0]):
        runs += [(a, b) for a, b in _runs(outside & (half == side)) if b - a >= shortest]
    return runs


def bridge_interruptions(dead: np.ndarray, speed: np.ndarray, half: np.ndarray, config: dict) -> np.ndarray:
    """dead, with each live stretch shorter than bridge_s between two stoppages of the same half
    dead too, if the players' median speed over it stays at walking pace."""
    settings = config["stoppages"]
    limit = round(settings["bridge_s"] * config["fps"])
    out = dead.copy()
    for side in np.unique(half[half > 0]):
        inside = half == side
        for a, b in _runs(inside & ~dead):
            if b - a >= limit or a == 0 or b >= len(dead):
                continue
            if not (inside[a - 1] and dead[a - 1] and inside[b] and dead[b]):
                continue  # the start or the end of a half is no stoppage
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                walking = np.nanmedian(speed[a:b])
            if walking <= settings["walking_speed_ms"]:
                out[a:b] = True
    return out


def stoppage_mask(n_frames: int, judged: pd.DataFrame, ball: pd.DataFrame, restarts: pd.DataFrame,
                  speed: np.ndarray, config: dict) -> np.ndarray:
    """Per frame: is the ball out of play? See the module docstring."""
    graded = graded_mask(n_frames, config)
    half = half_of_frames(n_frames, config)
    dead = np.zeros(n_frames, dtype=bool)
    for gap in judged[judged["verdict"] == DEAD].itertuples():
        dead[int(gap.first) : int(gap.last) + 1] = True
    for a, b in _outside_runs(ball, half, config):
        dead[a:b] = True
    for restart in restarts.itertuples():
        dead[int(restart.first) : int(restart.last)] = True  # the last still frame is the release
    dead |= graded & np.isnan(match_seconds(np.arange(n_frames), config))  # before a half's kickoff
    dead = bridge_interruptions(dead, speed, half, config)
    dead[~graded] = False
    return dead


def live_stretches(dead: np.ndarray, config: dict) -> list[tuple[int, int]]:
    """(start, stop) of the graded live stretches, never across a half."""
    n = len(dead)
    graded, half = graded_mask(n, config), half_of_frames(n, config)
    out = []
    for side in np.unique(half[half > 0]):
        out += _runs(graded & ~dead & (half == side))
    return sorted(out)


def restart_seeds(dead: np.ndarray, ball: pd.DataFrame, tracking: Tracking, restarts: pd.DataFrame,
                  config: dict) -> pd.DataFrame:
    """One row per live stretch: start, stop, the restarting team ("" if unknown) and how it was
    found (placed-ball restart, corner, goal kick, taker, none)."""
    x, y, first = _xy(ball)
    length = config["pitch"][0]
    observed = (ball["source"].to_numpy() == "observed") & np.isfinite(x) & np.isfinite(y)
    window = round(config["stoppages"]["seed_window_s"] * config["fps"])  # 0 disables restart teams
    released = {int(r.last): str(r.team) for r in restarts.itertuples() if r.team in ("H", "A")}
    goals = defended_goals(tracking, config)
    rows = []
    for a, b in live_stretches(dead, config):
        team, how = "", "none"
        stop = min(b, a + window)
        frame = next((f for f in range(a, stop) if f in released), None)
        if window <= 0:
            pass
        elif frame is not None:
            team, how = released[frame], "placed-ball restart"
        else:
            seen = np.flatnonzero(observed[a - first : stop - first])
            if seen.size:
                f = a + int(seen[0])
                fx, fy = x[f - first], y[f - first]
                kind = restart_kind(fx, fy, config["pitch"], config["restarts"])
                phase = clock_phase(f, config)
                if kind in ("corner", "goal kick") and phase is not None:
                    end = "left" if fx < length / 2 else "right"
                    defender = next(side for side in "HA" if goals[(phase["name"], side)] == end)
                    team = defender if kind == "goal kick" else ("A" if defender == "H" else "H")
                    how = kind
                else:
                    team, _ = control_team(tracking, f, fx, fy, config)
                    how = "taker" if team else "none"
        rows.append({"start": a, "stop": b, "team": team, "how": how})
    return pd.DataFrame(rows, columns=["start", "stop", "team", "how"])
