"""Restarts: whoever takes the corner, the throw-in or the free kick has the ball.

    python -m src.restarts                                   # every restart of the match, by id
    python -m src.visualization --restarts                   # look at them on the pitch (keys n / p)
    python -m src.visualization --restart 12                 # ... starting at restart 12
    python -m src.visualization --restarts --rejected        # the still balls judged in play

Most stoppages end the same way: the ball lies still (placed for a free kick, a corner or a
goal kick, held for a throw-in) and then it is played. The dead-ball list of src.dead_ball has
only the long gaps of the ball. With the ball nearly always seen, most stoppages are not in
it, so here the ball itself is used.

Candidates: stretches in which the analysed ball (data.ball) stays still, its speed (rolling
median over smooth_s) at most still_speed_ms, for at least min_still_s, and whose last still
frame, the moment it is played, lies in a half. Where the ball lies then tells the kind, and
the kind tells the team (settings under `restarts` in config.yaml):

    corner       within corner_radius_m of a corner flag: the team attacking that goal
    off pitch    further than out_margin_m beyond a line: the ball waits off the pitch, no team
    goal kick    in a goal area: the team defending that goal
    goal line    on a goal line (within line_margin_m) outside the goal area: waiting, no team
    throw-in     on a touchline (within line_margin_m, or up to out_margin_m beyond): the taker's
    kickoff      within spot_radius_m of the centre spot: the taker's
    penalty      within spot_radius_m of a penalty spot: the team attacking that goal
    keeper       anywhere else with a keeper (slot GKH / GKA) the nearest player: his team
    free kick    anywhere else: the taker's

The taker is the nearest player at the moment the ball is played. Keepers held where last seen
are left out (src/goalkeeper.py), and so is anyone further than max_player_out_m outside the
lines (the bench, players warming up).

A throw-in whose ball is not seen (held over the head, out of the picture) is found from the
thrower instead (by = "thrower"): a player standing still at least thrower_behind_m behind a
touchline for at least min_still_s, the ball - where seen - within ball_near_m of him, and back
on the pitch within back_on_s after the throw. Unless a still ball covers the same time already.

A ball also lies still in open play: a player shields it at the touchline, is fouled, pushed.
A candidate is a restart (verdict) only if, for its kind,

    it lasts      at least min_duration_s[kind]: a placed ball waits, a ball in play does not
    opponents     stand back: the nearest at least opponents_back_m[kind] away. The laws ask
                  2 m at a throw-in and 9.15 m at the others; in a duel one is right there
    the players   slow down: their median speed over the stretch (src.dead_ball.median_speed)
                  at most stoppage_speed_ms

A free kick has a story before it: a foul, a duel - an opponent at the ball, who then backs
off. A player holding the ball in the build-up has the opponents far away all along. So a free
kick also needs, in the before_s before the ball lies still, an opponent of the taker within
duel_m of the ball (duel_m column: the closest one came; NaN where the ball was not seen).

Otherwise the verdict is "in play", with the reasons. "keeper holds" (the keeper has the ball
in his hands, play goes on) and "waiting" (off pitch, goal line) are no restarts either. For
the kinds the goal decides, `agrees` says whether the taker is of that team, a check on the
call.

The restart team then has the ball. From the restart on, this is a possession label made by
code, the first piece of the possession model. The public ground truth is shown for
understanding only: tuning on it is against the rules (RULES.md).
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from src.data_loader import Tracking
from src.dead_ball import GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH, PENALTY_SPOT_DISTANCE, _runs, median_speed
from src.goalkeeper import defended_goals
from src.phases import match_clock, video_time
from src.progress import Steps

ATTACKS, DEFENDS = "attacks", "defends"
BY_GOAL = {"corner": ATTACKS, "goal kick": DEFENDS, "penalty": ATTACKS}  # the goal at that end decides
NO_TEAM = ("off pitch", "goal line")
RESTART, IN_PLAY, KEEPER_HOLDS, WAITING = "restart", "in play", "keeper holds", "waiting"
COLUMNS = ["id", "half", "match", "video", "first", "last", "duration_s", "kind", "x_m", "y_m", "team", "by",
           "taker", "taker_m", "opponent_m", "players_ms", "duel_m", "agrees", "verdict", "why"]


def still_stretches(ball: pd.DataFrame, config: dict) -> list[tuple[int, int]]:
    """(start, stop) row positions of data.ball where the ball lies still at least min_still_s."""
    settings, fps = config["restarts"], config["fps"]
    speed = ball["speed_ms"].rolling(_window(config), center=True, min_periods=1).median().to_numpy(dtype=float)
    seen = ~np.isnan(ball["x_m"].to_numpy(dtype=float))
    still = seen & (speed <= settings["still_speed_ms"])  # NaN speed is never still
    shortest = int(round(settings["min_still_s"] * fps))
    return [(int(start), int(stop)) for start, stop in _runs(still) if stop - start >= shortest]


def _window(config: dict) -> int:
    """smooth_s in frames, odd: a centred window that does not shift the edges."""
    return 2 * int(round(config["restarts"]["smooth_s"] * config["fps"] / 2)) + 1


def restart_kind(x: float, y: float, pitch, settings: dict) -> str:
    """The kind of restart for a ball lying at (x, y) in metres, before the keeper check."""
    length, width = pitch
    line, out = settings["line_margin_m"], settings["out_margin_m"]
    flags = np.array([[0, 0], [0, width], [length, 0], [length, width]])
    if np.hypot(*(flags - [x, y]).T).min() <= settings["corner_radius_m"]:
        return "corner"
    if x < -out or x > length + out or y < -out or y > width + out:
        return "off pitch"
    deep = min(x, length - x)  # distance from the nearer goal line, negative behind it
    if deep <= GOAL_AREA_DEPTH + line and abs(y - width / 2) <= GOAL_AREA_HALF_WIDTH + line:
        return "goal kick"
    if deep <= line:
        return "goal line"
    if y <= line or y >= width - line:
        return "throw-in"
    if np.hypot(x - length / 2, y - width / 2) <= settings["spot_radius_m"]:
        return "kickoff"
    if np.hypot(deep - PENALTY_SPOT_DISTANCE, y - width / 2) <= settings["spot_radius_m"]:
        return "penalty"
    return "free kick"


def _nearest(tracking: Tracking, index: int, x: float, y: float, config: dict) -> dict[str, tuple[float, int]]:
    """Per team: (distance in m, slot) of its player nearest to the ball, (inf, -1) if none."""
    length, width = config["pitch"]
    out = config["restarts"]["max_player_out_m"]
    xy = tracking.players[index].astype(float) * [length, width]
    near_pitch = ((xy[:, 0] >= -out) & (xy[:, 0] <= length + out) & (xy[:, 1] >= -out) & (xy[:, 1] <= width + out))
    distance = np.where(near_pitch, np.hypot(xy[:, 0] - x, xy[:, 1] - y), np.inf)  # NaN compares False
    nearest = {}
    for side in "HA":
        d = np.where(tracking.team == side, distance, np.inf)
        slot = int(np.argmin(d))
        nearest[side] = (float(d[slot]), slot) if np.isfinite(d[slot]) else (np.inf, -1)
    return nearest


def _verdict(kind: str, duration_s: float, opponent_m: float, players_ms: float, settings: dict,
             more: list[str] | tuple = ()) -> tuple[str, str]:
    """(verdict, why): is the still ball a restart, or in play? more: further reasons against it."""
    if kind in NO_TEAM:
        return WAITING, "the ball waits beyond a line"
    if kind == "keeper":
        return KEEPER_HOLDS, "the keeper has the ball, play goes on"
    against = []
    if duration_s < settings["min_duration_s"][kind]:
        against.append(f"too short ({duration_s:g} s)")
    if opponent_m < settings["opponents_back_m"][kind]:
        against.append(f"opponent at {opponent_m:.1f} m")
    if not players_ms <= settings["stoppage_speed_ms"]:  # NaN: nobody to tell, not a stoppage
        against.append(f"players run ({players_ms:.1f} m/s)")
    against += list(more)
    return (IN_PLAY, "; ".join(against)) if against else (RESTART, "")


def candidates(tracking: Tracking, ball: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Every still ball inside the halves, in time order, judged (columns: COLUMNS and summary).
    tracking: the players for the taker, without the keepers held at their last position
    (src.pipeline.measured); ball: the analysed ball (data.ball)."""
    settings, fps, pitch = config["restarts"], config["fps"], config["pitch"]
    length = pitch[0]
    halves = [p for p in config["phases"] if "clock_start_min" in p]
    goals = defended_goals(tracking, config)
    speed = median_speed(tracking, pitch, fps, config["dead_ball"]["speed_window_s"])
    frames = ball.index.to_numpy()
    x_m, y_m = ball["x_m"].to_numpy(dtype=float), ball["y_m"].to_numpy(dtype=float)
    window = _window(config)
    to_ball = _team_distances(tracking, ball, config)
    before = int(round(settings["before_s"] * fps))
    rows = []
    for start, stop in still_stretches(ball, config):
        played = int(frames[stop - 1])  # the last still frame: the ball is played
        # the kickoff of a half is played at its first frame: the ball may lie there from before
        half = next((h for h in halves if h["first"] - fps <= played <= h["last"]), None)
        if half is None:
            continue
        placed = slice(max(stop - window, start), stop)  # where the ball lies when it is played
        x, y = float(np.nanmedian(x_m[placed])), float(np.nanmedian(y_m[placed]))
        nearest = _nearest(tracking, played - tracking.first_frame, x, y, config)
        closest = min("HA", key=lambda side: nearest[side][0])
        kind = restart_kind(x, y, pitch, settings)
        closest_slot = nearest[closest][1]
        keeper_at_ball = closest_slot >= 0 and str(tracking.slot_ids[closest_slot]).startswith("GK")
        if kind == "free kick" and keeper_at_ball:
            kind = "keeper"
        no_penalty = ""
        if kind == "penalty":  # only with the scene of a penalty: the box cleared, nobody running off
            others, off = _penalty_scene(tracking, int(frames[start]), played, x, config)
            if others > settings["penalty_box_max"] or off > settings["penalty_spread_ms"]:
                if keeper_at_ball:
                    kind = "keeper"  # he holds it, the others run off: a throw-out to come, play
                else:
                    no_penalty = (f"no penalty scene ({others:.0f} in the box besides the keepers, "
                                  f"players move off at {off:.1f} m/s)")
        if kind in BY_GOAL:
            end = "left" if x < length / 2 else "right"
            defender = next(side for side in "HA" if goals[(half["name"], side)] == end)
            team = defender if BY_GOAL[kind] == DEFENDS else ("A" if defender == "H" else "H")
            by, agrees = "goal", "yes" if closest == team else "no"
        elif kind in NO_TEAM:
            team, by, agrees = "", "", "-"
        else:
            team, by, agrees = closest, "taker", "-"
        own = team or closest
        other = "A" if own == "H" else "H"
        taker_m, taker = nearest[own]
        opponent_m = nearest[other][0]
        first = int(frames[start])
        index = slice(first - tracking.first_frame, played - tracking.first_frame + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # no player speed in the stretch: NaN
            players_ms = float(np.nanmedian(speed[index]))
        duration_s = round((stop - start) / fps, 1)
        # the story before: how close an opponent of the taker came to the ball
        lead_in = slice(max(first - tracking.first_frame - before, 0), first - tracking.first_frame + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # the ball not seen before: NaN
            duel_m = float(np.nanmin(to_ball[other][lead_in]))
        more = [no_penalty] if no_penalty else []
        duel_needed = settings.get("duel_m")  # null in the config: no such check
        if kind == "free kick" and duel_needed is not None and duel_m > duel_needed:  # NaN: not seen, no verdict
            more.append(f"no duel before (closest opponent {duel_m:.1f} m)")
        verdict, why = _verdict(kind, duration_s, opponent_m, players_ms, settings, more)
        rows.append({
            "half": half["name"], "match": match_clock(played, config), "video": video_time(played, config),
            "first": first, "last": played, "duration_s": duration_s,
            "kind": kind, "x_m": round(x, 1), "y_m": round(y, 1), "team": team, "by": by,
            "taker": str(tracking.slot_ids[taker]) if taker >= 0 else "-",
            "taker_m": round(taker_m, 1) if np.isfinite(taker_m) else np.nan,
            "opponent_m": round(opponent_m, 1) if np.isfinite(opponent_m) else np.nan,
            "players_ms": round(players_ms, 1), "duel_m": round(duel_m, 1) if np.isfinite(duel_m) else np.nan,
            "agrees": agrees, "verdict": verdict, "why": why,
        })
    rows += _thrower_rows(tracking, ball, speed, config, [(row["first"], row["last"]) for row in rows])
    rows.sort(key=lambda row: row["first"])
    return _numbered(pd.DataFrame(rows, columns=COLUMNS[1:]))


def _penalty_scene(tracking: Tracking, first: int, played: int, x: float, config: dict) -> tuple[float, float]:
    """Over the last penalty_check_s before the ball is played (video frames first..played): the
    players in the penalty box at that end besides the keepers (median count), and how fast the
    players move away from the ball (median, m/s; negative: towards it). At a penalty the box is
    cleared but for the taker and nobody runs off; when the keeper holds the ball the box is
    full or everybody runs upfield."""
    from src.dead_ball import PENALTY_BOX_DEPTH, PENALTY_BOX_HALF_WIDTH

    settings, fps = config["restarts"], config["fps"]
    length, width = config["pitch"]
    a = max(played - int(round(settings["penalty_check_s"] * fps)), first) - tracking.first_frame
    b = played - tracking.first_frame + 1
    xy = tracking.players[a:b].astype(float) * [length, width]
    field = ~np.char.startswith(tracking.slot_ids.astype(str), "GK")
    depth = xy[:, :, 0] if x < length / 2 else length - xy[:, :, 0]
    with np.errstate(invalid="ignore"):
        in_box = (depth >= 0) & (depth <= PENALTY_BOX_DEPTH) & (np.abs(xy[:, :, 1] - width / 2) <= PENALTY_BOX_HALF_WIDTH)
    others = float(np.median((in_box & field).sum(axis=1)))
    if len(xy) < 2:
        return others, 0.0
    spot = np.array([PENALTY_SPOT_DISTANCE if x < length / 2 else length - PENALTY_SPOT_DISTANCE, width / 2])
    start, end = np.hypot(*(xy[0] - spot).T), np.hypot(*(xy[-1] - spot).T)
    moved = (end - start)[np.isfinite(start) & np.isfinite(end)]
    off = float(np.median(moved) / ((len(xy) - 1) / fps)) if len(moved) else 0.0
    return others, off


def _team_distances(tracking: Tracking, ball: pd.DataFrame, config: dict) -> dict[str, np.ndarray]:
    """Per team and frame (tracking index): its player nearest to the ball, in m (NaN: no ball, no player)."""
    frames = np.arange(tracking.first_frame, tracking.first_frame + tracking.n_frames)
    ball_xy = ball[["x_m", "y_m"]].reindex(frames).to_numpy(dtype=np.float32)
    xy = tracking.players.astype(np.float32) * np.asarray(config["pitch"], dtype=np.float32)
    distance = np.hypot(*(xy - ball_xy[:, None, :]).transpose(2, 0, 1))  # (frames, slots)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames without ball or players: NaN
        return {side: np.nanmin(distance[:, tracking.team == side], axis=1) for side in "HA"}


def _own_speeds(tracking: Tracking, config: dict) -> np.ndarray:
    """(frames, slots): every track's speed in m/s, from its positions speed_window_s before and
    after - off the pitch too, where src.dead_ball.median_speed leaves people out."""
    fps = config["fps"]
    xy = tracking.players.astype(np.float32) * np.asarray(config["pitch"], dtype=np.float32)
    k = max(int(round(config["dead_ball"]["speed_window_s"] * fps)), 1)
    speed = np.full(xy.shape[:2], np.nan, dtype=np.float32)
    if len(xy) > 2 * k:
        speed[k:-k] = np.hypot(*(xy[2 * k :] - xy[: -2 * k]).transpose(2, 0, 1)) / (2 * k / fps)
    return speed


def _thrower_rows(tracking: Tracking, ball: pd.DataFrame, speed: np.ndarray, config: dict,
                  taken: list[tuple[int, int]]) -> list[dict]:
    """Throw-ins found from the thrower (see the module docstring), outside the frames `taken`
    by the still balls. speed: the players' median speed per frame."""
    settings, fps = config["restarts"], config["fps"]
    length, width = config["pitch"]
    behind, out = settings["thrower_behind_m"], settings["max_player_out_m"]
    halves = [p for p in config["phases"] if "clock_start_min" in p]
    frames = np.arange(tracking.first_frame, tracking.first_frame + tracking.n_frames)
    in_half = np.zeros(len(frames), dtype=bool)
    for half in halves:
        in_half |= (frames >= half["first"]) & (frames <= half["last"])
    xy = tracking.players.astype(np.float32) * np.asarray([length, width], dtype=np.float32)
    x, y = xy[:, :, 0], xy[:, :, 1]
    with np.errstate(invalid="ignore"):  # NaN: no player, never a thrower
        outside = ((y < -behind) & (y >= -out)) | ((y > width + behind) & (y <= width + out))
        thrower = outside & (x >= 0) & (x <= length) & (_own_speeds(tracking, config) <= settings["still_speed_ms"])
    thrower &= in_half[:, None]
    ball_xy = ball[["x_m", "y_m"]].reindex(frames).to_numpy(dtype=float)
    seen = (ball["source"] == "observed") if "source" in ball else ball["x_m"].notna()
    seen = seen.reindex(frames, fill_value=False).to_numpy(dtype=bool)
    shortest, window = int(round(settings["min_still_s"] * fps)), _window(config)
    back_on = int(round(settings["back_on_s"] * fps))
    stretches = sorted((a, b, slot) for slot in range(thrower.shape[1]) for a, b in _runs(thrower[:, slot])
                       if b - a >= shortest)
    rows, busy = [], list(taken)
    for a, b, slot in stretches:
        first, played = int(frames[a]), int(frames[b - 1])
        if any(first <= last and other_first <= played for other_first, last in busy):
            continue
        busy.append((first, played))
        half = next(h for h in halves if h["first"] <= played <= h["last"])
        side = str(tracking.team[slot])
        tx, ty = (float(v) for v in np.nanmedian(xy[max(b - window, a) : b, slot], axis=0))
        nearest = _nearest(tracking, b - 1, tx, ty, config)
        opponent_m = nearest["A" if side == "H" else "H"][0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            players_ms = float(np.nanmedian(speed[a:b]))
        more, ball_m = [], np.nan
        if seen[a:b].any():
            away = np.hypot(*(ball_xy[a:b] - xy[a:b, slot])[seen[a:b]].T)
            if np.median(away) > settings["ball_near_m"]:
                more.append(f"ball in play elsewhere ({np.median(away):.0f} m away)")
            if seen[b - 1]:
                ball_m = float(np.hypot(*(ball_xy[b - 1] - xy[b - 1, slot])))
        later = y[b : b + back_on, slot]
        with np.errstate(invalid="ignore"):
            if not ((later >= 0) & (later <= width)).any():
                more.append(f"not back on the pitch within {settings['back_on_s']:g} s")
        duration_s = round((b - a) / fps, 1)
        verdict, why = _verdict("throw-in", duration_s, opponent_m, players_ms, settings, more)
        rows.append({
            "half": half["name"], "match": match_clock(played, config), "video": video_time(played, config),
            "first": first, "last": played, "duration_s": duration_s, "kind": "throw-in",
            "x_m": round(tx, 1), "y_m": round(ty, 1), "team": side, "by": "thrower",
            "taker": str(tracking.slot_ids[slot]), "taker_m": round(ball_m, 1) if np.isfinite(ball_m) else np.nan,
            "opponent_m": round(opponent_m, 1) if np.isfinite(opponent_m) else np.nan,
            "players_ms": round(players_ms, 1), "duel_m": np.nan, "agrees": "-", "verdict": verdict, "why": why,
        })
    return rows


def _numbered(table: pd.DataFrame) -> pd.DataFrame:
    """Ids 1, 2, ... in time order, and the line the viewer shows."""
    table = table.drop(columns=["id", "summary"], errors="ignore").reset_index(drop=True)
    table.insert(0, "id", np.arange(1, len(table) + 1))
    table["summary"] = [_summary(row) for row in table.itertuples()]
    return table


def find_restarts(tracking: Tracking, ball: pd.DataFrame, config: dict, verdict: str = RESTART) -> pd.DataFrame:
    """The candidates with this verdict (by default the restarts), numbered 1, 2, ... in time order."""
    table = candidates(tracking, ball, config)
    return _numbered(table[table["verdict"] == verdict])


def _summary(row) -> str:
    if row.verdict == WAITING:
        return f"{row.kind}: the ball waits, no restart yet (nearest {row.taker} at {row.taker_m} m)"
    if row.verdict == KEEPER_HOLDS:
        return f"keeper {row.taker} ({row.team}) holds the ball, play goes on"
    if row.verdict == IN_PLAY:
        return f"in play, no {row.kind}: {row.why} (nearest {row.taker}, {row.team})"
    if row.by == "thrower":
        ball = "not seen" if pd.isna(row.taker_m) else f"{row.taker_m} m away"
        return (f"throw-in {row.team} (by the thrower {row.taker} behind the line, ball {ball}): "
                f"nearest opponent {row.opponent_m} m, players {row.players_ms} m/s")
    text = (f"{row.kind} {row.team} (by the {row.by}): taker {row.taker} at {row.taker_m} m, "
            f"nearest opponent {row.opponent_m} m, players {row.players_ms} m/s")
    if row.kind == "free kick":
        text += f", an opponent at the ball before ({row.duel_m} m)"
    return text + (" - but the nearest player is of the other team" if row.agrees == "no" else "")


def _majority(labels: np.ndarray) -> str:
    labels = labels[np.isin(labels, ["H", "A", "D"])]
    if len(labels) == 0:
        return ""
    values, counts = np.unique(labels, return_counts=True)
    return str(values[counts.argmax()])


def with_ground_truth(table: pd.DataFrame, gt: np.ndarray, config: dict) -> pd.DataFrame:
    """Adds the public ground truth's majority label while the ball lies still (gt_still) and
    in the check_s after it is played (gt_after); "" outside it. For understanding only."""
    after = int(round(config["restarts"]["check_s"] * config["fps"]))
    table = table.copy()
    table["gt_still"] = [_majority(gt[a : b + 1]) for a, b in zip(table["first"], table["last"])]
    table["gt_after"] = [_majority(gt[b + 1 : b + 1 + after]) for b in table["last"]]
    return table


def ground_truth_table(table: pd.DataFrame) -> pd.DataFrame:
    """Per verdict and kind, over the candidates the public ground truth covers: how often it
    calls the still ball dead, and gives our team the ball afterwards. For understanding only."""
    covered = table[table["gt_after"] != ""]
    rows = {}
    for (verdict, kind), part in covered.groupby(["verdict", "kind"]):
        with_team = part[part["team"] != ""]
        rows[f"{verdict}: {kind}"] = {
            "still balls": len(part),
            "gt still = D": f"{(part['gt_still'] == 'D').mean():.0%}",
            "gt after = our team": f"{(with_team['gt_after'] == with_team['team']).mean():.0%}"
            if len(with_team) else "-"}
    return pd.DataFrame(rows).T


def main():
    from src.data_loader import DEFAULT_CONFIG, load_config
    from src.pipeline import load_match, load_public_gt, measured

    parser = argparse.ArgumentParser(description="Restarts: who takes them, and so has the ball.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["restarts"]
    steps = Steps(8)

    data = load_match(config, steps)
    with steps(f"Still balls (at least {settings['min_still_s']:g} s), judged: restart or in play"):
        table = candidates(measured(data), data.ball, config)
    restarts = _numbered(table[table["verdict"] == RESTART])

    print(f"\n{len(table)} times the ball lies still in the halves, per verdict and kind:")
    print(pd.crosstab(table["verdict"], table["kind"], margins=True, margins_name="all").to_string())
    in_play = table.loc[table["verdict"] == IN_PLAY, "why"]
    if len(in_play):
        print(f"\nwhy {len(in_play)} are in play (a candidate may fail several checks):")
        for check in ("too short", "opponent at", "players run", "no duel before", "ball in play elsewhere",
                      "not back on the pitch"):
            print(f"  {check:22s} {in_play.str.contains(check).sum()}")
    print(f"\n{len(restarts)} restarts, per kind and team:")
    print(pd.crosstab(restarts["kind"], restarts["team"], margins=True, margins_name="all").to_string()
          if len(restarts) else "  none")
    by_goal = restarts[restarts["by"] == "goal"]
    if len(by_goal):
        print(f"\ndecided by the goal, the taker is of that team: {(by_goal['agrees'] == 'yes').mean():.0%} "
              f"of {len(by_goal)}")
    gt = load_public_gt(config, data.tracking.n_frames)
    if gt is not None:
        summary = ground_truth_table(with_ground_truth(table, gt, config))
        print("\nThe public ground truth (understanding only, never tune on it), per verdict and kind:")
        print(summary.to_string() if len(summary) else "  no still ball inside it")
    print("\nThe restarts (match data - keep this list to yourself):")
    print(restarts.drop(columns=["summary", "verdict", "why"]).to_string(index=False))
    print("\nlook at them on the pitch: .venv/bin/python -m src.visualization --restart <id>   (then n / p)"
          "\nthe still balls judged in play: .venv/bin/python -m src.visualization --restarts --rejected")
    return table


if __name__ == "__main__":
    table = main()
