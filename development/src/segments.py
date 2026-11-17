"""The match cut at the dead balls, and who has the ball before and after each.

    python -m src.segments                       # the dead-ball events, possession around them, live stretches
    python -m src.visualization --segments       # step through the events (keys n / p)
    python -m src.visualization --segment 12     # ... starting at event 12

Dead-ball events (`dead_ball_events`): the restarts of src/restarts.py (the ball lying still
until it is played, verdict restart) and the long gaps of the ball judged dead (src/dead_ball.py),
merged where they overlap or lie within merge_s of each other. The crowds of src.dead_ball are
evidence, not events, and stay out. An event is clear if it ends in a restart: then its kind,
its team and the moment the ball is played (played) are known. Otherwise it ends where the
ball is seen again. Settings under `segments` in config.yaml.

Where the stoppage begins (`stoppage_starts`): the event begins where the ball comes to rest,
but play stopped earlier. Looking back from there - at most max_lookback_s, never past the
event before:

    ball out       the ball more than out_margin_m beyond a line for at least out_min_s - robust
                   against the tracking error of the ball near a line: the first frame of it. Stretches
                   out less than rejoin_s apart count as one (the ball fetched along the line). Only
                   if the ball comes to rest at most max_out_gap_s after it was last out - an old
                   out far back is no part of this stoppage.
    players slow   else, if the players walk (their median speed at most slow_speed_ms) when the
                   ball comes to rest: where they started to, reaction_s earlier (the whistle).
    -              else where the ball comes to rest.

The dead ball runs from start to played; dead_s is its length.

Possession around an event (`possession`): someone controls the ball where a player is within
control_m of it - rather than the ball flying over the heads. Over the frames with control:

    after_team     the team of the nearest player, by majority, in the after_s from played;
                   no such frame: the restart's team. after_share: the majority's share.
    before_team    the same in the before_s up to start: who had the ball when play stopped.
    last_touch     the team of the nearest player in the last frame with control before start.
                   At a throw-in, corner or goal kick the other team plays it out, so out_check
                   says "ok" if last_touch is not the restart's team, "odd" if it is.

Between the events, inside the halves: the live stretches, from played + 1 to the next start - 1,
with the team that starts them and the one that ends them. The public ground truth is shown for
understanding only: tuning on it is against the rules.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data_loader import Tracking
from src.dead_ball import _runs, median_speed
from src.phases import match_clock, video_time
from src.progress import Steps
from src.restarts import RESTART, _majority, _team_distances

GAP = "gap"
OUT_KINDS = ("throw-in", "corner", "goal kick")  # the ball went out: the other team played it out


def _played_by_keeper(goals: dict, half: str, ball: pd.DataFrame, gap_last: int, config: dict) -> str:
    """The team whose keeper played the ball, if the ball, seen again in flight after a gap, comes
    out of a penalty box: run back along its flight (at most catch_back_m), the first box it
    reaches is that team's own. "" otherwise. The keeper's tracked position is not used: when he
    is out of the picture it is only where he was last seen, often far off.
    goals: (half, team) -> the goal it defends (src.goalkeeper.defended_goals)."""
    from src.dead_ball import PENALTY_BOX_DEPTH, PENALTY_BOX_HALF_WIDTH

    settings, fps = config["segments"], config["fps"]
    length, width = config["pitch"]
    look = max(int(round(settings["catch_look_s"] * fps)), 1)
    after = ball[["x_m", "y_m"]].reindex(range(gap_last + 1, gap_last + 2 + look)).to_numpy(dtype=float)
    seen = after[np.isfinite(after).all(axis=1)]
    if len(seen) < 2:
        return ""
    flight = seen[-1] - seen[0]
    if np.hypot(*flight) / ((len(seen) - 1) / fps) < settings["catch_pass_min_ms"]:
        return ""  # not a pass: the ball rolls or lies, its origin cannot be run back
    back = -flight / np.hypot(*flight)
    for step in np.arange(0.0, settings["catch_back_m"] + 0.25, 0.5):
        x, y = seen[0] + back * step
        if abs(y - width / 2) > PENALTY_BOX_HALF_WIDTH:
            continue
        end = "left" if 0 <= x <= PENALTY_BOX_DEPTH else "right" if length - PENALTY_BOX_DEPTH <= x <= length else ""
        if end:
            return next((side for side in "HA" if goals.get((half, side)) == end), "")
    return ""


def keeper_catches(gaps: pd.DataFrame, tracking: Tracking, ball: pd.DataFrame, config: dict,
                   keepers: Tracking | None = None) -> pd.DataFrame:
    """The long gaps of the ball that are a keeper holding it (first, last, team, keeper, how), and
    not last seen out of play (then it is a goal kick, a dead ball):

        vanished at keeper  the ball vanished within catch_m of a keeper, nearer than anyone else
        played out of his box  the ball turns up again in flight, and run back along its flight
                            it comes out of a penalty box (_played_by_keeper): that box's keeper
                            caught it unseen and threw or kicked it out

    A catch is play, not a dead ball (Law 12); his team has the ball. tracking: without the held
    keepers (src.pipeline.measured); keepers: no longer used (kept for the call)."""
    from src.dead_ball import OUT_ZONES
    from src.restarts import _nearest

    from src.goalkeeper import defended_goals

    catch_m = config["segments"]["catch_m"]
    goals = defended_goals(tracking, config)
    xy = ball[["x_m", "y_m"]]
    rows = []
    for gap in (gaps.itertuples() if len(gaps) else []):
        if getattr(gap, "ball_before", "-") in OUT_ZONES:
            continue
        team = keeper = how = ""
        if gap.first - 1 in xy.index:
            x, y = xy.loc[gap.first - 1]
            if np.isfinite(x) and np.isfinite(y):
                nearest = _nearest(tracking, gap.first - 1 - tracking.first_frame, float(x), float(y), config)
                side = min("HA", key=lambda s: nearest[s][0])
                distance, slot = nearest[side]
                if slot >= 0 and distance <= catch_m and str(tracking.slot_ids[slot]).startswith("GK"):
                    team, keeper, how = side, str(tracking.slot_ids[slot]), "vanished at keeper"
        if not team:
            team = _played_by_keeper(goals, gap.half, ball, int(gap.last), config)
            keeper, how = (f"GK{team}", "played out of his box") if team else ("", "")
        if team:
            rows.append({"first": int(gap.first), "last": int(gap.last), "team": team, "keeper": keeper, "how": how})
    return pd.DataFrame(rows, columns=["first", "last", "team", "keeper", "how"])


def dead_ball_events(still_balls: pd.DataFrame, gaps: pd.DataFrame, config: dict,
                     catches: pd.DataFrame | None = None) -> pd.DataFrame:
    """The dead-ball events in time order: id, half, first, last, played (the frame the ball is
    played, the last one of the event), kind ("throw-in", "gap", "gap + throw-in", ...), team
    (the restart's), clear (it ends in a restart), match, video, duration_s.
    catches: gaps that are a keeper holding the ball (keeper_catches) - no dead balls."""
    fps = config["fps"]
    merge = int(round(config["segments"]["merge_s"] * fps))
    restarts = still_balls[still_balls["verdict"] == RESTART]
    dead = gaps[gaps["verdict"] == "dead"] if len(gaps) else gaps
    if catches is not None and len(catches) and len(dead):
        dead = dead[~dead["first"].isin(catches["first"])]
    parts = sorted([(int(r.first), int(r.last), r.kind, r.team, r.half) for r in restarts.itertuples()]
                   + [(int(g.first), int(g.last), GAP, "", g.half) for g in dead.itertuples()])
    events = []
    for first, last, kind, team, half in parts:
        if events and first <= events[-1]["last"] + merge and half == events[-1]["half"]:
            event = events[-1]
            event["kinds"].append(kind)
            if kind != GAP and last >= event.get("restart_last", -1):  # the latest restart in it decides
                event.update(restart_last=last, restart=kind, team=team)
            event["last"] = max(event["last"], last)
        else:
            event = {"half": half, "first": first, "last": last, "kinds": [kind], "team": "", "restart": ""}
            if kind != GAP:
                event.update(restart_last=last, restart=kind, team=team)
            events.append(event)
    rows = []
    for event in events:
        clear = bool(event["restart"]) and event["restart_last"] >= event["last"] - merge
        played = event["restart_last"] if clear else event["last"]
        kinds = list(dict.fromkeys(event["kinds"]))  # in order, once each
        rows.append({"half": event["half"], "match": match_clock(played, config), "video": video_time(played, config),
                     "first": event["first"], "last": event["last"], "played": played,
                     "duration_s": round((event["last"] - event["first"] + 1) / fps, 1),
                     "kind": " + ".join(kinds), "restart": event["restart"] if clear else "",
                     "team": event["team"] if clear else "", "clear": clear})
    columns = ["half", "match", "video", "first", "last", "played", "duration_s", "kind", "restart", "team", "clear"]
    table = pd.DataFrame(rows, columns=columns)
    table.insert(0, "id", np.arange(1, len(table) + 1))
    return table


def stoppage_starts(events: pd.DataFrame, tracking: Tracking, ball: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Adds start (the frame play stopped), start_by ("ball out", "players slow", "-") and dead_s
    (start to played), see the module docstring. tracking: without the held keepers."""
    settings, fps = config["segments"], config["fps"]
    length, width = config["pitch"]
    frames = np.arange(tracking.first_frame, tracking.first_frame + tracking.n_frames)
    x, y = (ball[column].reindex(frames).to_numpy(dtype=float) for column in ("x_m", "y_m"))
    margin = settings["out_margin_m"]
    with np.errstate(invalid="ignore"):  # NaN: the ball not seen, not out
        out = (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)
    shortest = max(int(round(settings.get("out_min_s", 0.0) * fps)), 1)
    for first, stop in _runs(out):  # a glimpse beyond the line is tracking error, not a ball out
        if stop - first < shortest:
            out[first:stop] = False
    window = 2 * int(round(settings["smooth_s"] * fps / 2)) + 1
    speed = median_speed(tracking, config["pitch"], fps, config["dead_ball"]["speed_window_s"])
    speed = pd.Series(speed).rolling(window, center=True, min_periods=1).median().to_numpy()
    with np.errstate(invalid="ignore"):
        slow = speed <= settings["slow_speed_ms"]
    rejoin, reaction = int(round(settings["rejoin_s"] * fps)), int(round(settings["reaction_s"] * fps))
    lookback, out_gap = int(round(settings["max_lookback_s"] * fps)), int(round(settings["max_out_gap_s"] * fps))
    halves = {p["name"]: p for p in config["phases"] if "clock_start_min" in p}
    events = events.sort_values("first", kind="stable").copy()
    starts, how, previous = [], [], {}
    for row in events.itertuples():
        lo = max(row.first - lookback, previous.get(row.half, -1) + 1, halves[row.half]["first"])
        a, b = lo - tracking.first_frame, row.first - tracking.first_frame  # the window, b included
        previous[row.half] = row.played
        start, by = row.first, "-"
        runs = _runs(out[a : b + 1]) if b >= a else []
        if runs and b - a - (runs[-1][1] - 1) <= out_gap:
            first_out = runs[-1][0]
            for earlier_start, earlier_stop in reversed(runs[:-1]):  # out along the line: one stretch
                if first_out - earlier_stop > rejoin:
                    break
                first_out = earlier_start
            start, by = int(frames[a + first_out]), "ball out"
        elif b >= a and slow[b]:
            moving = np.flatnonzero(~slow[a : b + 1])
            since = moving[-1] + 1 if len(moving) else 0
            start, by = max(int(frames[a + since]) - reaction, lo), "players slow"
        starts.append(min(start, row.first))
        how.append(by)
    events["start"], events["start_by"] = np.array(starts, dtype=np.int64), how  # typed, also with no event
    events["dead_s"] = ((events["played"].astype(float) - events["start"] + 1) / fps).round(1)
    return events.sort_index()


def _control(tracking: Tracking, ball: pd.DataFrame, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per frame (tracking index): someone within control_m of the ball, and the nearest player's team."""
    near = _team_distances(tracking, ball, config)
    with np.errstate(invalid="ignore"):  # NaN: no ball or no player
        control = np.fmin(near["H"], near["A"]) <= config["segments"]["control_m"]
        team = np.where(np.isnan(near["A"]) | (near["H"] <= near["A"]), "H", "A")
    return control, team


def _held_by(control: np.ndarray, team: np.ndarray, window: slice) -> tuple[str, float, int, str]:
    """(majority team, its share, frames with control, team in the last of them) in a window."""
    held = team[window][control[window]]
    if len(held) == 0:
        return "", np.nan, 0, ""
    values, counts = np.unique(held, return_counts=True)
    return str(values[counts.argmax()]), round(counts.max() / len(held), 2), len(held), str(held[-1])


def possession(events: pd.DataFrame, tracking: Tracking, ball: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Adds who has the ball after the event (after_team, after_share, control_frames, agrees
    with the restart's team) and before it (before_team, before_share, last_touch, out_check),
    and the summary. events: with start (stoppage_starts); tracking: without the held keepers."""
    settings, fps = config["segments"], config["fps"]
    after, before = int(round(settings["after_s"] * fps)), int(round(settings["before_s"] * fps))
    control, team = _control(tracking, ball, config)
    offset = tracking.first_frame
    events = events.copy()
    rows = []
    for row in events.itertuples():
        after_team, after_share, frames_after, _ = _held_by(control, team,
                                                            slice(row.played + 1 - offset, row.played + 1 + after - offset))
        before_team, before_share, _, last_touch = _held_by(control, team,
                                                            slice(max(row.start - before - offset, 0), row.start - offset))
        out_check = "-"
        if row.restart in OUT_KINDS and row.team and last_touch:
            out_check = "ok" if last_touch != row.team else "odd"
        rows.append({"after_team": after_team or row.team, "after_share": after_share, "control_frames": frames_after,
                     "before_team": before_team, "before_share": before_share, "last_touch": last_touch,
                     "out_check": out_check})
    for column in ("after_team", "after_share", "control_frames", "before_team", "before_share", "last_touch",
                   "out_check"):
        events[column] = [r[column] for r in rows]
    events["agrees"] = ["-" if not (r and a) else ("yes" if r == a else "no")
                        for r, a in zip(events["team"], events["after_team"])]
    events["summary"] = [_summary(row, config) for row in events.itertuples()]
    return events


def _summary(row, config: dict) -> str:
    settings = config["segments"]
    what = f"{row.restart} {row.team}" if row.clear else "not clear (no restart at its end)"
    if row.kind not in (row.restart, ""):
        what += f" [{row.kind}]"
    stopped = (f"stopped {match_clock(row.start, config)} ({row.start_by}), dead {row.dead_s} s"
               if row.start_by != "-" else f"dead {row.dead_s} s")
    before = f"before: {row.before_team} ({row.before_share:.0%})" if row.before_team else "before: no one at the ball"
    if row.out_check == "odd":
        before += f", but {row.last_touch} played it out"
    if not row.after_team:
        after = "after: no one at the ball"
    elif row.control_frames == 0:
        after = f"after: {row.after_team} (the restart's team, no one within {settings['control_m']:g} m)"
    else:
        after = f"after: {row.after_team} ({row.after_share:.0%} of {row.control_frames} frames)"
    return (f"{what}; {stopped}; {before}; {after}"
            + (" - not the restart's team" if row.agrees == "no" else ""))


def possession_after(events: pd.DataFrame, tracking: Tracking, ball: pd.DataFrame, config: dict) -> pd.DataFrame:
    """stoppage_starts, then possession: both sides of every event."""
    return possession(stoppage_starts(events, tracking, ball, config), tracking, ball, config)


def live_stretches(events: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Between the events, inside the halves (from played + 1 to the next start - 1): first, last,
    team (who has the ball at its start: the after_team of the event before; "" at the start of
    a half), end_team (who has it at its end: the before_team of the next event; "" at the end
    of a half), duration_s."""
    rows = []
    for half in (p for p in config["phases"] if "clock_start_min" in p):
        inside = events[events["half"] == half["name"]].sort_values("first")
        begin, team = half["first"], ""
        for event in inside.itertuples():
            start = getattr(event, "start", event.first)
            if start > begin:
                rows.append({"half": half["name"], "first": begin, "last": start - 1, "team": team,
                             "end_team": getattr(event, "before_team", "")})
            begin, team = event.played + 1, event.after_team
        if begin <= half["last"]:
            rows.append({"half": half["name"], "first": begin, "last": half["last"], "team": team, "end_team": ""})
    table = pd.DataFrame(rows, columns=["half", "first", "last", "team", "end_team"])
    table["duration_s"] = ((table["last"] - table["first"] + 1) / config["fps"]).round(1)
    return table


def ground_truth_check(events: pd.DataFrame, gt: np.ndarray, config: dict) -> dict[str, str]:
    """For understanding only: over the events the public ground truth covers, how much of the
    dead balls (start to played) it calls D - also of the part before the ball comes to rest -
    and how often its label after and before is ours."""
    settings, fps = config["segments"], config["fps"]
    after, before = int(round(settings["after_s"] * fps)), int(round(settings["before_s"] * fps))

    def graded(spans):
        parts = [gt[a : b + 1] for a, b in spans]
        joined = np.concatenate(parts) if parts else np.array([])
        return joined[np.isin(joined, ["H", "A", "D"])]

    def share_d(labels):
        return f"{(labels == 'D').mean():.0%} of {len(labels)}" if len(labels) else "-"

    def agreement(labels, ours):
        pairs = [(g, t) for g, t in zip(labels, ours) if g and t]
        return f"{np.mean([g == t for g, t in pairs]):.0%} of {len(pairs)}" if pairs else "-"

    starts = events["start"] if "start" in events else events["first"]
    check = {"D in the dead balls (start to played)": share_d(graded(zip(starts, events["played"]))),
             "D in the part before the ball rests": share_d(graded((s, f - 1) for s, f in zip(starts, events["first"])
                                                                   if f > s)),
             "its label after = our after_team": agreement([_majority(gt[p + 1 : p + 1 + after])
                                                            for p in events["played"]], events["after_team"])}
    if "before_team" in events:
        check["its label before = our before_team"] = agreement(
            [_majority(gt[max(s - before, 0) : s]) for s in starts], events["before_team"])
    return check


def find_segments(data, config: dict, dead=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(events with the possession around them, live stretches) for the match of src.pipeline.load_match."""
    from src.pipeline import dead_ball_analysis, measured
    from src.restarts import candidates

    dead = dead if dead is not None else dead_ball_analysis(data, config)
    tracking = measured(data)
    catches = (keeper_catches(dead.gaps, tracking, data.ball, config, keepers=data.tracking)
               if config["segments"].get("catches", True) else pd.DataFrame(columns=["first", "last", "team"]))
    events = dead_ball_events(candidates(tracking, data.ball, config), dead.gaps, config, catches)
    events = possession_after(events, tracking, data.ball, config)
    events.attrs["catches"] = catches  # the keeper holds the unseen ball: play, his team (for the labels)
    return events, live_stretches(events, config)


def main():
    from src.data_loader import DEFAULT_CONFIG, load_config
    from src.pipeline import load_match, load_public_gt

    parser = argparse.ArgumentParser(description="The match cut at the dead balls; possession around each.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["segments"]
    steps = Steps(8)

    data = load_match(config, steps)
    with steps("Dead-ball events: where play stopped, and who has the ball before and after each"):
        events, live = find_segments(data, config)

    clear = events[events["clear"]]
    print(f"\n{len(events)} dead-ball events in the halves, {len(clear)} of them clear (ending in a restart), per kind:")
    print(events["kind"].value_counts().to_string() if len(events) else "  none")
    if len(events):
        print("\nwhere play stopped, found by:")
        print(events["start_by"].value_counts().to_string())
        print(f"dead balls: {events['dead_s'].sum() / 60:.1f} min, of which before the ball rests "
              f"{(events['first'] - events['start']).sum() / config['fps'] / 60:.1f} min")
    if len(clear):
        print("\nthe clear ones: who has the ball before (rows) and after (columns):")
        print(pd.crosstab(clear["before_team"].replace("", "-"), clear["after_team"].replace("", "-"),
                          margins=True, margins_name="all").to_string())
        print(f"after: the restart's team in {(clear['agrees'] == 'yes').mean():.0%}; no one within "
              f"{settings['control_m']:g} m in the {settings['after_s']:g} s after: {(clear['control_frames'] == 0).sum()}")
        out = clear[clear["out_check"] != "-"]
        if len(out):
            print(f"throw-ins, corners, goal kicks: played out by the other team in "
                  f"{(out['out_check'] == 'ok').mean():.0%} of {len(out)} (odd: {(out['out_check'] == 'odd').sum()})")
    if len(live):
        same = live[(live["team"] != "") & (live["end_team"] != "")]
        print(f"\n{len(live)} live stretches, {live['duration_s'].sum() / 60:.1f} min; median "
              f"{live['duration_s'].median():.0f} s, longest {live['duration_s'].max():.0f} s")
        if len(same):
            print(f"the same team at both ends: {(same['team'] == same['end_team']).mean():.0%} of {len(same)}")
    gt = load_public_gt(config, data.tracking.n_frames)
    if gt is not None:
        print("\nThe public ground truth (understanding only, never tune on it):")
        for name, value in ground_truth_check(events, gt, config).items():
            print(f"  {name:38s} {value}")
    print("\nThe events (match data - keep this list to yourself):")
    print(events.drop(columns="summary").to_string(index=False))
    print("\nlook at them on the pitch: .venv/bin/python -m src.visualization --segment <id>   (then n / p)")
    return events, live


if __name__ == "__main__":
    events, live = main()
