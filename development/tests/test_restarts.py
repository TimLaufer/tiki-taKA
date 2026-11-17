"""Restarts on a made-up half of 1000 frames on a 100 m x 50 m pitch. H defends the left goal
(it stands in the left half at the kickoff), A the right one. The ball lies still six times
for 3.2 s, and rolls in between; the players stand (median speed 0):

    frames    ball            nearest player
    0-79      centre spot     H1 at the ball            kickoff H
    150-229   corner right    H1 at the flag            corner H (A defends that goal)
    300-379   touchline       A2, just outside          throw-in A
    450-529   goal area left  GKH                       goal kick H
    600-679   (70, 30)        A2, H1 10 m away          free kick A
    750-829   (92, 30)        GKA                       the keeper holds the ball: no restart
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.data_loader import Tracking
from src.restarts import (
    candidates,
    find_restarts,
    ground_truth_table,
    restart_kind,
    still_stretches,
    with_ground_truth,
)

PITCH = [100.0, 50.0]
T = 1000
SETTINGS = {"still_speed_ms": 1.5, "smooth_s": 0.4, "min_still_s": 1.0, "corner_radius_m": 2.5,
            "spot_radius_m": 1.5, "line_margin_m": 1.5, "out_margin_m": 3.0, "max_player_out_m": 5.0,
            "min_duration_s": {"throw-in": 2.0, "free kick": 3.0, "corner": 3.0, "goal kick": 3.0, "penalty": 3.0,
                               "kickoff": 1.0},
            "opponents_back_m": {"throw-in": 1.5, "free kick": 3.0, "corner": 4.0, "goal kick": 6.0, "penalty": 6.0,
                                 "kickoff": 6.0},
            "stoppage_speed_ms": 2.0, "check_s": 2.0, "thrower_behind_m": 0.3, "ball_near_m": 5.0,
            "before_s": 10.0, "duel_m": 2.0, "penalty_check_s": 2.0, "penalty_box_max": 1, "penalty_spread_ms": 1.0,
            "back_on_s": 3.0}
CONFIG = {
    "fps": 25,
    "pitch": PITCH,
    "phases": [{"name": "first_half", "first": 0, "last": T - 1, "half": 1, "graded": True, "clock_start_min": 0}],
    "restarts": SETTINGS,
    "dead_ball": {"speed_window_s": 0.4},
}
STILL = {(0, 80): (50.0, 25.0), (150, 230): (100.0, 0.0), (300, 380): (60.0, 50.5), (450, 530): (3.0, 25.0),
         (600, 680): (70.0, 30.0), (750, 830): (92.0, 30.0)}


def made_up_half():
    pos = np.full((T, 6, 2), np.nan)
    pos[:] = [[49.5, 25.0], [30.0, 20.0], [2.0, 25.0], [60.0, 20.0], [70.0, 35.0], [98.0, 25.0]]
    pos[150:230, 0] = [99.5, 0.8]  # H1 at the corner flag
    pos[300:380, 4] = [60.0, 51.2]  # A2 throws in from just outside
    pos[600:680, 4] = [70.5, 30.0]  # A2 at the free kick ...
    pos[600:680, 0] = [60.0, 30.0]  # ... H1 10 m away
    pos[750:830, 5] = [92.5, 30.0]  # the A keeper holds the ball
    tracking = Tracking(
        ball=np.full((T, 2), np.nan, np.float32), ball_z=np.full(T, np.nan, np.float32),
        ball_speed=np.full(T, np.nan, np.float32), players=(pos / PITCH).astype(np.float32),
        speed=np.full((T, 6), np.nan, np.float32), speed_unit="", team=np.array(list("HHHAAA")),
        slot_ids=np.array(["1", "2", "GKH", "26", "27", "GKA"]), team_names=np.array(["Home", "Away"]),
        home_attacks_first_half="left_to_right",
    )
    xy = np.tile([50.0, 25.0], (T, 1))
    speed = np.full(T, 10.0)  # rolling ...
    for (a, b), spot in STILL.items():
        xy[a:b], speed[a:b] = spot, 0.0  # ... except where it lies still
    ball = pd.DataFrame({"x_m": xy[:, 0], "y_m": xy[:, 1], "speed_ms": speed}, index=pd.RangeIndex(T, name="frame"))
    return tracking, ball


def at(tracking, frames, slot, xy):
    players = tracking.players.copy()
    players[frames, slot] = np.array(xy) / PITCH
    return replace(tracking, players=players)


@pytest.fixture(scope="module")
def judged():
    tracking, ball = made_up_half()
    return candidates(tracking, ball, CONFIG)


def test_still_stretches():
    _, ball = made_up_half()
    assert still_stretches(ball, CONFIG) == list(STILL)
    short = ball.copy()
    short.loc[750:829, "speed_ms"] = 10.0
    short.loc[770:780, "speed_ms"] = 0.0  # under a second
    assert (750, 830) not in still_stretches(short, CONFIG)


def test_kind_team_and_verdict_of_every_still_ball(judged):
    assert list(judged["kind"]) == ["kickoff", "corner", "throw-in", "goal kick", "free kick", "keeper"]
    assert list(judged["team"]) == ["H", "H", "A", "H", "A", "A"]
    assert list(judged["by"]) == ["taker", "goal", "taker", "goal", "taker", "taker"]
    assert list(judged["taker"]) == ["1", "1", "27", "GKH", "27", "GKA"]
    assert list(judged["verdict"]) == ["restart"] * 5 + ["keeper holds"]
    assert list(judged["first"]) == [a for a, _ in STILL] and list(judged["last"]) == [b - 1 for _, b in STILL]
    free_kick = judged.iloc[4]
    assert free_kick["taker_m"] == 0.5 and free_kick["opponent_m"] == 10.0 and free_kick["players_ms"] == 0.0
    assert (judged.loc[judged["by"] == "goal", "agrees"] == "yes").all()


def test_find_restarts_numbers_the_restarts():
    tracking, ball = made_up_half()
    restarts = find_restarts(tracking, ball, CONFIG)
    assert list(restarts["id"]) == [1, 2, 3, 4, 5] and list(restarts["kind"])[-1] == "free kick"
    assert "corner H (by the goal)" in restarts.iloc[1]["summary"]
    held = find_restarts(tracking, ball, CONFIG, "keeper holds")
    assert list(held["id"]) == [1] and "holds the ball" in held.iloc[0]["summary"]


def test_restart_kinds():
    kinds = {(1.0, 1.0): "corner", (50.0, -5.0): "off pitch", (102.0, 25.0): "goal kick",
             (0.5, 5.0): "goal line", (40.0, 49.0): "throw-in", (89.0, 25.0): "penalty",
             (50.5, 25.0): "kickoff", (30.0, 30.0): "free kick"}
    assert {xy: restart_kind(*xy, PITCH, SETTINGS) for xy in kinds} == kinds


def test_a_duel_at_the_touchline_is_in_play():
    tracking, ball = made_up_half()
    duel = at(tracking, slice(300, 380), 1, [60.5, 50.0])  # an H player right at A's "throw-in"
    throw_in = candidates(duel, ball, CONFIG).iloc[2]
    assert throw_in["verdict"] == "in play" and throw_in["why"] == "opponent at 0.7 m"
    assert throw_in["summary"].startswith("in play, no throw-in: opponent at 0.7 m")


def test_a_short_rest_is_in_play():
    tracking, ball = made_up_half()
    ball.loc[300:342, "speed_ms"] = 10.0  # still from 343 on: 1.5 s
    throw_in = candidates(tracking, ball, CONFIG).iloc[2]
    assert throw_in["verdict"] == "in play" and throw_in["why"] == "too short (1.5 s)"


def test_a_free_kick_needs_a_duel_before():
    """Before the free kick (frame 600) the ball rolls at the centre spot, where H1 stands: a duel.
    Looking back only 2 s with H1 gone from there, the opponents stayed away: build-up play."""
    tracking, ball = made_up_half()
    assert candidates(tracking, ball, CONFIG).iloc[4]["duel_m"] == 0.5
    away = at(tracking, slice(550, 600), 0, [30.0, 40.0])
    config = {**CONFIG, "restarts": {**SETTINGS, "before_s": 2.0}}
    held = candidates(away, ball, config).iloc[4]
    assert held["kind"] == "free kick" and held["verdict"] == "in play"
    assert held["why"] == "no duel before (closest opponent 10.0 m)"  # H1, 10 m off as the ball comes to rest
    unchecked = {**CONFIG, "restarts": {**SETTINGS, "before_s": 2.0, "duel_m": None}}
    assert candidates(away, ball, unchecked).iloc[4]["verdict"] == "restart"  # the check switched off


def test_running_players_are_no_stoppage():
    tracking, ball = made_up_half()
    players = tracking.players.copy()
    run = np.arange(80)[:, None] * 0.2 / PITCH[1]  # 5 m/s down the pitch ...
    for slot in (0, 1, 2, 3, 5):  # ... all but the thrower
        players[300:380, slot, 1] += run[:, 0]
    throw_in = candidates(replace(tracking, players=players), ball, CONFIG).iloc[2]
    assert throw_in["verdict"] == "in play" and throw_in["why"].startswith("players run (5.0 m/s")


def without_ball(ball, frames):
    ball = ball.copy()
    ball.loc[frames, ["x_m", "y_m", "speed_ms"]] = np.nan
    return ball


def test_a_throw_in_without_the_ball_is_found_from_the_thrower():
    tracking, ball = made_up_half()
    judged = candidates(tracking, without_ball(ball, slice(300, 379)), CONFIG)
    throw_in = judged.iloc[2]
    assert throw_in[["kind", "team", "by", "taker", "verdict"]].tolist() == ["throw-in", "A", "thrower", "27", "restart"]
    # his own speed over +-0.4 s: still from 0.4 s after he stops to 0.4 s before he moves
    assert throw_in["first"] == 310 and throw_in["last"] == 369 and np.isnan(throw_in["taker_m"])
    assert "by the thrower 27 behind the line, ball not seen" in throw_in["summary"]
    assert (judged["by"] == "thrower").sum() == 1  # with the ball seen, the still ball has it already
    assert (candidates(tracking, ball, CONFIG)["by"] != "thrower").all()


def test_standing_behind_the_line_while_play_goes_on():
    tracking, ball = made_up_half()
    away = at(tracking, slice(880, 960), 1, [40.0, -1.0])  # H2 behind the line, the ball 50 m away
    stays = at(tracking, slice(880, 1000), 1, [40.0, -1.0])  # ... and he stays there
    one = candidates(away, ball, CONFIG).iloc[-1]
    assert one["by"] == "thrower" and one["verdict"] == "in play"
    assert one["why"] == "ball in play elsewhere (28 m away)"
    two = candidates(stays, ball, CONFIG).iloc[-1]
    assert "not back on the pitch within 3 s" in two["why"]


def test_the_goal_decides_and_the_taker_is_checked():
    tracking, ball = made_up_half()
    near = at(tracking, slice(150, 230), 4, [99.8, 0.2])  # an A player nearer the corner flag than H1
    corner = candidates(near, ball, CONFIG).iloc[1]
    assert corner["team"] == "H" and corner["agrees"] == "no"
    assert corner["verdict"] == "in play" and "opponent at 0.3 m" in corner["why"]


def test_a_waiting_ball_has_no_team():
    tracking, ball = made_up_half()
    ball.loc[300:379, ["x_m", "y_m"]] = [60.0, 56.0]  # 6 m beyond the touchline: waiting
    waiting = candidates(tracking, ball, CONFIG).iloc[2]
    assert waiting["kind"] == "off pitch" and waiting["team"] == "" and waiting["verdict"] == "waiting"


def test_public_ground_truth_for_understanding(judged):
    gt = np.full(T, "", dtype=object)
    gt[150:230], gt[230:300] = "D", "H"  # the corner: dead while still, then H
    gt[300:380], gt[380:450] = "D", "H"  # the throw-in: dead, then H - not our A
    table = with_ground_truth(judged, gt, CONFIG)
    assert list(table["gt_after"])[1:3] == ["H", "H"] and table["gt_after"].iloc[0] == ""
    summary = ground_truth_table(table)
    assert summary.loc["restart: corner", "gt after = our team"] == "100%"
    assert summary.loc["restart: throw-in", "gt after = our team"] == "0%"
    assert summary.loc["restart: throw-in", "gt still = D"] == "100%"


def test_restarts_in_the_viewer():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.visualization import PitchPlot

    tracking, ball = made_up_half()
    fig, ax = plt.subplots()
    plot = PitchPlot(ax, tracking, CONFIG, episodes=find_restarts(tracking, ball, CONFIG), segment_name="restart")
    plot.draw(160)
    assert "restart 2/5" in plot.caption.get_text() and "corner H" in plot.caption.get_text()
    plt.close(fig)


def penalty_half(box_full=False, keeper_at_ball=False):
    """The made-up half with the ball still on the right penalty spot (89, 25) at 880-959."""
    tracking, ball = made_up_half()
    ball.loc[880:959, ["x_m", "y_m", "speed_ms"]] = [89.0, 25.0, 0.0]
    players = tracking.players.copy()
    players[880:960, 0] = np.array([88.0, 25.0]) / PITCH  # H1 the taker, the A keeper on his line (98, 25)
    if box_full:
        players[880:960, 1] = np.array([90.0, 32.0]) / PITCH  # H2 in the box ...
        players[880:960, 3] = np.array([92.0, 18.0]) / PITCH  # ... and A1
    if keeper_at_ball:
        players[880:960, 0] = np.array([70.0, 25.0]) / PITCH
        players[880:960, 5] = np.array([89.5, 25.0]) / PITCH  # the A keeper holds the ball
    return replace(tracking, players=players), ball


def test_a_penalty_needs_its_scene():
    tracking, ball = penalty_half()
    penalty = candidates(tracking, ball, CONFIG).iloc[-1]
    assert penalty["kind"] == "penalty" and penalty["team"] == "H" and penalty["verdict"] == "restart"
    crowded = candidates(*penalty_half(box_full=True), CONFIG).iloc[-1]
    assert crowded["kind"] == "penalty" and crowded["verdict"] == "in play"
    assert crowded["why"].startswith("no penalty scene (3 in the box besides the keepers")  # the taker counts too
    held = candidates(*penalty_half(box_full=True, keeper_at_ball=True), CONFIG).iloc[-1]
    assert held["kind"] == "keeper" and held["verdict"] == "keeper holds"
