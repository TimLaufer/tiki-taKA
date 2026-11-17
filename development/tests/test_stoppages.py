"""Stoppages on made-up halves of a 100 m x 50 m pitch, 25 fps. H defends the left goal, A
the right one (their centres at the kickoff). Players stand still:

    slot  team  where
    1     H     (30, 20)
    2     H     (40, 30)
    GKH   H     (2, 25)    in front of the left goal
    26    A     (60, 20)
    27    A     (70, 30)
    GKA   A     (98, 25)   in front of the right goal
"""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import Tracking, load_config
from src.stoppages import (
    DEAD,
    KEEPER,
    LIVE,
    bridge_interruptions,
    control_team,
    judge_gaps,
    live_stretches,
    restart_seeds,
    set_piece_at,
    stoppage_mask,
)

PITCH = [100.0, 50.0]
PLAYERS = np.array([[30, 20], [40, 30], [2, 25], [60, 20], [70, 30], [98, 25]], float)


def make_config(n, phases=None):
    config = load_config()
    config.update(fps=25, pitch=PITCH)
    config["phases"] = phases or [{"name": "first_half", "first": 0, "last": n - 1, "half": 1, "graded": True,
                                   "clock_start_min": 0}]
    return config


def make_tracking(n, players=PLAYERS):
    pos = np.broadcast_to(players, (n, len(players), 2)).astype(float)
    return Tracking(
        ball=np.full((n, 2), np.nan, np.float32), ball_z=np.full(n, np.nan, np.float32),
        ball_speed=np.full(n, np.nan, np.float32), players=(pos / PITCH).astype(np.float32),
        speed=np.full((n, len(players)), np.nan, np.float32), speed_unit="", team=np.array(list("HHHAAA")),
        slot_ids=np.array(["1", "2", "GKH", "26", "27", "GKA"]), team_names=np.array(["Home", "Away"]),
        home_attacks_first_half="left_to_right",
    )


def make_ball(n, seen=None):
    """The analysed-ball columns the stoppage code reads; seen: [(frame or (first, stop)), (x_m, y_m))]
    (a list: slices are dict keys only from Python 3.12 on)."""
    xy = np.full((n, 2), np.nan)
    for frames, spot in (seen or []):
        frames = slice(*frames) if isinstance(frames, tuple) else frames
        xy[frames] = spot
    source = np.where(np.isfinite(xy[:, 0]), "observed", "missing")
    return pd.DataFrame({"x": xy[:, 0] / PITCH[0], "y": xy[:, 1] / PITCH[1], "x_m": xy[:, 0], "y_m": xy[:, 1],
                         "source": source, "speed_ms": np.where(np.isfinite(xy[:, 0]), 0.0, np.nan)},
                        index=pd.RangeIndex(n, name="frame"))


def gap(first=100, last=299, speed=1.0, before="in play", after="in play", crowded=False):
    return {"first": first, "last": last, "duration_s": (last - first + 1) / 25, "speed_ms": speed,
            "crowded": crowded, "ball_before": before, "ball_after": after, "verdict": "dead",
            "reasons": "", "summary": ""}


def judge(*rows, set_pieces=None, ball=None, n=1000):
    config = make_config(n)
    return judge_gaps(pd.DataFrame(list(rows)), make_tracking(n), make_ball(n) if ball is None else ball,
                      set_pieces, config)


NO_TAGS = pd.DataFrame(columns=["first", "last", "type", "team"])


def test_walking_players_make_a_long_gap_dead_and_running_ones_keep_it_live():
    judged = judge(gap(speed=1.2), gap(speed=3.0), gap(speed=np.nan), set_pieces=NO_TAGS)
    assert list(judged["verdict"]) == [DEAD, LIVE, LIVE]  # nobody tracked: no evidence of a stoppage
    assert "players walk" in judged["why"][0]


def test_ball_last_seen_out_of_play_is_dead_even_with_running_players():
    assert judge(gap(speed=4.0, before="out of play"), set_pieces=NO_TAGS)["verdict"][0] == DEAD


@pytest.mark.parametrize("cue", [dict(before="touchline"), dict(after="corner"), dict(crowded=True)])
def test_slow_players_need_one_more_cue(cue):
    assert judge(gap(speed=1.8), set_pieces=NO_TAGS)["verdict"][0] == LIVE
    judged = judge(gap(speed=1.8, **cue), set_pieces=NO_TAGS)
    assert judged["verdict"][0] == DEAD and judged["cues"][0]


def test_smart_tagging_corroborates_a_slow_gap_but_never_decides_alone():
    # The tag interval must contain the restart (the first frame after the gap), +- 1 s.
    tags = pd.DataFrame({"first": [310], "last": [600], "type": ["Throw-in"], "team": [""]})
    judged = judge(gap(speed=1.8), gap(speed=3.0), gap(first=500, last=700, speed=1.8), set_pieces=tags)
    assert list(judged["verdict"]) == [DEAD, LIVE, LIVE]
    assert judged["cues"][0] == "tagged: Throw-in"
    assert judged["cues"][2] == ""  # its restart, frame 701, lies outside the tag
    assert set_piece_at(285, tags, 25) == "Throw-in" and set_piece_at(284, tags, 25) == ""


def test_the_same_goalkeeper_at_both_ends_holds_the_ball():
    n = 1000
    ball = make_ball(n, [(99, (97.5, 25.0)), (300, (97.0, 25.5))])  # at GKA before and after
    judged = judge(gap(speed=1.0), ball=ball, set_pieces=NO_TAGS)
    assert judged["verdict"][0] == KEEPER and judged["team"][0] == "A"
    # last seen at a line: the ball went out, the keeper then restarts - a stoppage
    assert judge(gap(speed=1.0, before="goal line"), ball=ball, set_pieces=NO_TAGS)["verdict"][0] == DEAD
    # reappearing at an outfield player: no hold
    moved = make_ball(n, [(99, (97.5, 25.0)), (300, (70.5, 30.0))])
    assert judge(gap(speed=1.0), ball=moved, set_pieces=NO_TAGS)["verdict"][0] == DEAD


def test_control_needs_reach_and_a_margin_over_every_opponent():
    config = make_config(10)
    tracking = make_tracking(10)
    assert control_team(tracking, 0, 31.0, 20.0, config) == ("H", 0)
    assert control_team(tracking, 0, 35.0, 20.0, config) == ("", -1)  # 5 m away: out of reach
    duel = make_tracking(10, np.vstack([PLAYERS[:3], [[31.6, 20.0]], PLAYERS[4:]]))
    assert control_team(duel, 0, 31.0, 20.0, config) == ("", -1)  # opponent 0.6 m: contested
    assert control_team(tracking, 0, np.nan, 20.0, config) == ("", -1)


def test_mask_unions_dead_gaps_outside_ball_and_waiting_before_a_restart():
    n = 1000
    config = make_config(n)
    judged = pd.DataFrame([{**gap(100, 299), "verdict": DEAD}, {**gap(400, 499), "verdict": LIVE},
                           {**gap(600, 700), "verdict": KEEPER}])
    ball = make_ball(n, [((800, 820), (-3.0, 20.0)), (850, (-3.0, 20.0))])  # 0.8 s outside; a 1-frame glitch
    restarts = pd.DataFrame({"first": [900], "last": [960], "team": ["H"]})
    speed = np.full(n, 3.0)
    dead = stoppage_mask(n, judged, ball, restarts, speed, config)
    assert dead[100:300].all() and not dead[400:500].any() and not dead[600:701].any()
    assert dead[800:820].all() and not dead[850]
    assert dead[900:960].all() and not dead[960]  # the release frame is live
    assert dead.sum() == 200 + 20 + 60


def test_graded_frames_before_a_kickoff_are_dead_but_ungraded_ones_are_not():
    phases = [{"name": "pre_match", "first": 0, "last": 49, "half": 0, "graded": False},
              {"name": "first_half", "first": 50, "last": 149, "half": 1, "graded": True, "clock_start_min": 0},
              {"name": "halftime", "first": 150, "last": 199, "half": 0, "graded": False},
              {"name": "second_half_pre_kickoff", "first": 200, "last": 249, "half": 2, "graded": True},
              {"name": "second_half", "first": 250, "last": 399, "half": 2, "graded": True, "clock_start_min": 45}]
    config = make_config(400, phases)
    empty = pd.DataFrame([gap()]).iloc[:0]
    dead = stoppage_mask(400, empty, make_ball(400), pd.DataFrame(columns=["first", "last"]),
                         np.full(400, 3.0), config)
    assert dead[200:250].all() and dead.sum() == 50  # Law 8: in play from the kickoff only


def test_a_short_walking_interruption_between_stoppages_is_no_restart():
    config = make_config(1000)
    config["stoppages"]["bridge_s"] = 5.0  # optional rule, off by default
    half = np.ones(1000, dtype=np.int8)
    dead = np.zeros(1000, dtype=bool)
    dead[100:300] = dead[350:500] = True  # 2 s visible between two stoppages
    walking, running = np.full(1000, 1.0), np.full(1000, 3.0)
    assert bridge_interruptions(dead, walking, half, config)[300:350].all()
    assert not bridge_interruptions(dead, running, half, config)[300:350].any()  # play restarted
    long_gap = dead.copy()
    long_gap[350:500] = False
    long_gap[500:600] = True  # 8 s visible: a restart, however slow
    assert not bridge_interruptions(long_gap, walking, half, config)[300:500].any()
    edge = np.zeros(1000, dtype=bool)
    edge[30:300] = True  # the 30 frames before are the start of the half, not an interruption
    assert not bridge_interruptions(edge, walking, half, config)[:30].any()


def test_bridging_is_off_by_default_and_with_zero():
    config = make_config(1000)
    assert config["stoppages"]["bridge_s"] == 0.0
    dead = np.zeros(1000, dtype=bool)
    dead[100:300] = dead[350:500] = True
    assert not bridge_interruptions(dead, np.full(1000, 1.0), np.ones(1000, dtype=np.int8), config)[300:350].any()


def test_bridging_never_joins_two_halves():
    phases = [{"name": "first_half", "first": 0, "last": 99, "half": 1, "graded": True, "clock_start_min": 0},
              {"name": "second_half", "first": 100, "last": 199, "half": 2, "graded": True, "clock_start_min": 45}]
    config = make_config(200, phases)
    config["stoppages"]["bridge_s"] = 5.0
    half = np.r_[np.ones(100), np.full(100, 2)].astype(np.int8)
    dead = np.zeros(200, dtype=bool)
    dead[50:95] = dead[105:150] = True
    out = bridge_interruptions(dead, np.full(200, 1.0), half, config)
    assert not out[95:105].any()
    assert live_stretches(out, config) == [(0, 50), (95, 100), (100, 105), (150, 200)]


def seeds_after(spot, players=PLAYERS, restarts=None, n=200):
    """The restart seed of the live stretch starting at frame 100, the ball seen at `spot`."""
    config = make_config(n)
    dead = np.zeros(n, dtype=bool)
    dead[40:100] = True
    ball = make_ball(n, [] if spot is None else [((100, 110), spot)])
    restarts = pd.DataFrame(columns=["first", "last", "team"]) if restarts is None else restarts
    table = restart_seeds(dead, ball, make_tracking(n, players), restarts, config)
    return table.set_index("start").loc[100, ["team", "how"]].tolist()


def test_corner_goes_to_the_attackers_and_goal_kick_to_the_defenders_by_the_laws():
    defender_at_flag = np.vstack([PLAYERS[:3], [[98.8, 1.2]], PLAYERS[4:]])
    assert seeds_after((99.5, 0.5), defender_at_flag) == ["H", "corner"]  # A defends the right goal
    attacker_in_area = np.vstack([PLAYERS[:3], [[3.5, 25.0]], PLAYERS[4:]])
    assert seeds_after((3.0, 25.0), attacker_in_area) == ["H", "goal kick"]  # H defends the left goal


def test_elsewhere_the_taker_restarts_and_a_ball_in_flight_decides_nothing():
    thrower = np.vstack([PLAYERS[:4], [[60.0, 51.2]], PLAYERS[5:]])
    assert seeds_after((60.0, 50.4), thrower) == ["A", "taker"]
    assert seeds_after((50.0, 10.0)) == ["", "none"]  # nobody within reach
    assert seeds_after(None) == ["", "none"]  # nothing seen


def test_a_zero_seed_window_switches_restart_teams_off():
    config = make_config(200)
    config["stoppages"]["seed_window_s"] = 0.0
    dead = np.zeros(200, dtype=bool)
    dead[40:100] = True
    thrower = np.vstack([PLAYERS[:4], [[60.0, 51.2]], PLAYERS[5:]])
    table = restart_seeds(dead, make_ball(200, [((100, 110), (60.0, 50.4))]), make_tracking(200, thrower),
                          pd.DataFrame(columns=["first", "last", "team"]), config)
    assert set(table["how"]) == {"none"} and set(table["team"]) == {""}


def test_a_placed_ball_restart_takes_precedence():
    restarts = pd.DataFrame({"first": [40], "last": [100], "team": ["A"]})
    assert seeds_after((99.5, 0.5), restarts=restarts) == ["A", "placed-ball restart"]
