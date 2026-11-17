"""The match cut at the dead balls, on the made-up half of test_restarts. Restarts: kickoff H,
corner H, throw-in A (300-379), goal kick H, free kick A (600-679); the keeper holding the ball
(750-829) is none. Between them the ball rolls at the centre spot, 0.5 m from H1: H controls it.
Here also two long gaps judged dead - one overlapping the throw-in, one alone (880-950) - and
one unclear gap, which is no event. The players there stand all the time, so "players slow"
is switched off (slow_speed_ms < 0) and play stops where the ball comes to rest; where play
stops is tested on a scene of its own, with running players."""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import Tracking
from src.restarts import candidates
from src.segments import (
    dead_ball_events,
    ground_truth_check,
    live_stretches,
    possession_after,
    stoppage_starts,
)
from tests.test_restarts import CONFIG as RESTART_CONFIG
from tests.test_restarts import made_up_half

SEGMENTS = {"merge_s": 1.0, "after_s": 3.0, "control_m": 2.0, "before_s": 3.0, "catch_m": 2.0, "catch_look_s": 0.24, "catch_pass_min_ms": 5.0, "catch_back_m": 60.0, "max_lookback_s": 60.0,
            "out_margin_m": 1.5, "out_min_s": 0.4, "rejoin_s": 3.0, "max_out_gap_s": 15.0, "slow_speed_ms": 1.5, "reaction_s": 1.0,
            "smooth_s": 1.0}
CONFIG = {**RESTART_CONFIG, "segments": {**SEGMENTS, "slow_speed_ms": -1.0}}
GAPS = pd.DataFrame({"half": "first_half", "first": [370, 880, 960], "last": [400, 950, 990],
                     "verdict": ["dead", "dead", "unclear"]})


@pytest.fixture(scope="module")
def cut():
    tracking, ball = made_up_half()
    ball.loc[380:449, ["x_m", "y_m"]] = [60.5, 20.0]  # after the throw-in, the ball goes to A1
    events = dead_ball_events(candidates(tracking, ball, CONFIG), GAPS, CONFIG)
    events = possession_after(events, tracking, ball, CONFIG)
    return events, live_stretches(events, CONFIG)


def test_events(cut):
    events, _ = cut
    assert list(events["kind"]) == ["kickoff", "corner", "throw-in + gap", "goal kick", "free kick", "gap"]
    assert list(events["first"]) == [0, 150, 300, 450, 600, 880]
    assert list(events["last"]) == [79, 229, 400, 529, 679, 950]
    assert list(events["played"]) == [79, 229, 379, 529, 679, 950]  # the throw-in is played at 379
    assert list(events["clear"]) == [True] * 5 + [False]
    assert list(events["team"]) == ["H", "H", "A", "H", "A", ""]
    assert list(events["start"]) == list(events["first"]) and set(events["start_by"]) == {"-"}


def test_possession_after(cut):
    events, _ = cut
    assert list(events["after_team"]) == ["H", "H", "A", "H", "H", "H"]
    assert list(events["agrees"]) == ["yes", "yes", "yes", "yes", "no", "-"]
    assert events.loc[0, "after_share"] == 1.0 and events.loc[0, "control_frames"] == 75
    assert events.loc[4, "after_share"] == 0.93  # 70 frames H1 at the ball, then 5 the A keeper
    # 70 frames A1 at the ball, then 5 the ball at the goal kick, at GKH
    assert events.loc[2, "summary"] == ("throw-in A [throw-in + gap]; dead 3.2 s; before: H (100%); "
                                        "after: A (93% of 75 frames)")
    assert events.loc[4, "summary"].endswith("not the restart's team")
    assert events.loc[5, "summary"].startswith("not clear (no restart at its end) [gap]; dead 2.8 s")


def test_possession_before(cut):
    events, _ = cut
    assert list(events["before_team"]) == ["", "H", "H", "A", "H", "H"]  # nothing before the kickoff
    assert list(events["last_touch"]) == ["", "H", "H", "A", "H", "H"]
    # H plays it out, A throws in: fine; the goal kick H after A had it: fine
    assert list(events["out_check"]) == ["-", "odd", "ok", "ok", "-", "-"]
    assert "but H played it out" in events.loc[1, "summary"]  # the made-up corner: H had the ball


def test_no_one_at_the_ball_after():
    tracking, ball = made_up_half()
    ball.loc[680:754, ["x_m", "y_m"]] = [50.0, 45.0]  # after the free kick nobody near the ball
    events = possession_after(dead_ball_events(candidates(tracking, ball, CONFIG), GAPS[:0], CONFIG),
                              tracking, ball, CONFIG)
    free_kick = events[events["kind"] == "free kick"].iloc[0]
    assert free_kick["after_team"] == "A" and free_kick["control_frames"] == 0  # the restart's team
    assert "after: A (the restart's team, no one within 2 m)" in free_kick["summary"]


def test_live_stretches(cut):
    _, live = cut
    assert list(live["first"]) == [80, 230, 380, 530, 680, 951]
    assert list(live["last"]) == [149, 299, 449, 599, 879, 999]
    assert list(live["team"]) == ["H", "H", "A", "H", "H", "H"]
    assert list(live["end_team"]) == ["H", "H", "A", "H", "H", ""]


def running_scene():
    """1200 frames, 4 players running circles at 5 m/s on a 100 x 50 m pitch, the ball 1 m off
    one of them. Three events: a throw-in resting from 300 (the ball out 250-299, back in at
    270-272), a free kick resting from 600 (the players stop at 560), a free kick resting from
    1000 (the ball out at 700-710 only, and the players run on)."""
    t = np.arange(1200)[:, None]
    phase = np.array([0.0, 1.5, 3.0, 4.5])[None, :]
    pos = np.stack([40 + 10 * np.cos(0.02 * t + phase), 25 + 10 * np.sin(0.02 * t + phase)], axis=-1)
    pos[260:380] = pos[260]  # everybody stops for the throw-in ...
    pos[560:680] = pos[560]  # ... and for the first free kick
    tracking = Tracking(
        ball=np.full((1200, 2), np.nan, np.float32), ball_z=np.full(1200, np.nan, np.float32),
        ball_speed=np.full(1200, np.nan, np.float32), players=(pos / [100.0, 50.0]).astype(np.float32),
        speed=np.full((1200, 4), np.nan, np.float32), speed_unit="", team=np.array(list("HHAA")),
        slot_ids=np.array(["1", "2", "3", "4"]), team_names=np.array(["Home", "Away"]),
        home_attacks_first_half="left_to_right",
    )
    xy = pos[:, 0] + [1.0, 0.0]
    xy[250:300], xy[270:273] = [30.0, -2.0], [30.0, 1.0]
    xy[700:711] = [50.0, 53.0]
    ball = pd.DataFrame({"x_m": xy[:, 0], "y_m": xy[:, 1]}, index=pd.RangeIndex(1200, name="frame"))
    events = pd.DataFrame({"half": "first_half", "first": [300, 600, 1000], "last": [379, 679, 1079],
                           "played": [379, 679, 1079]})
    config = {**RESTART_CONFIG, "phases": [{**RESTART_CONFIG["phases"][0], "last": 1199}],
              "segments": {**SEGMENTS, "max_out_gap_s": 5.0}}
    return tracking, ball, events, config


def test_where_play_stopped():
    tracking, ball, events, config = running_scene()
    starts = stoppage_starts(events, tracking, ball, config)
    assert list(starts["start_by"]) == ["ball out", "players slow", "-"]
    assert starts.loc[0, "start"] == 250  # the first frame out, the short return at 270 bridged
    assert 535 <= starts.loc[1, "start"] <= 560  # the players stop at 560, 1 s for the whistle
    assert starts.loc[2, "start"] == 1000  # out 290 frames before the rest: too long ago, no part of it
    assert list(starts["dead_s"]) == [round((379 - 250 + 1) / 25, 1), round((679 - starts.loc[1, "start"] + 1) / 25, 1),
                                      3.2]


def test_public_ground_truth_for_understanding(cut):
    events, _ = cut
    gt = np.full(1000, "", dtype=object)
    gt[150:230], gt[230:300] = "D", "H"  # the corner: dead, then H as ours
    gt[300:401], gt[380:455] = "D", "H"  # the throw-in: dead, then H - ours says A
    gt[379] = "D"
    check = ground_truth_check(events, gt, CONFIG)
    # after the kickoff D (the corner's stoppage begins), after the corner H = ours, after the throw-in H, not A
    assert check["its label after = our after_team"] == "33% of 3"
    assert check["D in the dead balls (start to played)"] == "97% of 165"  # the goal kick from 450: 5 frames H
    assert check["D in the part before the ball rests"] == "-"


def test_events_in_the_viewer(cut):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.visualization import PitchPlot

    events, _ = cut
    tracking, _ = made_up_half()
    fig, ax = plt.subplots()
    plot = PitchPlot(ax, tracking, CONFIG, episodes=events, segment_name="event")
    plot.draw(320)
    assert "event 3/6" in plot.caption.get_text() and "after: A" in plot.caption.get_text()
    plt.close(fig)


def test_a_ball_vanishing_at_the_keeper_is_a_catch():
    from src.segments import keeper_catches

    tracking, ball = made_up_half()
    ball.loc[840:899, ["x_m", "y_m"]] = np.nan  # gone at 840 ...
    ball.loc[839, ["x_m", "y_m"]] = [97.0, 25.5]  # ... from the A keeper (98, 25)
    gaps = pd.DataFrame({"half": "first_half", "first": [840, 900], "last": [899, 950], "verdict": ["dead", "dead"],
                         "ball_before": ["in play", "in play"]})
    catches = keeper_catches(gaps, tracking, ball, CONFIG)
    assert catches[["first", "last", "team", "keeper"]].values.tolist() == [[840, 899, "A", "GKA"]]
    out = gaps.assign(ball_before=["touchline", "in play"])  # last seen out of play: no catch
    assert keeper_catches(out, tracking, ball, CONFIG).empty
    events = dead_ball_events(candidates(tracking, ball, CONFIG), gaps[:1], CONFIG, catches)
    assert 840 not in list(events["first"])  # the catch is no dead ball


def test_a_ball_turning_up_in_flight_from_the_keeper_is_a_catch():
    """Gone at 840 in midfield, seen again at 900 flying away from A's goal (right): run back
    along its flight it comes out of A's box - the A keeper caught it unseen and threw it out.
    Where the keeper is tracked plays no part (out of the picture it is often wrong)."""
    from src.segments import keeper_catches

    tracking, ball = made_up_half()
    ball.loc[840:899, ["x_m", "y_m"]] = np.nan
    ball.loc[839, ["x_m", "y_m"]] = [60.0, 30.0]  # nobody's: H1 is 10 m off
    for i, f in enumerate(range(900, 910)):
        ball.loc[f, ["x_m", "y_m"]] = [80.0 - 0.6 * i, 25.0 + 0.1 * i]  # 15 m/s, from (98, 25) towards the left
    gaps = pd.DataFrame({"half": "first_half", "first": [840], "last": [899], "verdict": ["dead"],
                         "ball_before": ["in play"]})
    catches = keeper_catches(gaps, tracking, ball, CONFIG)
    assert catches[["team", "keeper", "how"]].values.tolist() == [["A", "GKA", "played out of his box"]]
    rolling = ball.copy()
    rolling.loc[900:909, ["x_m", "y_m"]] = [80.0, 25.0]  # turns up lying still: no flight to run back
    assert keeper_catches(gaps, tracking, rolling, CONFIG).empty
    sideways = ball.copy()
    for i, f in enumerate(range(900, 910)):
        sideways.loc[f, ["x_m", "y_m"]] = [60.0, 10.0 + 0.6 * i]  # flying across the pitch: not from him
    assert keeper_catches(gaps, tracking, sideways, CONFIG).empty


def test_the_keepers_tracked_position_plays_no_part():
    from dataclasses import replace

    from src.segments import keeper_catches

    tracking, ball = made_up_half()
    players = tracking.players.copy()
    players[:, 5] = np.array([60.0, 45.0]) / [100.0, 50.0]  # the A keeper tracked far off his goal
    ball.loc[840:899, ["x_m", "y_m"]] = np.nan
    for i, f in enumerate(range(900, 910)):
        ball.loc[f, ["x_m", "y_m"]] = [80.0 - 0.6 * i, 25.0 + 0.1 * i]
    gaps = pd.DataFrame({"half": "first_half", "first": [840], "last": [899], "verdict": ["dead"],
                         "ball_before": ["in play"]})
    assert keeper_catches(gaps, replace(tracking, players=players), ball, CONFIG)["team"].tolist() == ["A"]


def test_a_glimpse_beyond_the_line_is_no_ball_out():
    tracking, ball, events, config = running_scene()
    ball.loc[250:299, ["x_m", "y_m"]] = [30.0, 1.0]  # the ball stays in ...
    ball.loc[280:284, ["x_m", "y_m"]] = [30.0, -2.0]  # ... but for 0.2 s it is tracked 2 m out
    starts = stoppage_starts(events, tracking, ball, config)
    assert starts.loc[0, "start_by"] != "ball out"
