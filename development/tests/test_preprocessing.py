from dataclasses import replace

import numpy as np
import pandas as pd

from src.data_loader import Tracking, load_config
from src.pipeline import MatchData
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.possession_model import PossessionModel
from src.postprocessing import postprocess
from test_restarts import made_up_half, CONFIG, PITCH

NO_TAGS = pd.DataFrame(columns=["first", "last", "type", "team"])


def example():
    tracking, ball = made_up_half()
    ball["x"], ball["y"] = ball["x_m"] / PITCH[0], ball["y_m"] / PITCH[1]
    ball["phase"], ball["source"] = "first_half", "observed"
    config = load_config()
    config.update({key: value for key, value in CONFIG.items() if key != "dead_ball"})
    # The made-up players never move, so every visible stretch between its placed-ball
    # restarts would be a walking interruption; bridging is tested in test_stoppages.
    config["stoppages"] = {**config["stoppages"], "bridge_s": 0.0}
    config["smart_tagging"] = {"enabled": False}
    data = MatchData(tracking=tracking, df=pd.DataFrame(), parts={}, dfs={}, ball=ball)
    return data, config


def test_restart_waiting_is_dead_and_taker_is_available_at_release():
    data, config = example()
    prepared = preprocess(data, config)
    assert prepared.dead[150:229].all()
    assert not prepared.dead[229]
    assert extract_features(prepared, config).loc[229, "restart_team"] == "H"
    assert not prepared.dead[750:830].any()  # goalkeeper holding in open play


def test_disappearance_after_still_ball_is_not_a_confirmed_restart():
    data, config = example()
    data.ball.loc[230:240, ["speed_ms", "x_m", "y_m", "x", "y"]] = np.nan
    data.ball.loc[230:240, "source"] = "missing"
    prepared = preprocess(data, config)
    assert 229 not in set(prepared.restarts["last"])


def test_sustained_outside_ball_is_dead_but_one_frame_glitch_is_not():
    data, config = example()
    data.ball["speed_ms"] = 10.
    data.ball.loc[25, ["x_m", "x"]] = [-2., -0.02]
    data.ball.loc[50:69, ["x_m", "x"]] = [-2., -0.02]
    dead = preprocess(data, config).dead
    assert not dead[25]
    assert dead[50:70].all()


N = 1000
BASES = np.array([[30, 20], [40, 30], [2, 25], [60, 20], [70, 30], [98, 25]], float)


def circling_match(standing=slice(100, 300), gaps=((150, 300), (500, 700))):
    """Six players circling their spots at 3 m/s, standing still in `standing`; the ball lies
    at (50, 25), far from everyone, and the tactical camera loses it in `gaps`."""
    running = np.ones(N)
    running[standing] = 0.0
    angle = np.cumsum(running * 1.0 / 25)  # 1 rad/s on a 3 m circle: 3 m/s
    pos = BASES[None] + 3.0 * np.stack([np.cos(angle), np.sin(angle)], axis=1)[:, None, :]
    tracking = Tracking(
        ball=np.full((N, 2), np.nan, np.float32), ball_z=np.full(N, np.nan, np.float32),
        ball_speed=np.full(N, np.nan, np.float32), players=(pos / PITCH).astype(np.float32),
        speed=np.full((N, 6), np.nan, np.float32), speed_unit="", team=np.array(list("HHHAAA")),
        slot_ids=np.array(["1", "2", "GKH", "26", "27", "GKA"]), team_names=np.array(["Home", "Away"]),
        home_attacks_first_half="left_to_right",
    )
    xy = np.tile([50.0, 25.0], (N, 1))
    for a, b in gaps:
        xy[a:b] = np.nan
    ball = pd.DataFrame({"phase": "first_half", "x": xy[:, 0] / PITCH[0], "y": xy[:, 1] / PITCH[1],
                         "x_m": xy[:, 0], "y_m": xy[:, 1],
                         "source": np.where(np.isfinite(xy[:, 0]), "observed", "missing"),
                         "speed_ms": np.where(np.isfinite(xy[:, 0]), 5.0, np.nan)},
                        index=pd.RangeIndex(N, name="frame"))
    config = load_config()
    config.update(fps=25, pitch=PITCH, phases=[{"name": "first_half", "first": 0, "last": N - 1, "half": 1,
                                                "graded": True, "clock_start_min": 0}])
    config["restarts"] = CONFIG["restarts"]
    return MatchData(tracking=tracking, df=pd.DataFrame(), parts={}, dfs={}, ball=ball), config


def test_a_long_gap_is_dead_when_the_players_walk_and_live_when_they_run():
    data, config = circling_match()
    prepared = preprocess(data, config, set_pieces=NO_TAGS)
    assert prepared.dead[150:300].all()  # standing still: the referee has stopped play
    assert not prepared.dead[500:700].any()  # running: play goes on unseen
    assert list(prepared.gaps["verdict"]) == ["dead", "live"]


def test_stoppages_are_judged_on_the_tactical_ball_not_the_broadcast_completed_one():
    data, config = circling_match()
    merged = data.ball.copy()
    merged.loc[200:260, "source"] = "observed"  # the broadcast saw the ball being placed
    merged.loc[200:260, ["x_m", "y_m"]] = [41.0, 30.0]
    merged.loc[200:260, ["x", "y"]] = [0.41, 0.6]
    data = replace(data, ball=merged, tactical_ball=data.ball)
    prepared = preprocess(data, config, set_pieces=NO_TAGS)
    assert prepared.dead[150:300].all()
    features = extract_features(prepared, config)
    labels = postprocess(PossessionModel(config).predict(features), config)
    assert set(labels[150:300]) == {"D"}  # a ball glimpsed during a stoppage is no possession


def test_prediction_end_to_end_has_one_label_per_frame_and_no_abstention_inside_possession():
    data, config = circling_match()
    dribble = data.tracking.players[400:500, 1].astype(float) * PITCH + [0.5, 0.0]  # at H's slot 2
    data.ball.loc[400:499, ["x_m", "y_m"]] = dribble
    data.ball.loc[400:499, ["x", "y"]] = dribble / PITCH
    prepared = preprocess(data, config, set_pieces=NO_TAGS)
    labels = postprocess(PossessionModel(config).predict(extract_features(prepared, config)), config)
    assert len(labels) == N and set(labels) <= set("HADX")
    assert set(labels[150:300]) == {"D"}
    assert set(labels[300:1000]) == {"H"}  # from the restart, and retained through the live occlusion
