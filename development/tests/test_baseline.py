"""The first labelling on a made-up live stretch: H has the ball after a restart, A wins it."""
import numpy as np
import pandas as pd

from src.baseline import label_frames, score
from tests.test_restarts import CONFIG as RESTART_CONFIG
from tests.test_restarts import made_up_half

CONFIG = {**RESTART_CONFIG, "baseline": {"control_m": 1.0, "switch_s": 2.0}}


def test_dead_balls_and_a_change_of_possession():
    tracking, ball = made_up_half()
    ball.loc[:, ["x_m", "y_m"]] = [50.0, 25.0]  # at H1 (0.5 m) ...
    ball.loc[300:339, ["x_m", "y_m"]] = [60.5, 20.0]  # ... A1 for 1.6 s: no change ...
    ball.loc[400:499, ["x_m", "y_m"]] = [60.5, 20.0]  # ... A1 for 4 s: a change, from 400
    ball.loc[500:899, ["x_m", "y_m"]] = [50.0, 45.0]  # nobody near: A keeps it
    events = pd.DataFrame({"first": [0], "last": [79]})
    live = pd.DataFrame({"first": [80], "last": [899], "team": ["H"]})
    labels = label_frames(1000, events, live, tracking, ball, CONFIG)
    assert set(labels[:80]) == {"D"} and set(labels[900:]) == {"D"}
    assert set(labels[80:400]) == {"H"} and set(labels[400:900]) == {"A"}


def test_a_half_starts_with_the_first_team_in_control():
    tracking, ball = made_up_half()
    ball.loc[:, ["x_m", "y_m"]] = [60.5, 20.0]  # A1 has it from the start
    live = pd.DataFrame({"first": [0], "last": [999], "team": [""]})
    labels = label_frames(1000, pd.DataFrame(), live, tracking, ball, CONFIG)
    assert set(labels) == {"A"}


def test_score():
    gt = np.array(list("HHHHAAAADD") + [""] * 2, dtype=object)
    ours = np.array(list("HHHAAAAADDHH"))
    summary, table = score(ours, gt)
    assert summary["graded frames"] == 10 and summary["accuracy"] == "90.0%"
    assert summary["F1 D"] == "1.000" and table.loc["H", "A"] == 1


def test_our_dead_balls_with_stefans_possession_model():
    """Our cut (D from start to played, the restart team at the start of live play), Stefan's
    control rule inside: within 3 m and 1 m nearer than every opponent, a change after 0.6 s."""
    from src.baseline import stefan_labels

    tracking, ball = made_up_half()
    ball.loc[:, ["x_m", "y_m"]] = [50.0, 25.0]  # at H1
    ball.loc[400:419, ["x_m", "y_m"]] = [60.5, 20.0]  # A1 for 0.8 s: a change ...
    ball.loc[420:999, ["x_m", "y_m"]] = [50.0, 45.0]  # ... and nobody near after: A keeps it
    ball["source"], ball["speed_ms"] = "observed", 5.0
    config = {**CONFIG, "model": {"control_radius_m": 3.0, "control_margin_m": 1.0, "control_max_speed_ms": 12.0,
                                  "confirmation_s": 0.6}}
    events = pd.DataFrame({"start": [0], "played": [79]})
    live = pd.DataFrame({"first": [80], "last": [999], "team": ["H"]})
    labels = stefan_labels(1000, events, live, tracking, ball, config)
    assert set(labels[:80]) == {"D"}
    assert set(labels[80:400]) == {"H"} and set(labels[400:1000]) == {"A"}


def test_a_catch_is_play_for_the_keepers_team():
    from src.baseline import stefan_labels

    tracking, ball = made_up_half()
    ball.loc[:, ["x_m", "y_m"]] = [50.0, 25.0]  # H1 has it ...
    ball.loc[500:699, ["x_m", "y_m"]] = np.nan  # ... then the A keeper catches it, unseen for 8 s
    ball["source"] = np.where(ball["x_m"].isna(), "missing", "observed")
    ball["speed_ms"] = 5.0
    config = {**CONFIG, "model": {"control_radius_m": 3.0, "control_margin_m": 1.0, "control_max_speed_ms": 12.0,
                                  "confirmation_s": 0.6}}
    events = pd.DataFrame({"start": [0], "played": [79]})
    events.attrs["catches"] = pd.DataFrame({"first": [500], "last": [699], "team": ["A"]})
    live = pd.DataFrame({"first": [80], "last": [999], "team": ["H"]})
    labels = stefan_labels(1000, events, live, tracking, ball, config)
    assert set(labels[500:700]) == {"A"}  # no D: the keeper's team has the ball


def test_restart_based_replaces_only_the_dead_balls_in_stefans_features():
    from src.baseline import restart_based

    features = pd.DataFrame({"nearest_H_m": 1.0, "nearest_A_m": 5.0, "dead": False, "restart_team": "",
                             "hold_team": ""}, index=range(1000))
    features.loc[500:520, "dead"] = True  # Stefan's stoppage, ours has none there
    events = pd.DataFrame({"start": [100], "played": [179]})
    events.attrs["catches"] = pd.DataFrame({"first": [600], "last": [650], "team": ["A"]})
    live = pd.DataFrame({"first": [0, 180], "last": [99, 999], "team": ["", "H"]})
    out = restart_based(features, events, live, CONFIG)
    assert out["dead"].to_numpy()[100:180].all() and not out["dead"].to_numpy()[500:521].any()
    assert out.loc[180, "restart_team"] == "H" and out.loc[0, "restart_team"] == ""
    assert set(out.loc[600:650, "hold_team"]) == {"A"}
    assert (out["nearest_H_m"] == features["nearest_H_m"]).all()  # his evidence untouched


def test_per_stoppage_the_logic_with_the_richer_data():
    from src.baseline import choose_by_evidence

    features = pd.DataFrame({"dead": False, "restart_team": "", "hold_team": ""}, index=range(1000))
    features.loc[120:179, "dead"] = True  # Stefan: from where the tracker loses the ball
    features.loc[180, "restart_team"] = "A"
    features.loc[400:459, "dead"] = True  # Stefan alone: ball unseen
    features.loc[460, "restart_team"] = "H"
    events = pd.DataFrame({"start": [100, 600, 800], "played": [179, 659, 859],
                           "start_by": ["ball out", "-", "players slow"], "clear": [True, True, False],
                           "after_team": ["H", "A", "A"]})
    out = choose_by_evidence(features, events)
    dead = out["dead"].to_numpy()
    assert dead[100:180].all() and out.loc[180, "restart_team"] == "H"  # ball seen going out: Tim
    assert dead[400:460].all() and out.loc[460, "restart_team"] == "H"  # unseen: Stefan
    assert dead[600:660].all() and out.loc[660, "restart_team"] == "A"  # a placed ball Stefan missed
    assert not dead[800:860].any()  # only Tim, from walking pace alone: live
