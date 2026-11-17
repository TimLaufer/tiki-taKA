from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.feature_extraction import extract_features
from src.pipeline import load_match
from src.possession_model import PossessionModel
from src.postprocessing import postprocess
from src.preprocessing import preprocess


def config_for(n, window=1):
    return {"fps": 1, "model": {"smoothing_half_window_s": window},
            "phases": [{"name": "first_half", "first": 0, "last": n - 1,
                        "half": 1, "graded": True}]}


def control_features(sequence):
    distances = {"H": (1., 5.), "A": (5., 1.), "C": (1., 1.),
                 "L": (8., 9.), "X": (np.inf, np.inf), "D": (1., 5.)}
    pairs = np.array([distances[s] for s in sequence])
    return pd.DataFrame({"nearest_H_m": pairs[:, 0], "nearest_A_m": pairs[:, 1],
                         "dead": [s == "D" for s in sequence], "graded": True, "half": 1,
                         "ball_source": ["missing" if s == "X" else "observed" for s in sequence],
                         "ball_visible": [s != "X" for s in sequence], "ball_speed_ms": 2.,
                         "restart_team": ""})


def decode(sequence, **settings):
    return "".join(PossessionModel({"fps": 10, "model": settings}).predict(control_features(sequence)))


def test_missing_evidence_is_not_automatically_dead():
    assert decode("XXXX") == "XXXX"


def test_brief_touch_contested_and_loose_balls_retain_control():
    assert decode("HHHHAACCLLLLHHHH") == "H" * 16


def test_sustained_opponent_control_changes_at_first_evidence():
    assert decode("HHHHAAAA") == "HHHHAAAA"


def test_no_control_from_fast_pass_or_interpolated_ball():
    for column, value in [("ball_speed_ms", 20.), ("ball_source", "interpolated")]:
        features = control_features("HHHHAAAA")
        features.loc[4:, column] = value
        assert "".join(PossessionModel({"fps": 10}).predict(features)) == "H" * 8


def test_stoppage_resets_owner_and_restart_seeds_new_owner():
    features = control_features("HHHHDDCCCLLL")
    features.loc[6, "restart_team"] = "A"
    assert "".join(PossessionModel({"fps": 10}).predict(features)) == "HHHHDDAAAAAA"


def test_unknown_after_stoppage_does_not_inherit_old_team():
    assert decode("HHHHDXXX") == "HHHHDXXX"


def test_occlusion_of_any_length_keeps_the_owner_and_halves_reset():
    assert decode("HHHH" + "X" * 300) == "H" * 304  # possession persists until a change is confirmed
    features = control_features("HHHHCCCC")
    features.loc[4:, "half"] = 2
    assert "".join(PossessionModel({"fps": 10}).predict(features)) == "HHHHXXXX"


def test_first_confirmed_control_owns_a_stretch_without_known_restart():
    assert decode("DDLLLHHHHLL") == "DD" + "H" * 9
    assert decode("DDLLAAH") == "DDXXXXX"  # never confirmed: abstain, the scorer repeats D


def test_restart_team_keeps_the_ball_until_the_opponent_confirms_control():
    features = control_features("DDLLAAAAL")
    features.loc[2, "restart_team"] = "H"
    assert "".join(PossessionModel({"fps": 10}).predict(features)) == "DDHHAAAAA"
    features.loc[2, "restart_team"] = "A"
    assert "".join(PossessionModel({"fps": 10}).predict(features)) == "DDAAAAAAA"


def test_goalkeeper_holding_the_unseen_ball_gives_his_team_possession_from_his_catch():
    features = control_features("AAAAHHXXXXLL")
    features["hold_team"] = ""
    features.loc[6:9, "hold_team"] = "H"  # his two frames of evidence fall short of confirmation
    assert "".join(PossessionModel({"fps": 10}).predict(features)) == "AAAA" + "H" * 8
    fresh = control_features("DDXXXXLL")
    fresh["hold_team"] = ""
    fresh.loc[2:5, "hold_team"] = "A"  # no restart team known: the keeper's team from the start
    assert "".join(PossessionModel({"fps": 10}).predict(fresh)) == "DD" + "A" * 6


def test_model_rejects_labels_for_training():
    with pytest.raises(ValueError, match="does not train"):
        PossessionModel({"fps": 10}).fit(control_features("HH"), labels=["H", "H"])


def test_nan_distance_is_unavailable_and_remote_player_is_not_control():
    features = control_features("AAAA")
    features["nearest_H_m"] = np.nan
    assert "".join(PossessionModel({"fps": 10}).predict(features)) == "AAAA"
    assert decode("LLLL") == "XXXX"


def test_smoothing_preserves_stoppage_and_does_not_cross_it():
    labels = np.array(list("HHXDXAA"))
    assert "".join(postprocess(labels, config_for(7, 50))) == "HHHDAAA"
    assert "".join(labels) == "HHXDXAA"  # does not mutate caller


def test_no_evidence_in_a_run_stays_unknown():
    assert "".join(postprocess(list("XXDHH"), config_for(5, 50))) == "XXDHH"


def test_smoothing_does_not_cross_adjacent_halves():
    config = config_for(4, 50)
    config["phases"] = [
        {"name": "first_half", "first": 0, "last": 1, "half": 1, "graded": True},
        {"name": "second_half", "first": 2, "last": 3, "half": 2, "graded": True},
    ]
    assert "".join(postprocess(list("HXAA"), config)) == "HHAA"


@pytest.mark.parametrize("window", [-1, float("nan"), float("inf")])
def test_invalid_window_rejected(window):
    with pytest.raises(ValueError):
        postprocess(["H"], config_for(1, window))


def test_end_to_end_without_ground_truth(epts_config):
    epts_config["public_gt"] = "/does/not/exist.csv"
    match = load_match(epts_config, use_video_2=False)
    with pytest.warns(UserWarning, match="Smart Tagging not found"):  # optional corroboration only
        prepared = preprocess(match, epts_config)
    features = extract_features(prepared, epts_config)
    predictions = postprocess(PossessionModel(epts_config).predict(features), epts_config)
    assert len(predictions) == 5
    assert predictions[3] == "X"  # halftime
    assert predictions[4] == "X"  # absent observations cannot borrow from half one
    assert np.isin(predictions, list("HADX")).all()
    with pytest.raises(ValueError, match="timeline"):
        extract_features(replace(prepared, ball=prepared.ball.set_axis(range(1, 6))), epts_config)


def test_distance_uses_pitch_metres_and_both_coordinates(epts_config):
    epts_config["smart_tagging"] = {"enabled": False}
    prepared = preprocess(load_match(epts_config, use_video_2=False), epts_config)
    players = prepared.tracking.players.copy()
    players[:] = np.nan
    players[0, 0] = [0.6, 0.5]  # H: 10.53 m
    players[0, 1] = [0.5, 0.64]  # A: 9.52 m (despite larger normalized distance)
    ball = prepared.ball.copy()
    ball.loc[0, ["x", "y"]] = [0.5, 0.5]
    prepared = replace(prepared, tracking=replace(prepared.tracking, players=players), ball=ball)
    features = extract_features(prepared, epts_config)
    assert features.loc[0, "nearest_H_m"] == pytest.approx(10.53, abs=1e-4)
    assert features.loc[0, "nearest_A_m"] == pytest.approx(9.52, abs=1e-4)
    assert PossessionModel(epts_config).predict(features)[0] == "X"  # both players too far to control
