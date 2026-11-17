"""Team shape (src.team_shape) on a made-up half of a 100 m x 50 m pitch, 25 fps: H defends the
left goal and attacks to the right, spread on a 15 m circle and advancing at 1 m/s; A defends the
right goal, compact on a 5 m circle and retreating towards it at 2 m/s."""
import numpy as np
import pytest

from src.data_loader import Tracking, load_config
from src.team_shape import apply, fit, home_logit, refine, shape_differences

PITCH = [100.0, 50.0]
N = 200


def made_up_half():
    t = np.arange(N) / 25
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    home = np.c_[40 + 15 * np.cos(angles), 25 + 15 * np.sin(angles) * 0.5]
    away = np.c_[60 + 5 * np.cos(angles), 25 + 5 * np.sin(angles)]
    pos = np.zeros((N, 10, 2))
    pos[:, :4] = home[None] + np.c_[1.0 * t, 0 * t][:, None, :]
    pos[:, 4] = [2.0, 25.0]                                   # GKH: not an outfield player
    pos[:, 5:9] = away[None] + np.c_[2.0 * t, 0 * t][:, None, :]
    pos[:, 9] = [98.0, 25.0]                                  # GKA
    config = load_config()
    config.update(fps=25, pitch=PITCH, phases=[{"name": "first_half", "first": 0, "last": N - 1, "half": 1,
                                                "graded": True, "clock_start_min": 0}])
    tracking = Tracking(
        ball=np.full((N, 2), np.nan, np.float32), ball_z=np.full(N, np.nan, np.float32),
        ball_speed=np.full(N, np.nan, np.float32), players=(pos / PITCH).astype(np.float32),
        speed=np.full((N, 10), np.nan, np.float32), speed_unit="", team=np.array(list("HHHHHAAAAA")),
        slot_ids=np.array(["1", "2", "3", "4", "GKH", "26", "27", "28", "29", "GKA"]),
        team_names=np.array(["Home", "Away"]), home_attacks_first_half="left_to_right")
    return tracking, config


def test_differences_measure_spread_direction_and_pace_from_the_home_side():
    tracking, config = made_up_half()
    spread, advance, pace = np.nanmedian(shape_differences(tracking, config)[20:-20], axis=0)
    expected_home_spread = np.mean(np.hypot(15 * np.cos(np.linspace(0, 2 * np.pi, 4, endpoint=False)),
                                            7.5 * np.sin(np.linspace(0, 2 * np.pi, 4, endpoint=False))))
    assert spread == pytest.approx(expected_home_spread - 5.0, abs=0.01)  # keepers left out
    assert advance == pytest.approx(1.0 - (-2.0), abs=0.01)  # A retreats towards its own goal
    assert pace == pytest.approx(1.0 - 2.0, abs=0.01)


def test_fit_learns_the_direction_of_a_separating_difference_and_needs_enough_frames():
    rng = np.random.default_rng(0)
    labels = np.array(["H"] * 3000 + ["A"] * 3000)
    differences = np.c_[np.r_[rng.normal(3, 1, 3000), rng.normal(-3, 1, 3000)], rng.normal(0, 1, (6000, 2))]
    model = fit(differences, labels, labels.copy())
    assert model["weights"][1] > 2 and abs(model["weights"][2]) < 0.5
    logit = home_logit(differences, model)
    assert (logit[:3000] > 0).mean() > 0.95 and (logit[3000:] < 0).mean() > 0.95
    assert fit(differences[:1500], labels[:1500], labels[:1500].copy()) is None  # one team only
    no_evidence = np.full(6000, "", dtype="<U1")
    assert fit(differences, labels, no_evidence) is None  # pseudo-labels need control evidence


def test_veto_returns_a_long_possession_the_teams_never_reacted_to():
    config = {"fps": 25, "team_shape": {"lag_s": 3.0, "settle_s": 2.0, "veto_below": 0.2}}
    labels = np.array(["A"] * 300 + ["H"] * 300)
    graded = np.ones(600, dtype=bool)
    away_shape, home_shape = np.full(600, -3.0), np.full(600, 3.0)
    out, counts = apply(labels, away_shape, graded, config)
    assert set(out) == {"A"} and counts["vetoed"] == 1
    out, counts = apply(labels, home_shape, graded, config)
    assert np.array_equal(out, labels) and counts["vetoed"] == 0
    short = np.array(["A"] * 300 + ["H"] * 100 + ["A"] * 200)  # 4 s: too short to judge
    assert np.array_equal(apply(short, away_shape, graded, config)[0], short)
    after_stoppage = np.array(["D"] * 300 + ["H"] * 300)  # a restart is never vetoed
    assert np.array_equal(apply(after_stoppage, away_shape, graded, config)[0], after_stoppage)


def test_fill_gives_unattributed_graded_frames_to_the_clear_shape_only():
    config = {"fps": 25, "team_shape": {"lag_s": 3.0, "settle_s": 2.0, "veto_below": 0.2}}
    labels = np.array(["D"] * 10 + ["X"] * 50 + ["D"] * 10 + ["X"] * 30)
    graded = np.r_[np.ones(70, dtype=bool), np.zeros(30, dtype=bool)]
    out, counts = apply(labels, np.full(100, 3.0), graded, config)
    assert set(out[10:60]) == {"H"} and set(out[70:]) == {"X"} and counts["filled"] == 1
    assert set(apply(labels, np.full(100, 0.5), graded, config)[0][10:60]) == {"X"}  # not clear enough


def test_refine_without_enough_pseudo_labels_changes_nothing():
    tracking, config = made_up_half()
    labels = np.array(["H"] * N)
    out, info = refine(labels, tracking, labels.copy(), np.ones(N, dtype=bool), config)
    assert np.array_equal(out, labels) and info == {"vetoed": 0, "filled": 0, "fitted": False}
