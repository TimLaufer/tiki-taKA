"""Dead-ball indicators on a made-up half: 22 players spread out and running at 3 m/s; in
frames 100-199 (4 s) 14 of them stand crowded in the left penalty box, in 250-289 (1.6 s,
too short for an episode) they crowd again."""
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.data_loader import Tracking  # noqa: E402
from src.dead_ball import (  # noqa: E402
    ball_zone,
    crowding,
    episodes,
    gap_summary,
    ground_truth_share,
    indicators,
    long_gaps,
    plot_indicators,
    zone,
)
from src.positions import pack_filled  # noqa: E402

PITCH = [105.0, 68.0]
T = 400
CONFIG = {
    "fps": 25,
    "pitch": PITCH,
    "phases": [{"name": "first_half", "first": 0, "last": T - 1, "half": 1, "graded": True, "clock_start_min": 0}],
    "ball": {"max_gap_s": 5.0},
    "dead_ball": {"radius_m": 10.0, "speed_window_s": 0.4, "smooth_s": 1.0, "crowd_min_players": 12,
                  "min_duration_s": 3.0, "top": 25, "slow_speed_ms": 1.5, "line_margin_m": 1.5},
}
CROWDED, SHORT = slice(100, 200), slice(250, 290)


def make_half():
    grid = np.array([(x, y) for x in (40, 55, 70, 85, 100) for y in (8, 24, 40, 56)] + [(48, 32), (62, 32)], float)
    pos = np.repeat(grid[None], T, axis=0)  # (T, 22, 2), at least 14 m apart
    pos[:, :, 0] -= 0.12 * np.arange(T)[:, None] % 30  # everyone runs left at 3 m/s (0.12 m per frame)
    angles = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    huddle = np.c_[8 + 4 * np.cos(angles), 34 + 4 * np.sin(angles)]  # 14 players within 8 m, left box
    for part in (CROWDED, SHORT):
        pos[part, :14] = huddle  # standing still
    n = T
    return Tracking(
        ball=np.full((n, 2), np.nan, np.float32), ball_z=np.full(n, np.nan, np.float32),
        ball_speed=np.full(n, np.nan, np.float32), players=(pos / PITCH).astype(np.float32),
        speed=np.full((n, 22), np.nan, np.float32), speed_unit="", team=np.array(["H"] * 11 + ["A"] * 11),
        slot_ids=np.array([str(i + 1) for i in range(22)]), team_names=np.array(["Home", "Away"]),
        home_attacks_first_half="left_to_right",
    )


@pytest.fixture(scope="module")
def half():
    tracking = make_half()
    return tracking, indicators(tracking, CONFIG)


def test_pack_filled_moves_players_to_the_front():
    xy = np.full((2, 4, 2), np.nan)
    xy[0, 2] = [1, 1]
    xy[1, [1, 3]] = [[2, 2], [3, 3]]
    packed = pack_filled(xy)
    assert packed.shape == (2, 2, 2)
    np.testing.assert_array_equal(packed[1], [[2, 2], [3, 3]])
    assert np.isnan(packed[0, 1]).all()


def test_crowd_finds_the_densest_spot(half):
    tracking, ind = half
    assert ind.loc[50, "crowd"] == 1  # spread out: nobody within 10 m of anybody
    assert ind.loc[150, "crowd"] == 14
    assert ind.loc[150, "crowd_share"] == pytest.approx(14 / 22)
    assert np.hypot(ind.loc[150, "crowd_x"] - 8, ind.loc[150, "crowd_y"] - 34) <= 4.01  # in the huddle
    assert (ind["players"] == 22).all()
    assert crowding(tracking, PITCH, 1.0).loc[150, "crowd"] == 1  # a 1 m radius: the huddle is 1.7 m apart


def test_median_speed(half):
    _, ind = half
    assert ind.loc[50, "speed_ms"] == pytest.approx(3.0, abs=0.01)
    assert ind.loc[150, "speed_ms"] < 3.0  # 14 of 22 stand: the median is standing
    assert ind.loc[150, "speed_ms"] == pytest.approx(0.0, abs=0.01)


def test_one_episode_where_the_crowd_lasts(half):
    tracking, ind = half
    table = episodes(ind, tracking, CONFIG)
    assert len(table) == 1  # the 1.6 s huddle is too short
    assert list(table["id"]) == [1]
    episode = table.iloc[0]
    assert abs(episode["first"] - 100) <= 13 and abs(episode["last"] - 199) <= 13  # smoothing: 1 s
    assert episode["peak"] == 14 and episode["zone"] == "penalty box H"  # home defends the left in half 1
    assert episode["speed_ms"] == pytest.approx(0.0, abs=0.01)
    assert episode["match"] == "0:04"
    gt = np.full(T, "", dtype="<U1")
    gt[100:200] = "D"
    gt[200:] = "H"
    assert ground_truth_share(table, gt)[0] == f"{(gt[episode['first']:episode['last'] + 1] == 'D').mean():.0%}"


def test_ball_zone():
    margin = 1.5
    assert ball_zone(-3, 30, PITCH, margin) == "out of play"
    assert ball_zone(0.5, 67.6, PITCH, margin) == "corner"
    assert ball_zone(40, 0.5, PITCH, margin) == "touchline"
    assert ball_zone(104.5, 20, PITCH, margin) == "goal line"
    assert ball_zone(3, 34, PITCH, margin) == "goal area"
    assert ball_zone(52.5, 34.4, PITCH, margin) == "centre spot"
    assert ball_zone(94, 34, PITCH, margin) == "penalty spot"
    assert ball_zone(30, 20, PITCH, margin) == "in play"
    assert ball_zone(np.nan, 20, PITCH, margin) == "-"


def made_up_ball():
    """An analysed ball of 400 frames, missing in 100-249 (6 s: last seen out of play, seen again
    at the corner), in 270-369 (4 s: not longer than ball.max_gap_s) and from 380 to the end (0.8 s)."""
    n = 400
    x_m, y_m = np.full(n, 40.0), np.full(n, 30.0)
    source = np.full(n, "observed", dtype=object)
    source[100:250] = "missing"  # 6 s
    x_m[99], y_m[99] = -3.0, 30.0  # last seen out of play
    x_m[250], y_m[250] = 0.5, 0.5  # seen again at the corner
    source[270:370] = "missing"  # 4 s: shorter than ball.max_gap_s, not listed
    source[380:] = "missing"  # until the end of the half: too short as well (20 frames)
    import pandas as pd

    ball = pd.DataFrame({"phase": "first_half", "x_m": x_m, "y_m": y_m, "source": source},
                        index=pd.RangeIndex(n, name="frame"))
    return ball


def test_long_gaps_are_judged(half):
    import pandas as pd

    tracking, ind = half
    ball = made_up_ball()
    ind = ind.copy()
    ind["speed_ms"] = 3.0
    ind.loc[100:249, "speed_ms"] = 0.6  # the players walk in the first gap
    gaps = long_gaps(ball, ind, pd.DataFrame(columns=["id", "first", "last"]), CONFIG)
    assert list(gaps["id"]) == [1] and gaps.loc[0, "first"] == 100 and gaps.loc[0, "last"] == 249
    gap = gaps.iloc[0]
    assert gap["duration_s"] == 6.0 and gap["verdict"] == "dead"
    assert gap["ball_before"] == "out of play" and gap["ball_after"] == "corner"
    assert gap["reasons"] == "players walk (0.6 m/s); ball went out (out of play); restart at corner"

    # the same gap, but play goes on: nobody walks, the ball is lost and found in open play
    ball.loc[99, ["x_m", "y_m"]] = [40.0, 30.0]
    ball.loc[250, ["x_m", "y_m"]] = [45.0, 30.0]
    ind["speed_ms"] = 3.0
    gaps = long_gaps(ball, ind, episodes(ind, tracking, CONFIG), CONFIG)
    # frames 100-199 are the crowd episode of make_half: that alone says dead ball
    assert gaps.loc[0, "reasons"] == "players crowd (episode 1)"
    ind_spread = ind.copy()
    ind_spread["crowd"] = 2
    gaps = long_gaps(ball, ind_spread, pd.DataFrame(columns=["id", "first", "last"]), CONFIG)
    assert gaps.loc[0, "verdict"] == "unclear" and gaps.loc[0, "summary"] == "unclear: no sign of a stoppage"
    summary = gap_summary(gaps)
    assert summary.to_dict("records") == [{"half": "first_half", "verdict": "unclear", "gaps": 1, "seconds": 6.0}]


def test_zone_names():
    first, second = CONFIG["phases"][0], {"half": 2}
    assert zone(100, 34, first, "left_to_right", PITCH) == "penalty box A"  # A defends the right goal
    assert zone(100, 34, second, "left_to_right", PITCH) == "penalty box H"  # sides swapped
    assert zone(50, 2, first, "left_to_right", PITCH) == "touchline"
    assert zone(52.5, 34, first, "left_to_right", PITCH) == "centre circle"
    assert zone(30, 34, first, "unknown", PITCH) == "half left"
    assert zone(np.nan, 34, first, "left_to_right", PITCH) == "-"


def test_plot(half):
    tracking, ind = half
    table = episodes(ind, tracking, CONFIG)
    gt = np.full(T, "", dtype="<U1")
    gt[100:200] = "D"
    fig = plot_indicators(ind, table, CONFIG, gt)
    assert fig.axes[0].get_title(loc="left") == "first_half"
    matplotlib.pyplot.close(fig)
