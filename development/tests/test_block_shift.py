"""Block shifts on a made-up half of 400 frames: team A keeps its formation and shifts across
the pitch at 3 m/s in frames 100-299, the ball with it; two A tracks swap their players at
frame 200. Team H runs about, each player on his own."""
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.block_shift import (  # noqa: E402
    _direction,
    block_episodes,
    block_indicators,
    ground_truth_crosstab,
    ground_truth_majority,
    plot_blocks,
)
from src.data_loader import Tracking  # noqa: E402

PITCH = [105.0, 68.0]
T = 400
MOVING = slice(100, 300)
CONFIG = {
    "fps": 25,
    "pitch": PITCH,
    "phases": [{"name": "first_half", "first": 0, "last": T - 1, "half": 1, "graded": True, "clock_start_min": 0}],
    "ball": {"max_gap_s": 5.0},
    "block": {"window_s": 2.0, "min_shift_m": 3.0, "min_rigidity": 0.7, "min_players": 7, "min_ball_move_m": 3.0,
              "smooth_s": 1.0, "min_duration_s": 2.0},
}


def make_half():
    rng = np.random.default_rng(0)
    pos = np.empty((T, 22, 2))
    # H: 11 players running about on their own
    pos[0, :11] = rng.uniform([10, 10], [95, 58], (11, 2))
    velocity = np.zeros((11, 2))
    for t in range(1, T):
        velocity = 0.9 * velocity + rng.normal(0, 0.05, (11, 2))
        pos[t, :11] = np.clip(pos[t - 1, :11] + velocity, [2, 2], [103, 66])
    # A: a formation that shifts across (+y) at 0.12 m per frame in MOVING
    formation = np.array([(x, y) for x in (55, 68, 81) for y in (6, 16, 26)] + [(92, 16), (40, 16)], float)
    lateral = np.clip(np.arange(T) - MOVING.start, 0, MOVING.stop - MOVING.start) * 0.12
    pos[:, 11:] = formation[None] + np.c_[np.zeros(T), lateral][:, None, :]
    pos[200:, [11, 12]] = pos[200:, [12, 11]]  # two tracks swap their players
    ball = np.c_[np.full(T, 50.0), 20 + lateral]
    tracking = Tracking(
        ball=(ball / PITCH).astype(np.float32), ball_z=np.full(T, np.nan, np.float32),
        ball_speed=np.full(T, np.nan, np.float32), players=(pos / PITCH).astype(np.float32),
        speed=np.full((T, 22), np.nan, np.float32), speed_unit="", team=np.array(["H"] * 11 + ["A"] * 11),
        slot_ids=np.array([str(i + 1) for i in range(22)]), team_names=np.array(["Home", "Away"]),
        home_attacks_first_half="left_to_right",
    )
    analysed_ball = pd.DataFrame({"x_m": ball[:, 0], "y_m": ball[:, 1]}, index=pd.RangeIndex(T, name="frame"))
    return tracking, analysed_ball


@pytest.fixture(scope="module")
def half():
    tracking, ball = make_half()
    return tracking, block_indicators(tracking, ball, CONFIG)


def test_the_shifting_team_is_a_block(half):
    _, ind = half
    middle = ind.loc[150:250]  # the window lies inside MOVING; frame 200 has the swapped tracks
    assert middle["A_block"].all()
    np.testing.assert_allclose(middle["A_shift_y"], 6.0, atol=1e-4)  # 0.12 m x 50 frames
    np.testing.assert_allclose(middle["A_shift_x"], 0.0, atol=1e-4)
    assert (middle["A_rigidity"] > 0.99).all()  # the two swapped tracks are outvoted
    np.testing.assert_allclose(middle["A_follows_ball"], 1.0, atol=1e-4)
    assert not ind.loc[30:60, "A_block"].any()  # standing still: no shift
    assert ind["H_block"].mean() < 0.05  # everyone on his own: no block


def test_episode_and_hint(half):
    tracking, ind = half
    table = block_episodes(ind, tracking, CONFIG)
    a = table[table["team"] == "A"]
    assert len(a) == 1 and len(table) == len(a)
    episode = a.iloc[0]
    assert abs(episode["first"] - 100) <= 15 and abs(episode["last"] - 299) <= 15
    assert episode["direction"] == "across" and episode["hint"] == "follows the ball: H has it"
    assert episode["summary"].startswith("A shifts as a block across, 6.0 m per 2 s")
    assert list(table["id"]) == [1]


def test_direction_is_relative_to_the_attack():
    first, second = {"half": 1}, {"half": 2}
    assert _direction(5, 1, "H", first, "left_to_right") == "forward"  # H attacks to the right in half 1
    assert _direction(5, 1, "A", first, "left_to_right") == "back"
    assert _direction(5, 1, "H", second, "left_to_right") == "back"  # sides swapped
    assert _direction(1, 5, "H", first, "left_to_right") == "across"
    assert _direction(np.nan, 5, "H", first, "left_to_right") == "-"


def test_ground_truth_comparison(half):
    tracking, ind = half
    table = block_episodes(ind, tracking, CONFIG)
    gt = np.full(T, "", dtype="<U1")
    gt[50:350] = "H"
    gt[280:300] = "D"
    assert ground_truth_majority(table, gt)[0].startswith("H ")
    cross = ground_truth_crosstab(ind, gt)
    assert cross.loc["A in block shift", "gt H"].endswith("%") and int(cross.loc["A in block shift", "frames"]) > 100


def test_plot_and_viewer_arrows(half):
    from src.visualization import PitchPlot

    tracking, ind = half
    table = block_episodes(ind, tracking, CONFIG)
    gt = np.full(T, "H", dtype="<U1")
    fig = plot_blocks(ind, table, CONFIG, gt)
    assert len(fig.axes) == 2 and fig.axes[0].get_title(loc="left") == "first_half | shift (m per 2 s)"
    matplotlib.pyplot.close(fig)

    fig, ax = matplotlib.pyplot.subplots()
    plot = PitchPlot(ax, tracking, CONFIG, episodes=table, segment_name="block", blocks=ind)
    plot.draw(200)
    assert plot.arrows["A"].get_visible() and plot.arrows["A"].get_alpha() == 1.0
    assert "block 1/1 (" in plot.caption.get_text() and "H has it" in plot.caption.get_text()
    matplotlib.pyplot.close(fig)
