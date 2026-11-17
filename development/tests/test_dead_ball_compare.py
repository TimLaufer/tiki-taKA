"""The dead-ball comparison on the made-up half of test_restarts: its restarts give the kinds."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.dead_ball_compare import STEFAN, TIM, TRUTH, compare, dead_balls, plot_half  # noqa: E402
from src.restarts import candidates  # noqa: E402
from tests.test_restarts import CONFIG as RESTART_CONFIG  # noqa: E402
from tests.test_restarts import made_up_half  # noqa: E402

CONFIG = {**RESTART_CONFIG, "segments": {"merge_s": 1.0}}
HALF = CONFIG["phases"][0]


def masks():
    stefan, tim, truth = (np.zeros(1000, dtype=bool) for _ in range(3))
    stefan[140:230] = True  # the corner (played at 229)
    tim[120:230] = tim[300:380] = True  # the corner, earlier, and the throw-in (379)
    truth[130:230] = True
    tim[880:950] = True  # nothing played after it, the ball seen at the centre spot: kickoff
    return {STEFAN: stefan, TIM: tim, TRUTH: truth}


def test_kinds_from_the_restarts_and_the_ball():
    tracking, ball = made_up_half()
    still = candidates(tracking, ball, CONFIG)
    restarts = still[still["verdict"] == "restart"]
    assert dead_balls(masks()[TIM], restarts, ball, HALF, CONFIG) == [
        (120, 230, "corner"), (300, 380, "throw-in"), (880, 950, "kickoff")]


def test_compare_and_plot(tmp_path):
    table = compare(masks(), CONFIG)
    only_tim = table[table["source"] == "only Tim"].iloc[0]
    assert only_tim["dead balls"] == 3  # 120-139, the throw-in, 880-949
    tracking, ball = made_up_half()
    still = candidates(tracking, ball, CONFIG)
    restarts = still[still["verdict"] == "restart"]
    bars = {name: dead_balls(mask, restarts, ball, HALF, CONFIG) for name, mask in masks().items() if name != TRUTH}
    bars[TRUTH] = [(130, 230, "D")]
    fig = plot_half(bars, HALF, CONFIG)
    assert len(fig.axes) == 6  # three sources, frame and minute
    fig.savefig(tmp_path / "dead.png", dpi=80, bbox_inches="tight")
    plt.close(fig)


def test_viewer_list_and_kinds_per_frame():
    from src.dead_ball_compare import dead_ball_table, kinds_per_frame
    from src.visualization import PitchPlot

    bars = {STEFAN: [(140, 230, "corner")], TIM: [(120, 230, "corner"), (300, 380, "throw-in")]}
    table = dead_ball_table(bars, CONFIG)
    assert list(table["first"]) == [120, 300] and list(table["last"]) == [229, 379]
    assert table.loc[0, "summary"] == "gap-based: corner 3.6 s | restart-based: corner 4.4 s"
    assert table.loc[1, "summary"] == "gap-based: live | restart-based: throw-in 3.2 s"
    tracking, _ = made_up_half()
    kinds = {"Stefan": kinds_per_frame(bars[STEFAN], 1000), "Tim": kinds_per_frame(bars[TIM], 1000)}
    fig, ax = plt.subplots()
    plot = PitchPlot(ax, tracking, CONFIG, episodes=table, segment_name="dead ball", dead_kinds=kinds)
    plot.draw(130)
    assert "Stefan: live   Tim: corner" in plot.caption.get_text()
    assert "dead ball 1/2" in plot.caption.get_text()
    plt.close(fig)


def test_both_recognitions_judged():
    from src.dead_ball_compare import combine

    stefan = [(100, 200, "goal kick"), (300, 360, "other"), (500, 560, "kickoff"), (700, 760, "throw-in")]
    tim = [(110, 190, "penalty"), (290, 380, "throw-in"), (520, 560, "kickoff"), (900, 950, "other")]
    assert combine(stefan, tim, 1000) == [
        (100, 200, "goal kick", "Stefan"),  # Stefan's goal kick beats Tim's "penalty"
        (290, 380, "throw-in", "Tim"),  # Tim's throw-in, his span
        (500, 560, "kickoff", "both"),  # the same kind: the union
        (700, 760, "throw-in", "Stefan"),  # only Stefan sees it
        (900, 950, "other", "Tim"),  # only Tim
    ]


def test_live_play_starts_with_the_winning_side():
    import pandas as pd

    from src.dead_ball_compare import live_after

    spans = [(100, 200, "goal kick"), (290, 380, "throw-in")]
    tim_events = pd.DataFrame({"played": [379], "after_team": ["A"]})
    stefan_seeds = pd.DataFrame({"start": [200], "team": ["H"]})
    live = live_after(spans, tim_events, stefan_seeds, CONFIG)
    assert list(live["first"]) == [0, 200, 380] and list(live["last"]) == [99, 289, 999]
    assert list(live["team"]) == ["", "H", "A"]
