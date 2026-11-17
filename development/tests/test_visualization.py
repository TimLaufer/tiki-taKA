import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src.data_loader import load_tracking  # noqa: E402
from src.visualization import PitchPlot, read_labels, save  # noqa: E402


def test_draws_frames_with_and_without_ball(epts_config, tmp_path):
    tracking = load_tracking(epts_config, use_cache=False)
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("frame,label\n0,h\n1,A\n2,D\n3,X\n4,X\n")
    labels = {"sub": read_labels(labels_csv, tracking.n_frames)}

    fig, ax = plt.subplots()
    plot = PitchPlot(ax, tracking, epts_config, labels)
    plot.draw(2)
    assert "BALL MISSING" in plot.caption.get_text()
    plot.draw(0)
    assert "sub: H" in plot.caption.get_text()
    assert len(plot.dots["H"].get_offsets()) == 1

    save(fig, plot, tmp_path / "frame.png", 0, None, 1)
    save(fig, plot, tmp_path / "clip.gif", 0, 4, 1)
    assert (tmp_path / "frame.png").stat().st_size > 0
    assert (tmp_path / "clip.gif").stat().st_size > 0
    plt.close(fig)


def test_episodes_in_the_viewer(monkeypatch):
    """The made-up half of test_dead_ball: crowded in frames 100-199 and 250-289; with a
    minimum of 1 s both are episodes, id 1 the longer one."""
    from matplotlib.backend_bases import KeyEvent

    from src.dead_ball import episodes, indicators
    from src.visualization import run_viewer
    from tests.test_dead_ball import CONFIG, make_half

    config = {**CONFIG, "dead_ball": {**CONFIG["dead_ball"], "min_duration_s": 1.0}}
    tracking = make_half()
    crowd = indicators(tracking, config)
    table = episodes(crowd, tracking, config)
    assert list(table["id"]) == [1, 2] and table.loc[0, "first"] < table.loc[1, "first"]

    fig, ax = plt.subplots()
    plot = PitchPlot(ax, tracking, config, crowd=crowd, episodes=table)
    plot.draw(150)
    assert "episode 1/2 (4.0 s from 0:04): penalty box H, up to 14 players" in plot.caption.get_text()
    assert plot.circle.get_visible()
    assert plot.circle.get_edgecolor() == matplotlib.colors.to_rgba("#ef4444")  # red inside an episode
    assert plot.circle_label.get_text() == "14 within 10 m"
    plot.draw(50)
    assert "no episode here" in plot.caption.get_text()
    assert plot.circle.get_edgecolor() == matplotlib.colors.to_rgba("#9ca3af")  # grey elsewhere

    monkeypatch.setattr(plt, "show", lambda: None)
    starts = list(table["first"])
    run_viewer(fig, plot, starts[0], starts)

    def press(key):
        fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, key))

    def shown():
        return int(plot.caption.get_text().split()[1])

    press("n")
    assert shown() == starts[1]  # episode 2
    press("n")
    assert shown() == starts[1]  # there is no episode 3: stays
    press("p")
    assert shown() == starts[0]  # back to episode 1
    plt.close(fig)


def test_gaps_in_the_viewer():
    """The long gaps come with their own summary line."""
    import pandas as pd

    from tests.test_dead_ball import CONFIG, make_half

    tracking = make_half()
    gaps = pd.DataFrame({"id": [1], "first": [100], "last": [299], "duration_s": [8.0], "match": ["0:04"],
                         "summary": ["dead: players walk (0.6 m/s); restart at corner"]})
    fig, ax = plt.subplots()
    plot = PitchPlot(ax, tracking, CONFIG, episodes=gaps, segment_name="gap")
    plot.draw(150)
    assert "gap 1/1 (8.0 s from 0:04): dead: players walk (0.6 m/s); restart at corner" in plot.caption.get_text()
    plot.draw(50)
    assert "no gap here  -  n / p: next / previous gap" in plot.caption.get_text()
    plt.close(fig)


def test_our_label_against_the_truth(epts_config):
    import numpy as np

    tracking = load_tracking(epts_config, use_cache=False)
    ours = np.array(list("HHDAA"))
    truth = np.array(["H", "A", "D", "", ""])
    fig, ax = plt.subplots()
    plot = PitchPlot(ax, tracking, epts_config, {"ours": ours, "truth": truth})
    plot.draw(0)
    assert "differs" not in plot.caption.get_text() and plot.badge.get_bbox_patch().get_edgecolor()[:3] != (1, 0, 0)
    assert plot.badge.get_text() == " ours H\ntruth H"
    plot.draw(1)
    assert "ours: H   truth: A   <- differs" in plot.caption.get_text()
    edge = plot.badge.get_bbox_patch().get_edgecolor()
    assert edge[0] > 0.8 and edge[1] < 0.3  # red
    plot.draw(3)
    assert plot.badge.get_text() == " ours A\ntruth -" and "differs" not in plot.caption.get_text()
    plt.close(fig)
