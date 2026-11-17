import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.ball_analysis import (  # noqa: E402
    analyze_ball,
    ball_dataframe,
    gap_length,
    gaps_to_fill,
    insert_halves,
    merge_ball,
    frames_without_ball,
    gap_stats,
    gap_table,
    phase_table,
    plot_frames_without_ball,
    speed_check,
    summary,
)

# 100 m x 50 m pitch: x 0.01 = 1 m. Gaps up to 5 frames (0.2 s at 25 fps) are filled.
CONFIG = {
    "fps": 25,
    "pitch": [100.0, 50.0],
    "ball": {"max_gap_s": 0.2, "max_speed_ms": 40.0, "velocity_window": 2},
    "phases": [
        {"name": "first_half", "first": 0, "last": 19, "half": 1, "graded": True},
        {"name": "halftime", "first": 20, "last": 29, "half": 0, "graded": False},
    ],
}


def ball_df(x, y=None, phase=None, speed=None):
    x = np.asarray(x, dtype=float)
    df = pd.DataFrame({"ball_x": x, "ball_y": np.full(len(x), 0.5) if y is None else y})
    df.index.name = "frame"
    if phase is not None:
        df.insert(0, "phase", phase)
    if speed is not None:
        df["ball_s"] = speed
    return df


def rolling_ball(n=20):
    """Ball rolling along x at 1 m per frame = 25 m/s."""
    return 0.2 + 0.01 * np.arange(n)


def test_short_gap_is_filled_on_the_straight_line():
    x = rolling_ball()
    x[5:8] = np.nan  # 3 frames
    ball = analyze_ball(ball_df(x), CONFIG)
    np.testing.assert_allclose(ball["x"], rolling_ball())
    assert list(ball["source"][4:9]) == ["observed"] + ["interpolated"] * 3 + ["observed"]
    assert list(ball["gap_frames"][4:9]) == [0, 3, 3, 3, 0]


def test_long_gap_and_gaps_at_the_edges_stay_missing():
    x = rolling_ball()
    x[:2] = np.nan  # start of the data
    x[6:12] = np.nan  # 6 frames > 5
    x[-1] = np.nan  # end of the data
    ball = analyze_ball(ball_df(x), CONFIG)
    missing = ball["source"] == "missing"
    assert list(np.flatnonzero(missing)) == [0, 1, 6, 7, 8, 9, 10, 11, 19]
    assert ball["x"][missing].isna().all()
    assert ball["gap_frames"][6] == 6 and ball["gap_frames"][0] == 2


def test_gap_across_a_phase_boundary_is_not_bridged():
    x = np.full(30, 0.5)
    x[18:22] = np.nan  # 4 frames, short enough, but frames 18-19 and 20-21 are in different phases
    phase = ["first_half"] * 20 + ["halftime"] * 10
    ball = analyze_ball(ball_df(x, phase=phase), CONFIG)
    assert (ball["source"][18:22] == "missing").all()
    assert list(ball["gap_frames"][18:22]) == [2, 2, 2, 2]


def test_single_false_detection_is_dropped_and_filled():
    x = np.full(20, 0.5)
    x[10] = 0.9  # 40 m away for one frame
    ball = analyze_ball(ball_df(x), CONFIG)
    assert ball["outlier"][10] and ball["outlier"].sum() == 1
    assert ball["source"][10] == "interpolated" and ball["x"][10] == pytest.approx(0.5)


def test_fast_but_steady_ball_is_kept():
    x = 0.1 + 0.014 * np.arange(20)  # 1.4 m per frame = 35 m/s, a hard shot
    ball = analyze_ball(ball_df(x), CONFIG)
    assert not ball["outlier"].any()


def test_velocity_in_metres_per_second():
    ball = analyze_ball(ball_df(rolling_ball()), CONFIG)
    np.testing.assert_allclose(ball["vx"][2:-2], 25.0)
    np.testing.assert_allclose(ball["vy"][2:-2], 0.0, atol=1e-9)
    assert ball["vx"][:2].isna().all()  # needs 2 frames on each side
    np.testing.assert_allclose(ball["x_m"], rolling_ball() * 100)


def test_speed_check_tells_the_unit():
    df = ball_df(rolling_ball(), speed=np.full(20, 90.0))  # 25 m/s = 90 km/h
    df.attrs["speed_unit"] = "kph"
    text = speed_check(analyze_ball(ball_dataframe(df), CONFIG))
    assert "median 3.60" in text and "'kph'" in text


def half_with_gaps():
    """Like a half: the index are video frames from 100. Unseen: frames 100, 105-109 and 112."""
    x = np.full(13, 0.5)
    x[[0, 5, 6, 7, 8, 9, 12]] = np.nan
    df = ball_df(x)
    df.index = pd.RangeIndex(100, 113, name="frame")
    return df


def test_frames_without_ball_counts_up_and_drops_to_zero():
    counter = frames_without_ball(half_with_gaps())
    assert list(counter.index) == list(range(100, 113))
    # seen in 104, unseen in 105-109, seen again in 110: 1..5, then 0
    assert list(counter) == [1, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 1]


def test_gap_stats_name_the_longest_gap():
    stats = gap_stats(half_with_gaps(), CONFIG)
    assert stats["gaps"] == 3 and stats["frames_without_ball"] == 7
    assert stats["longest_gap_frames"] == 5 and stats["longest_gap_s"] == 0.2
    assert stats["longest_gap_at_frames"] == "105-109"
    assert stats["gaps_longer_than_0.2_s"] == 0  # 5 frames = 0.2 s is still filled
    assert stats["ball_seen"] == f"{6 / 13:.1%}"


def test_plot_frames_without_ball():
    ax = plot_frames_without_ball(half_with_gaps(), CONFIG, title="first_half")
    assert ax.get_title(loc="right") == "longest gap: 5 frames = 0.2 s  |  from frames 105-109, tactical video 0:00:04.20"
    assert ax.get_title(loc="left") == "first_half"
    assert ax.get_xlabel() == "video frame"
    matplotlib.pyplot.close(ax.figure)


def real_first_half():
    """Made-up ball data on the real first-half frames (kickoff 11185), with a 10 s gap
    starting at match minute 47:00 = 45+2:00, in stoppage time."""
    from src.data_loader import load_config

    config = load_config()
    frames = np.arange(11185, 90652)
    x = np.full(len(frames), 0.5)
    start = 11185 + 25 * 47 * 60
    x[start - 11185 : start - 11185 + 250] = np.nan
    df = ball_df(x)
    df.index = pd.Index(frames, name="frame")
    return df, config


def test_minute_axis_uses_the_match_clock_and_shows_the_video_time():
    ball, config = real_first_half()
    ax = plot_frames_without_ball(ball, config, title="first_half", x_axis="minute")
    ax.figure.canvas.draw()
    x = ax.lines[0].get_xdata()
    assert x[0] == pytest.approx(0.0) and x[-1] == pytest.approx((90651 - 11185) / 25 / 60)  # 0 .. 52:58
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "45" in labels and "45+5" in labels and "50" not in labels  # stoppage time as on TV
    assert ax.get_xlabel() == "match minute"
    assert "from frames 81685-81934, match 45+2:00, tactical video 0:54:27.40" in ax.get_title(loc="right")
    top = ax.child_axes[0]  # the video time axis: match minute 0 is video minute 7.46 (frame 11185)
    assert top.get_xlim()[0] - ax.get_xlim()[0] == pytest.approx(11185 / 25 / 60)
    assert top.xaxis.get_major_formatter()(75.5, 0) == "1:15"  # h:mm
    matplotlib.pyplot.close(ax.figure)


def test_another_video_gets_its_own_time_and_frames_on_top():
    ball, config = real_first_half()
    offset = 4014  # tactical frame = broadcast frame + 4014
    ax = plot_frames_without_ball(ball, config, x_axis="minute", video="broadcast", video_offset=offset)
    ax.figure.canvas.draw()  # the top axis gets its limits when drawn
    top = ax.child_axes[0]  # match minute 0 is broadcast frame 11185 - 4014 = 7171, i.e. minute 4.78
    assert top.get_xlim()[0] - ax.get_xlim()[0] == pytest.approx((11185 - offset) / 25 / 60)
    assert top.get_xlabel() == "time in the broadcast video (h:mm)"
    assert "broadcast video 0:51:46.84" in ax.get_title(loc="right")  # (81685 - 4014) / 25 s
    matplotlib.pyplot.close(ax.figure)

    ax = plot_frames_without_ball(ball, config, x_axis="frame", video="broadcast", video_offset=offset,
                                  compact=True)
    assert ax.get_xlabel() == "tactical video frame"
    assert ax.child_axes[0].get_xlabel() == "broadcast video frame"
    assert ax.get_title(loc="right") == "longest 10.0 s, from 45+2:00"
    matplotlib.pyplot.close(ax.figure)


def test_gap_length():
    seen = np.array([False, True, False, False, True, False])
    assert list(gap_length(seen)) == [1, 0, 2, 2, 0, 1]


def test_merge_takes_the_second_source_only_where_the_first_has_none():
    first = ball_df([0.1, np.nan, np.nan, 0.4])
    second = ball_df([0.9, 0.2, np.nan, 0.9])
    both = merge_ball(first, second)
    np.testing.assert_allclose(both["ball_x"], [0.1, 0.2, np.nan, 0.4])
    assert list(both["ball_from"]) == ["first", "second", "none", "first"]


def test_merge_fills_only_the_long_gaps():
    x = np.full(20, 0.5)
    x[2:4] = np.nan  # 2 frames: short, left for the interpolation
    x[8:14] = np.nan  # 6 frames: long
    first, second = ball_df(x), ball_df(np.full(20, 0.9))
    merged = merge_ball(first, second, min_gap_frames=5)
    assert list(np.flatnonzero(merged["ball_from"] == "second")) == list(range(8, 14))
    assert merged["ball_x"][2:4].isna().all() and (merged["ball_x"][8:14] == 0.9).all()
    assert list(np.flatnonzero(gaps_to_fill(first, second, 5))) == list(range(8, 14))


def test_merged_halves_go_into_the_ball_of_the_whole_match():
    x = np.full(30, 0.5)
    x[22:24] = np.nan  # halftime, untouched
    whole = ball_df(x, phase=["first_half"] * 20 + ["halftime"] * 10)
    half = whole.iloc[:20].copy()
    half.loc[[5, 6], "ball_x"] = 0.9
    half["ball_from"] = "first"
    half.loc[[5, 6], "ball_from"] = "second"
    out = insert_halves(whole, {"first_half": half})
    assert list(out.loc[[4, 5, 6], "ball_x"]) == [0.5, 0.9, 0.9]
    assert list(out.loc[[5, 22, 25], "ball_from"]) == ["second", "none", "first"]
    # the analysis keeps track of which video each sighting came from
    analysed = analyze_ball(out, CONFIG)
    assert analysed.loc[5, "ball_from"] == "second"
    assert phase_table(analysed).loc["first_half", "from_video_2"] == "10.0%"


def test_merge_leaves_out_what_is_not_usable():
    x = np.full(10, 0.5)
    x[2:8] = np.nan
    usable = np.zeros(10, dtype=bool)
    usable[2:5] = True  # e.g. the broadcast shows frames 5-7 as a replay
    merged = merge_ball(ball_df(x), ball_df(np.full(10, 0.9)), min_gap_frames=0, usable=usable)
    assert list(merged["ball_from"][2:8]) == ["second"] * 3 + ["rejected"] * 3
    assert merged["ball_x"][5:8].isna().all()
    assert merged["ball_x"].dtype == np.float64


def test_minute_axis_needs_one_half():
    ball, config = real_first_half()
    with pytest.raises(ValueError, match="outside the halves"):
        plot_frames_without_ball(ball.set_axis(pd.RangeIndex(0, len(ball))).iloc[:100], config, x_axis="minute")
    with pytest.raises(ValueError, match="x_axis must be"):
        plot_frames_without_ball(ball, config, x_axis="seconds")
    matplotlib.pyplot.close("all")


def test_tables():
    x = np.full(30, 0.5)
    x[3:5] = np.nan  # 2 frames, filled
    x[10:17] = np.nan  # 7 frames, missing
    x[22:24] = np.nan  # in halftime: not graded, not in the gap table
    ball = analyze_ball(ball_df(x, phase=["first_half"] * 20 + ["halftime"] * 10), CONFIG)
    phases = phase_table(ball)
    assert phases.loc["first_half", "interpolated"] == "10.0%" and phases.loc["first_half", "missing"] == "35.0%"
    gaps = gap_table(ball, CONFIG)
    assert gaps.loc["up to 0.2 s", "gaps"] == 1 and gaps.loc["0.2-1 s", "frames"] == 7
    assert gaps["gaps"].sum() == 2
    assert "Ball per phase" in summary(ball, CONFIG)
