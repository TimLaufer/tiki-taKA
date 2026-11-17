from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from src.pipeline import (  # noqa: E402
    BOTH,
    MERGED,
    VIDEO_1,
    VIDEO_2,
    Video2,
    ball_halves,
    ball_sources,
    describe,
    gap_tables,
    load_match,
    load_video_2,
    merge_table,
    merge_videos,
)
from src.ball_analysis import gap_stats, plot_gap_grid  # noqa: E402


def fake_video_2(data):
    """Video 2 = the same recording (offset 0), but it sees the ball only in frame 2 of the
    first half, where video 1 has none."""
    balls = {half: ball.copy() for half, ball in ball_halves(data.dfs).items()}
    balls["first_half"]["ball_x"] = [np.nan, np.nan, 0.7]
    balls["first_half"]["ball_y"] = [np.nan, np.nan, 0.3]
    return Video2(tracking=data.tracking, offsets={"first_half": 0, "second_half": 0}, balls=balls)


def no_min_gap(config, min_players):
    """The fixture's gap is a single frame, far below merge.min_gap_s."""
    return {**config, "merge": {**config["merge"], "min_gap_s": 0, "min_players": min_players}}


def test_halves_go_to_the_ball_analysis(epts_config):
    halves = ball_halves(load_match(epts_config).dfs)
    assert list(halves) == ["first_half", "second_half"]
    assert list(halves["first_half"].index) == [0, 1, 2]
    assert list(halves["first_half"].columns) == ["phase", "ball_x", "ball_y", "ball_s"]
    # fixture: ball seen in frames 0 and 1, missing in frame 2
    assert gap_stats(halves["first_half"], epts_config)["longest_gap_at_frames"] == "2-2"


def test_video_2_fills_the_gaps_of_video_1(epts_config):
    data = load_match(epts_config)
    balls = ball_sources(replace(data, video_2=fake_video_2(data)))
    assert list(balls) == [VIDEO_1, VIDEO_2, BOTH]
    both = balls[BOTH]["first_half"]
    assert list(both["ball_from"]) == ["first", "first", "second"]
    assert both["ball_x"].iloc[2] == 0.7
    assert gap_stats(both, epts_config)["gaps"] == 0
    text = gap_tables(balls, epts_config)
    assert "first_half:" in text and VIDEO_2 in text and BOTH in text


def test_merge_checks_the_players(epts_config):
    data = load_match(epts_config)
    video_2 = fake_video_2(data)
    # the broadcast here is the tactical recording itself, so its players match perfectly ...
    merged = merge_videos(data.tracking, data.dfs, video_2, no_min_gap(epts_config, min_players=1))
    assert list(merged["first_half"]["ball_from"]) == ["first", "first", "second"]
    # ... but frame 2 shows only 2 players, too few to check against min_players = 4
    config = no_min_gap(epts_config, min_players=4)
    merged = merge_videos(data.tracking, data.dfs, video_2, config)
    assert list(merged["first_half"]["ball_from"]) == ["first", "first", "rejected"]
    assert "rejected" in merge_table(merged, config)
    balls = ball_sources(replace(data, video_2=video_2, merged=merged))
    assert list(balls) == [VIDEO_1, VIDEO_2, BOTH, MERGED]


def test_merge_waits_for_long_gaps(epts_config):
    data = load_match(epts_config)
    merged = merge_videos(data.tracking, data.dfs, fake_video_2(data), epts_config)  # 5 s: a 1-frame gap stays
    assert list(merged["first_half"]["ball_from"]) == ["first", "first", "none"]


def test_without_broadcast_data_video_2_is_left_out(epts_config):
    assert load_video_2(epts_config) is None  # the fixture has no broadcast files
    assert load_video_2({**epts_config, "broadcast_offsets": None}) is None
    data = load_match(epts_config)
    assert data.video_2 is None and data.merged is None
    assert list(ball_sources(data)) == [VIDEO_1]
    assert set(data.ball["ball_from"]) <= {"first", "none"}  # the ball data is video 1 alone


def test_one_plot_with_a_row_per_half_and_video(epts_config):
    data = load_match(epts_config)
    halves = ball_halves(data.dfs)
    merged = merge_videos(data.tracking, data.dfs, fake_video_2(data), epts_config)
    sources = {VIDEO_1: (halves, "tactical", {}), VIDEO_2: (halves, "broadcast", {"first_half": 2, "second_half": 3}),
               MERGED: (merged, "tactical", {})}
    fig = plot_gap_grid(sources, epts_config)
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert titles == [f"{half} | {video}" for half in ("first_half", "second_half")
                      for video in (VIDEO_1, VIDEO_2, MERGED) for _ in ("frame", "minute")]
    assert [ax.get_xlabel() for ax in fig.axes[:4]] == ["video frame", "match minute",
                                                         "tactical video frame", "match minute"]
    matplotlib.pyplot.close(fig)


def test_load_match_splits_by_phase(epts_config):
    data = load_match(epts_config)
    assert list(data.parts) == ["first_half", "halftime", "second_half"]
    assert list(data.dfs) == list(data.parts)
    first = data.parts["first_half"]
    # 2 tracks, plus a keeper slot per team (the fixture marks no keeper tracks)
    assert first.players.shape == (3, 4, 2) and first.ball.shape == (3, 2)
    assert list(data.tracking.slot_ids[2:]) == ["GKH", "GKA"]
    assert list(first.frames) == [0, 1, 2]
    assert list(data.dfs["second_half"].index) == [4]
    assert len(data.df) == data.tracking.n_frames == 5
    # ball: frame 2 has no ball and is the last frame of first_half, so nothing is bridged
    # into halftime (frame 3); frame 4 has no line at all
    assert list(data.ball.index) == list(data.df.index)
    assert list(data.ball["source"]) == ["observed", "observed", "missing", "observed", "missing"]


def test_describe_lists_every_phase_with_its_shapes(epts_config):
    text = describe(load_match(epts_config), epts_config)
    assert "first_half" in text and "halftime" in text and "second_half" in text
    assert "(3, 4, 2)" in text  # players of first_half: 3 frames, 2 tracks + 2 keepers, x/y
    assert "speed: per track (kph)" in text
