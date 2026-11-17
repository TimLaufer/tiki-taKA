import numpy as np
import pytest

from src.data_loader import load_config, load_tracking
from src.inspect_data import to_dataframe
from src.phases import (
    check_phases,
    clock_to_frame,
    first_graded_frame,
    graded_mask,
    match_clock,
    match_seconds,
    overview,
    phase_at,
    split_dataframe,
    split_tracking,
    video_time,
)


def test_video_time():
    config = load_config()
    assert video_time(0, config) == "0:00:00.00"
    assert video_time(20000, config) == "0:13:20.00"
    assert video_time(199318, config) == "2:12:52.72"


def test_match_clock_starts_at_each_kickoff():
    config = load_config()
    assert match_clock(11185, config) == "0:00"  # first kickoff
    assert match_clock(11185 + 25 * (23 * 60 + 10), config) == "23:10"
    assert match_clock(11185 + 25 * 45 * 60, config) == "45+0:00"
    assert match_clock(90651, config) == "45+7:58"  # last frame of the first half (DATA.md)
    assert match_clock(113742, config) == "45:00"  # second kickoff
    assert match_clock(190197, config) == "90+5:58"  # final whistle
    for no_play in (0, 11184, 100000, 112000, 199318):  # before, halftime, before 2nd kickoff, after
        assert match_clock(no_play, config) == ""


def test_match_seconds_for_many_frames():
    seconds = match_seconds([11185, 11185 + 25 * 60, 100000, 113742, 190197], load_config())
    np.testing.assert_allclose(seconds[[0, 1, 3]], [0, 60, 2700])
    assert np.isnan(seconds[2])  # halftime: the clock is not running
    assert seconds[4] == pytest.approx(2700 + (190197 - 113742) / 25)


def test_clock_to_frame():
    config = load_config()
    assert clock_to_frame("0:00", config) == 11185
    assert clock_to_frame("23:10", config) == 11185 + 25 * (23 * 60 + 10)
    assert clock_to_frame("23:10.5", config) == 11185 + 25 * (23 * 60 + 10) + 12  # 12.5 rounds to even
    assert clock_to_frame("45+2:30", config) == 11185 + 25 * (47 * 60 + 30)
    assert clock_to_frame("45", config) == 113742
    assert clock_to_frame("67:05", config) == 113742 + 25 * (22 * 60 + 5)
    assert clock_to_frame("90+1:00", config) == 113742 + 25 * 46 * 60
    for frame in (11185, 50000, 90651, 113742, 150000, 190197):
        # the clock shows whole seconds, so the way back lands up to 24 frames earlier
        assert 0 <= frame - clock_to_frame(match_clock(frame, config), config) < 25


def test_clock_to_frame_rejects_impossible_times():
    config = load_config()
    with pytest.raises(ValueError, match="after the end of first_half"):
        clock_to_frame("45+9:00", config)
    with pytest.raises(ValueError, match="stoppage time only after"):
        clock_to_frame("30+1:00", config)
    with pytest.raises(ValueError, match="after the end of second_half"):
        clock_to_frame("99:00", config)


def test_real_config_matches_data_md():
    config = load_config()
    check_phases(config, 199319)  # every frame in exactly one phase
    assert graded_mask(199319, config).sum() == 158857
    assert first_graded_frame(config) == 11185
    assert phase_at(113741, config) == "second_half_pre_kickoff"
    assert phase_at(113742, config) == "second_half"
    assert phase_at(199318, config) == "post_match"


def test_gaps_and_wrong_lengths_are_errors(epts_config):
    with pytest.raises(ValueError, match="data at frame 5"):
        check_phases(epts_config, 6)
    epts_config["phases"][1]["first"] = 4  # gap after frame 2
    with pytest.raises(ValueError, match="halftime should start at frame 3"):
        check_phases(epts_config, 5)


@pytest.fixture
def tracking(epts_config):
    return load_tracking(epts_config, use_cache=False)


def test_split_tracking_keeps_structure_and_frame_numbers(tracking, epts_config):
    parts = split_tracking(tracking, epts_config)
    assert list(parts) == ["first_half", "halftime", "second_half"]
    first, second = parts["first_half"], parts["second_half"]
    assert first.players.shape == (3, 2, 2) and first.speed.shape == (3, 2) and first.ball.shape == (3, 2)
    assert list(first.frames) == [0, 1, 2] and list(second.frames) == [4]
    np.testing.assert_array_equal(parts["halftime"].players[0], tracking.players[3])
    assert list(second.team) == ["H", "A"]  # per-track info stays whole
    # a part can be cut further, still in video frames
    assert list(first.slice(1, 2).frames) == [1, 2]
    with pytest.raises(IndexError):
        first.slice(2, 4)


def test_split_dataframe_keeps_the_frame_index(tracking, epts_config):
    dfs = split_dataframe(to_dataframe(tracking, epts_config), epts_config)
    assert list(dfs["halftime"].index) == [3]
    assert sum(len(d) for d in dfs.values()) == 5


def test_overview(tracking, epts_config):
    table = overview(tracking, epts_config)
    assert list(table.index) == ["first_half", "halftime", "second_half"]
    assert list(table["frames"]) == [3, 1, 1]
    assert table.loc["first_half", "ball_visible"] == "66.7%"  # ball missing in frame 2
