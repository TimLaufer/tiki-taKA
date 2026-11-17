import numpy as np
import pytest

from src.data_loader import atd_paths, load_tracking
from src.inspect_data import describe_metadata, frame_table, to_dataframe


def test_describe_metadata_explains_the_raw_line(epts_config):
    text = describe_metadata(atd_paths(epts_config, "tactical")[0])
    assert "<frameCount> : [track 1: x,y,s] ; [track 2: x,y,s] : [ball: x,y,s]" in text
    assert "speed      channel s    unit kph" in text
    assert "Home FC" in text and "1 tracks" in text


@pytest.fixture
def tracking(epts_config):
    return load_tracking(epts_config, use_cache=False)


def test_dataframe_has_one_row_per_frame(tracking, epts_config):
    df = to_dataframe(tracking, epts_config)
    assert list(df.columns) == [
        "time_s", "phase", "half", "ball_x", "ball_y", "ball_s", "1_x", "1_y", "1_s", "2_x", "2_y", "2_s"
    ]
    assert df.attrs["speed_unit"] == "kph"
    assert list(df.index) == [0, 1, 2, 3, 4]
    assert list(df["phase"]) == ["first_half"] * 3 + ["halftime", "second_half"]
    assert list(df["half"]) == [1, 1, 1, 0, 2]
    assert df.loc[1, "time_s"] == pytest.approx(0.04)
    assert df.loc[0, "1_x"] == pytest.approx(0.10)
    assert np.isnan(df.loc[2, "ball_x"])
    assert df.attrs["team"] == {"1": "H", "2": "A"}


def test_frame_table_puts_ball_first_and_nearest_player_next(tracking, epts_config):
    table = frame_table(tracking, epts_config, 1)
    assert list(table["object"]) == ["ball", "2", "1"]  # slot 2 is closer to the ball
    assert table.loc[0, "dist_to_ball_m"] == 0
    assert table.loc[0, "speed_kph"] == pytest.approx(12.5)  # exported ball speed
    assert table.loc[1, "speed_kph"] == pytest.approx(2.6)
    assert table.loc[1, "x_m"] == pytest.approx(0.61 * 105.3, rel=1e-5)


def test_frame_table_without_ball(tracking, epts_config):
    table = frame_table(tracking, epts_config, 2)
    assert len(table) == 3
    assert table["dist_to_ball_m"].isna().all()
