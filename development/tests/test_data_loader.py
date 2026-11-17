from pathlib import Path

import numpy as np
import pytest

from src.data_loader import _cache_path, get_data_dir, load_config, load_tracking, summary


def test_config_points_to_documented_data_directory(monkeypatch):
    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    assert get_data_dir(load_config()) == Path(__file__).resolve().parents[2] / "data"


def test_env_var_overrides_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MTAH_DATA_DIR", str(tmp_path))
    assert get_data_dir(load_config()) == tmp_path.resolve()


def test_explicit_file_paths_override_naming_scheme(epts_config, tmp_path):
    nested = tmp_path / "data" / "download-part-1" / "Dataset" / "tactical"
    nested.mkdir(parents=True)
    standard = tmp_path / "data" / "tactical"
    (standard / "TEST_tactical_atd.xml").rename(nested / "meta.xml")
    (standard / "TEST_tactical_atd.txt").rename(nested / "raw.txt")
    epts_config["data"]["metadata_file"] = "download-part-1/Dataset/tactical/meta.xml"
    epts_config["data"]["raw_file"] = "download-part-1/Dataset/tactical/raw.txt"
    assert load_tracking(epts_config, use_cache=False).n_frames == 5


def _move_to_download_part(tmp_path, part):
    nested = tmp_path / "data" / part / "Dataset" / "tactical"
    nested.mkdir(parents=True)
    for suffix in (".xml", ".txt"):
        (tmp_path / "data" / "tactical" / f"TEST_tactical_atd{suffix}").rename(nested / f"TEST_tactical_atd{suffix}")


def test_finds_files_in_unmerged_download_folders(epts_config, tmp_path):
    _move_to_download_part(tmp_path, "download-part-5")
    assert load_tracking(epts_config, use_cache=False).n_frames == 5


def test_stale_configured_path_falls_back_to_standard_layout(epts_config):
    epts_config["data"]["metadata_file"] = "moved-away/TEST_tactical_atd.xml"
    assert load_tracking(epts_config, use_cache=False).n_frames == 5


def test_missing_file_names_what_was_tried(epts_config, tmp_path):
    (tmp_path / "data" / "tactical" / "TEST_tactical_atd.txt").unlink()
    with pytest.raises(FileNotFoundError, match="TEST_tactical_atd.txt not found"):
        load_tracking(epts_config, use_cache=False)


def test_duplicate_files_are_an_error(epts_config, tmp_path):
    _move_to_download_part(tmp_path, "part-1")
    xml = tmp_path / "data" / "part-1" / "Dataset" / "tactical" / "TEST_tactical_atd.xml"
    copy = tmp_path / "data" / "part-2" / xml.name
    copy.parent.mkdir()
    copy.write_bytes(xml.read_bytes())
    with pytest.raises(RuntimeError, match="more than once"):
        load_tracking(epts_config, use_cache=False)


def test_parse_puts_atd_frame_ids_on_video_frames(epts_config):
    t = load_tracking(epts_config, use_cache=False)
    assert t.n_frames == 5
    assert list(t.team) == ["H", "A"] and list(t.slot_ids) == ["1", "2"]
    assert list(t.team_names) == ["Home FC", "Away FC"]
    assert t.home_attacks_first_half == "left_to_right"
    # ATD frame_id 1 is video frame 0
    np.testing.assert_allclose(t.players[0], [[0.10, 0.20], [0.60, 0.70]], rtol=1e-6)
    np.testing.assert_allclose(t.ball[1], [0.51, 0.41], rtol=1e-6)
    np.testing.assert_allclose(t.speed[0], [1.5, 2.5], rtol=1e-6)


def test_ball_speed_and_speed_unit_come_from_the_file(epts_config):
    t = load_tracking(epts_config, use_cache=False)
    assert t.speed_unit == "kph"
    np.testing.assert_allclose(t.ball_speed[:2], [12.0, 12.5], rtol=1e-6)
    assert np.isnan(t.ball_speed[3])  # empty field


def test_missing_values_are_nan(epts_config):
    t = load_tracking(epts_config, use_cache=False)
    assert np.isnan(t.ball[2]).all()  # NaN in the file
    assert np.isnan(t.players[3, 1]).all() and not np.isnan(t.players[3, 0]).any()  # empty fields
    assert np.isnan(t.players[4]).all() and np.isnan(t.ball[4]).all()  # no line at all
    assert np.isnan(t.ball_z).all()  # no ball height in this export


def test_cache_round_trip(epts_config):
    fresh = load_tracking(epts_config)
    assert _cache_path(epts_config, "tactical").exists()
    cached = load_tracking(epts_config)
    np.testing.assert_array_equal(cached.players, fresh.players)
    np.testing.assert_array_equal(cached.team, fresh.team)
    np.testing.assert_array_equal(cached.ball_speed, fresh.ball_speed)
    assert cached.home_attacks_first_half == fresh.home_attacks_first_half
    assert cached.speed_unit == "kph"


def test_switching_data_directory_cannot_reuse_old_cache(epts_config, tmp_path):
    import os
    import shutil

    fresh = load_tracking(epts_config)
    original = Path(epts_config["data"]["dir"])
    other = tmp_path / "other_data"
    shutil.copytree(original, other)
    raw = other / "tactical" / "TEST_tactical_atd.txt"
    raw.write_text(raw.read_text().replace("0.50,0.40,12.0", "0.90,0.40,12.0"))
    # Both exports predate the shared cache, defeating a simple mtime-only check.
    for path in (other / "tactical").iterdir():
        os.utime(path, (1, 1))
    changed = {**epts_config, "data": {**epts_config["data"], "dir": str(other)}}
    reloaded = load_tracking(changed)
    assert fresh.ball[0, 0] == pytest.approx(0.5)
    assert reloaded.ball[0, 0] == pytest.approx(0.9)


def test_frame_rate_and_homography_quality(epts_config, tmp_path):
    from src.data_loader import atd_frame_rate, load_homography_quality

    assert atd_frame_rate(epts_config, "tactical") == 25.0
    (tmp_path / "data" / "tactical" / "TEST_tactical_homography.csv").write_text(
        "frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n"
        "0,1,0,0,0,1,0,0,0,1,0.2,1\n"
        "1,,,,,,,,,,,0\n"
    )
    quality = load_homography_quality(epts_config, "tactical")
    assert list(quality["valid"]) == [True, False]
    assert quality["reprojection_error"].iloc[0] == 0.2


def test_summary(epts_config):
    text = summary(load_tracking(epts_config, use_cache=False), epts_config)
    assert "Home FC: 1 slots" in text
    assert "speed unit (metadata)   kph" in text
    assert "ball speed              median 12.2, 99th percentile 12.5, max 12.5 kph" in text


def test_a_missing_data_dir_falls_back(tmp_path, monkeypatch):
    from src.data_loader import get_data_dir

    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    (tmp_path / "local").mkdir()
    config = {"data": {"dir": str(tmp_path / "shared"), "fallback_dirs": [str(tmp_path / "local")]}}
    assert get_data_dir(config) == tmp_path / "local"
    (tmp_path / "shared").mkdir()
    assert get_data_dir(config) == tmp_path / "shared"  # the shared layout wins where it exists
    monkeypatch.setenv("MTAH_DATA_DIR", str(tmp_path / "local"))
    assert get_data_dir(config) == tmp_path / "local"  # and the variable wins over both
