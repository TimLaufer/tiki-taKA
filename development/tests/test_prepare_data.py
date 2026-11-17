"""Archive preparation uses synthetic files only."""
import importlib.util
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location("prepare_data", Path(__file__).resolve().parents[2] / "prepare_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_extracts_known_assets_safely_and_is_idempotent(tmp_path):
    with zipfile.ZipFile(tmp_path / "download.zip", "w") as bundle:
        bundle.writestr("../../BAR-ATH_tactical_atd.txt", "synthetic tracking")
        bundle.writestr("dataset/tactical/BAR-ATH_tactical_atd.xml", "synthetic metadata")
        bundle.writestr("../../unrelated.txt", "do not extract")
    assert len(module.prepare(tmp_path)) == 2
    assert (tmp_path / "tactical/BAR-ATH_tactical_atd.txt").read_text() == "synthetic tracking"
    assert not (tmp_path / "unrelated.txt").exists()
    assert module.prepare(tmp_path) == []
    target = tmp_path / "tactical/BAR-ATH_tactical_atd.txt"
    target.write_text("different content")
    with pytest.raises(ValueError, match="differs"):
        module.prepare(tmp_path)
    assert target.read_text() == "different content"


def test_zero_length_video_and_partial_download_are_not_linked(tmp_path):
    (tmp_path / "BAR-ATH_broadcast-001.mp4").touch()
    (tmp_path / "BAR-ATH_broadcast-001.mp4.part").write_bytes(b"partial")
    module.prepare(tmp_path)
    assert not (tmp_path / "broadcast/BAR-ATH_broadcast.mp4").exists()
