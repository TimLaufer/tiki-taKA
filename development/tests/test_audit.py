import os
from pathlib import Path

import numpy as np
import pytest

from src.data_loader import atd_paths, check_frame_ids, load_tracking, _cache_path
from src.synchronization import accepted_offsets


@pytest.mark.parametrize("ids", [[0, 1], [1, 1], [2, 1], [1, 6]])
def test_parser_rejects_bad_frame_order_or_range(epts_config, ids):
    _, raw = atd_paths(epts_config, "tactical")
    line = raw.read_text().splitlines()[0].split(":", 1)[1]
    raw.write_text("".join(f"{i}:{line}\n" for i in ids))
    with pytest.raises(ValueError, match="frame"):
        load_tracking(epts_config, use_cache=False)


def test_submission_preflight_rejects_missing_frame(epts_config):
    _, raw = atd_paths(epts_config, "tactical")
    with pytest.raises(ValueError, match="records"):
        check_frame_ids(raw, 5)  # fixture intentionally omits the last record
    check_frame_ids(raw, 4)


def test_partial_coordinate_is_unavailable(epts_config):
    _, raw = atd_paths(epts_config, "tactical")
    raw.write_text(raw.read_text().replace("0.50,0.40,12.0", "0.50,,12.0"))
    assert np.isnan(load_tracking(epts_config, use_cache=False).ball[0]).all()


def test_content_edit_invalidates_cache_even_with_same_size_and_timestamp(epts_config):
    load_tracking(epts_config)
    _, raw = atd_paths(epts_config, "tactical")
    before = raw.stat()
    raw.write_text(raw.read_text().replace("0.50,0.40,12.0", "0.90,0.40,12.0"))
    os.utime(raw, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert load_tracking(epts_config).ball[0, 0] == pytest.approx(0.9)


def test_parser_change_invalidates_cache(epts_config, monkeypatch):
    import src.data_loader as loader

    load_tracking(epts_config)
    _, raw = atd_paths(epts_config, "tactical")
    signature = loader._source_signature((raw,))
    assert any(entry.startswith("parser|") for entry in signature)
    assert any(entry.startswith("kloppy|") for entry in signature)
    parsed = []
    original = loader.parse_atd
    monkeypatch.setattr(loader, "parse_atd", lambda *a, **k: parsed.append(1) or original(*a, **k))
    load_tracking(epts_config)
    assert parsed == []  # same parser, same data: from the cache
    monkeypatch.setattr(loader, "_parser_identity", lambda: ["parser|changed", "kloppy|0"])
    load_tracking(epts_config)
    assert parsed == [1]  # a different parser must not reuse arrays parsed by the old one


def test_corrupt_cache_is_rebuilt(epts_config):
    load_tracking(epts_config)
    _cache_path(epts_config, "tactical").write_bytes(b"incomplete write")
    with pytest.warns(UserWarning, match="unreadable"):
        result = load_tracking(epts_config)
    assert result.ball[0, 0] == pytest.approx(0.5)


def test_alignment_quality_gate_rejects_weak_agreement_or_drift():
    import pandas as pd
    config = {"sync": {}, "phases": [{"name": "first_half", "clock_start_min": 0}]}
    row = {"hypothesis": "one offset per half", "part": "first_half", "matches": 20,
           "share": 0.95, "slope": 1., "offset": 5}
    assert accepted_offsets(pd.DataFrame([row]), config) == {"first_half": 5}
    for update in [{"share": 0.5}, {"slope": 1.01}, {"matches": 2}]:
        with pytest.raises(ValueError, match="unreliable"):
            accepted_offsets(pd.DataFrame([{**row, **update}]), config)
