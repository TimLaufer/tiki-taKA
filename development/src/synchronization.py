"""Compute, validate and cache broadcast alignment; no hand-entered offsets."""
import json
from pathlib import Path

import numpy as np

from src.broadcast_sync import estimate_offsets, match_samples
from src.data_loader import (_find_file, _resolve, atd_frame_rate, atd_paths,
                             file_sha256, get_data_dir)
from src.progress import say


def accepted_offsets(table, config):
    settings = config["sync"]
    offsets = {}
    for row in table.itertuples():
        if row.hypothesis != "one offset per half":
            continue
        if (row.matches < settings.get("min_matches", 10)
                or row.share < settings.get("min_agreement", 0.85)
                or not np.isfinite(row.slope)
                or abs(row.slope - 1) > settings.get("max_slope_error", 0.002)):
            raise ValueError(f"Broadcast synchronization unreliable for {row.part}")
        offsets[row.part] = int(row.offset)
    wanted = {p["name"] for p in config["phases"] if "clock_start_min" in p}
    if set(offsets) != wanted:
        raise ValueError("Broadcast synchronization did not cover every half")
    return offsets


def synchronize(tactical, broadcast, quality, config):
    """Recompute on any source/config/algorithm change; reject unreliable alignment."""
    t_fps, b_fps = (atd_frame_rate(config, source) for source in ("tactical", "broadcast"))
    if t_fps != b_fps or t_fps != config["fps"]:
        raise ValueError("Broadcast merge requires equal recording and configured frame rates")
    quality_path = _find_file(get_data_dir(config),
                             f"{config['data']['game']}_broadcast_homography.csv", "broadcast", None)
    sources = [*atd_paths(config, "tactical"), *atd_paths(config, "broadcast"), quality_path]
    code = [Path(__file__), Path(__file__).with_name("broadcast_sync.py"),
            Path(__file__).with_name("positions.py"), Path(__file__).with_name("data_loader.py")]
    signature = {"sources": {str(p.resolve()): file_sha256(p) for p in sources},
                 "code": {p.name: file_sha256(p) for p in code},
                 "sync": config["sync"], "pitch": config["pitch"], "fps": t_fps,
                 "phases": config["phases"]}
    cache = _resolve(config["data"]["cache_dir"]) / f"{config['data']['game']}_sync.json"
    if cache.is_file():
        try:
            stored = json.loads(cache.read_text())
            if stored["signature"] == signature:
                import pandas as pd
                offsets = accepted_offsets(pd.DataFrame(stored["quality"]), config)
                say(f"computed broadcast alignment from cache: {offsets}")
                return offsets
        except (ValueError, KeyError, TypeError):
            pass
    matches = match_samples(tactical, broadcast, quality, config, t_fps, b_fps, progress=True)
    _, table = estimate_offsets(matches, config, t_fps)
    offsets = accepted_offsets(table, config)
    records = table[table["hypothesis"] == "one offset per half"].to_dict("records")
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Cache contents contain computed results only, and are covered by the signature.
    cache.write_text(json.dumps({"signature": signature, "quality": records}, indent=2, allow_nan=False))
    say(f"computed broadcast alignment: {offsets}")
    return offsets
