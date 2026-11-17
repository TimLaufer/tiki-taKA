"""Smooth team evidence without crossing dead-ball or half boundaries."""
import numpy as np

from src.phases import check_phases, graded_mask, half_of_frames


def postprocess(predictions, config: dict):
    labels = np.asarray(predictions)
    if labels.ndim != 1 or not np.isin(labels, ["H", "A", "D", "X"]).all():
        raise ValueError("Predictions must be a one-dimensional H/A/D/X array")
    check_phases(config, len(labels))
    seconds = config.get("model", {}).get("smoothing_half_window_s", 2.0)
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError("model.smoothing_half_window_s must be finite and nonnegative")
    window = round(seconds * config["fps"])
    graded = graded_mask(len(labels), config)
    half = half_of_frames(len(labels), config)
    out = labels.copy()
    out[~graded] = "X"
    alive = graded & (labels != "D")
    boundaries = np.r_[0, np.flatnonzero((alive[1:] != alive[:-1]) |
                                        (half[1:] != half[:-1])) + 1, len(labels)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if not alive[start] or window == 0:
            continue
        part = labels[start:stop]
        index = np.arange(len(part))
        low, high = np.maximum(0, index - window), np.minimum(len(part), index + window + 1)
        counts = []
        for team in "HA":
            cumulative = np.r_[0, np.cumsum(part == team)]
            counts.append(cumulative[high] - cumulative[low])
        home, away = counts
        out[start:stop] = np.where(home + away == 0, "X", np.where(home >= away, "H", "A"))
    return out
