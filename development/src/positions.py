"""Player positions in metres, shared by the matching (broadcast_sync) and the dead-ball indicators."""
from __future__ import annotations

import numpy as np

from src.data_loader import Tracking

ON_PITCH_MARGIN = 0.02  # tracked people further outside the lines (in pitch fractions) are not in play


def players_on_pitch_m(tracking: Tracking, pitch) -> np.ndarray:
    """(frames, slots, 2) positions in metres; NaN for empty slots and anyone off the pitch."""
    xy = tracking.players.astype(np.float32)
    off = ((xy < -ON_PITCH_MARGIN) | (xy > 1 + ON_PITCH_MARGIN)).any(axis=-1, keepdims=True)
    return np.where(off, np.nan, xy) * np.asarray(pitch, dtype=np.float32)


def pack_filled(xy: np.ndarray) -> np.ndarray:
    """Per frame the filled slots first, and the slot columns no frame needs cut off. About
    half of the slots are empty at any moment, so this halves the work of pairwise distances.
    Which slot a player sits in is lost, which is fine wherever only positions count."""
    empty = np.isnan(xy[:, :, 0])
    order = np.argsort(empty, axis=1, kind="stable")
    packed = np.take_along_axis(xy, order[:, :, None], axis=1)
    return packed[:, : max(int((~empty).sum(axis=1).max()), 1)]
