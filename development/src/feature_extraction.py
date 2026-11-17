"""Collective proximity features on the tactical video timeline, plus the stoppage evidence."""
import numpy as np
import pandas as pd

from src.phases import check_phases, graded_mask, half_of_frames
from src.preprocessing import PreparedData
from src.positions import players_on_pitch_m


def extract_features(data: PreparedData, config: dict) -> pd.DataFrame:
    """Distances in metres to each team's nearest usable player; inf = unavailable."""
    tracking = data.tracking
    check_phases(config, tracking.n_frames)
    if tracking.first_frame != 0 or not np.array_equal(data.ball.index, tracking.frames):
        raise ValueError("Features require the full tactical timeline, starting at frame 0")
    ball = data.ball[["x", "y"]].to_numpy(dtype=float) * config["pitch"]
    players = players_on_pitch_m(tracking, config["pitch"])
    out = pd.DataFrame(index=pd.RangeIndex(tracking.n_frames, name="frame"))
    for team in "HA":
        nearest = np.full(tracking.n_frames, np.inf)
        slots = players[:, tracking.team == team]
        for start in range(0, tracking.n_frames, 8192):
            stop = start + 8192
            delta = slots[start:stop] - ball[start:stop, None]
            distances = np.linalg.norm(delta, axis=2)
            distances[~np.isfinite(distances)] = np.inf
            if distances.shape[1]:
                nearest[start:stop] = distances.min(axis=1)
        out[f"nearest_{team}_m"] = nearest
    out["dead"] = data.dead
    out["graded"] = graded_mask(tracking.n_frames, config)
    out["half"] = half_of_frames(tracking.n_frames, config)
    out["ball_source"] = data.ball["source"]
    out["ball_visible"] = np.isfinite(ball).all(axis=1)
    out["ball_speed_ms"] = data.ball["speed_ms"].to_numpy()
    # The team restarting play, at the first frame of each live stretch (src.stoppages).
    restart_team = np.full(len(out), "", dtype="<U1")
    for seed in data.seeds.itertuples():
        if seed.team in ("H", "A"):
            restart_team[int(seed.start)] = seed.team
    out["restart_team"] = restart_team
    out["hold_team"] = data.holds if data.holds is not None else np.full(len(out), "", dtype="<U1")
    return out
