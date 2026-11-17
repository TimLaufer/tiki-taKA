import numpy as np
import pytest

from src.data_loader import Tracking
from src.player_cleaning import (
    NONE,
    OBSERVED,
    OTHER_TRACK,
    SAME_TRACK,
    VIDEO_2,
    bridge_tracks,
    clean_players,
    player_table,
)

PITCH = [100.0, 50.0]
T = 600
CONFIG = {
    "fps": 25,
    "pitch": PITCH,
    "phases": [{"name": "first_half", "first": 0, "last": T - 1, "half": 1, "graded": True, "clock_start_min": 0}],
    "players": {"max_gap_s": 10.0, "max_speed_ms": 7.0, "slack_m": 2.0, "switch_penalty_m": 2.0,
                "from_video_2": True, "per_team": 11, "extra_slots": 5, "match_radius_m": 3.0},
    "merge": {"max_player_rms_m": 3.0, "min_players": 4},
    "sync": {"cap_m": 5.0},
}


def tracking(positions_m, team):
    """positions in metres (frames, slots, 2) -> a Tracking in 0-1."""
    n, slots = positions_m.shape[:2]
    return Tracking(
        ball=np.full((n, 2), np.nan, np.float32), ball_z=np.full(n, np.nan, np.float32),
        ball_speed=np.full(n, np.nan, np.float32), players=(positions_m / PITCH).astype(np.float32),
        speed=np.full((n, slots), np.nan, np.float32), speed_unit="", team=np.array(team),
        slot_ids=np.array([str(i + 1) for i in range(slots)]), team_names=np.array(["Home", "Away"]),
        home_attacks_first_half="left_to_right",
    )


def gappy_tracks():
    """Slot 0 (H): running, lost in 100-149. Slot 1 (H): ends at 299, picked up by slot 3 (H) at
    340, 1 m on. Slot 2 (A): lost for 12 s in 200-499. Slot 4 (H): ends at 100; slot 5 (H) starts
    at 125, 60 m away - nobody runs that fast."""
    t = np.arange(T)
    pos = np.full((T, 6, 2), np.nan)
    pos[:, 0] = np.c_[10 + 0.1 * t, np.full(T, 45.0)]
    pos[100:150, 0] = np.nan
    pos[:300, 1] = [50.0, 30.0]
    pos[340:, 3] = [51.0, 30.0]
    pos[:, 2] = [70.0, 20.0]
    pos[200:500, 2] = np.nan
    pos[:101, 4] = [10.0, 10.0]
    pos[125:, 5] = [70.0, 10.0]
    return tracking(pos, ["H", "H", "A", "H", "H", "H"])


def test_a_track_picking_up_again_is_bridged_on_the_straight_line():
    players, source = bridge_tracks(gappy_tracks(), CONFIG)
    np.testing.assert_allclose(players[100:150, 0, 0] * PITCH[0], 10 + 0.1 * np.arange(100, 150), atol=1e-4)
    assert (source[100:150, 0] == SAME_TRACK).all()
    assert (source[:100, 0] == OBSERVED).all()


def test_a_new_track_taking_over_is_bridged_too():
    players, source = bridge_tracks(gappy_tracks(), CONFIG)
    assert (source[300:340, 1] == OTHER_TRACK).all()  # written into the slot that ended
    np.testing.assert_allclose(players[339, 1] * PITCH, [51.0 - 1 / 41, 30.0], atol=1e-3)
    assert (source[:340, 3] == NONE).all()  # the new track itself is left as it is


def test_too_long_impossible_or_other_team_stays_missing():
    players, source = bridge_tracks(gappy_tracks(), CONFIG)
    assert np.isnan(players[200:500, 2]).all()  # 12 s > 10 s; and no H track may take over an A player
    assert np.isnan(players[101:125, 4]).all()  # 60 m in 1 s


def full_teams(missing_home: int = 1):
    """11 H and 11 A standing in a grid; the tactical tracking misses the last H player(s)."""
    xs, ys = np.meshgrid(np.linspace(10, 90, 11), [12.0, 38.0])
    spots = np.c_[xs.ravel(), ys.ravel()]  # 22 spots, 8 m apart
    pos = np.repeat(spots[None], 120, axis=0)
    tactical = pos.copy()
    tactical[:, 11 - missing_home : 11] = np.nan
    broadcast = pos + np.random.default_rng(0).normal(0, 0.3, pos.shape)
    team = ["H"] * 11 + ["A"] * 11
    return tracking(tactical, team), tracking(broadcast, team)


def test_missing_players_come_from_video_2():
    from src.pipeline import Video2

    tactical, broadcast = full_teams()
    config = {**CONFIG, "phases": [{**CONFIG["phases"][0], "last": 119}]}
    cleaned, source = clean_players(tactical, Video2(tracking=broadcast, offsets={"first_half": 0}, balls={}), config)
    assert cleaned.players.shape == (120, 22 + 10, 2) and list(cleaned.slot_ids[22:24]) == ["v2H1", "v2H2"]
    extra_h = cleaned.players[:, 22:27]
    np.testing.assert_allclose(extra_h[:, 0] * PITCH, np.repeat([[90.0, 12.0]], 120, axis=0), atol=1.5)
    assert np.isnan(extra_h[:, 1:]).all()  # only the one missing, the others are duplicates
    assert np.isnan(cleaned.players[:, 27:]).all()  # A is complete
    assert (source[:, 22] == VIDEO_2).all() and (source[:, 23:] == NONE).all()
    table = player_table(tactical, cleaned, source, config)
    assert table.loc["first_half", "H players before"] == 10 and table.loc["first_half", "H after"] == 11


def test_no_players_from_a_replay():
    from src.pipeline import Video2

    tactical, broadcast = full_teams()
    moved = broadcast.players.copy()
    moved[50:100, :, 0] += 0.2  # the broadcast shows another moment here: 20 m off
    config = {**CONFIG, "phases": [{**CONFIG["phases"][0], "last": 119}]}
    video_2 = Video2(tracking=tracking(moved * PITCH, list(broadcast.team)), offsets={"first_half": 0}, balls={})
    _, source = clean_players(tactical, video_2, config)
    assert (source[:50, 22] == VIDEO_2).all() and (source[50:100, 22] == NONE).all()


def test_without_video_2_nothing_is_added():
    cleaned, source = clean_players(gappy_tracks(), None, CONFIG)
    assert cleaned.players.shape == (T, 6, 2) and source.shape == (T, 6)


@pytest.mark.parametrize("missing", [0])
def test_a_full_team_gets_nobody(missing):
    from src.pipeline import Video2

    tactical, broadcast = full_teams(missing_home=missing)
    config = {**CONFIG, "phases": [{**CONFIG["phases"][0], "last": 119}]}
    _, source = clean_players(tactical, Video2(tracking=broadcast, offsets={"first_half": 0}, balls={}), config)
    assert (source[:, 22:] == NONE).all()
