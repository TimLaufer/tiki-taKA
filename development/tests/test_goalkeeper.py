"""The goalkeeper model on a made-up half of 300 frames on a 100 m x 50 m pitch. H defends the
left goal (it stands in the left half at the kickoff), slot 0 is its keeper track. A has no
keeper track and defends the right goal."""
from dataclasses import replace

import numpy as np
import pytest

from src.data_loader import Tracking
from src.goalkeeper import defended_goals, keeper_slots, keeper_table, place_keepers
from src.player_cleaning import KEEPER_DEEPEST, KEEPER_HELD, NONE, OBSERVED

PITCH = [100.0, 50.0]
T = 300
CONFIG = {
    "fps": 25,
    "pitch": PITCH,
    "phases": [{"name": "first_half", "first": 0, "last": T - 1, "half": 1, "graded": True, "clock_start_min": 0}],
    "goalkeeper": {"margin_m": 5.0, "fallback_depth_m": 5.0},
}


def made_up_half(direction="left_to_right"):
    pos = np.full((T, 6, 2), np.nan)
    pos[:100, 0] = [5.0, 25.0]  # H keeper track, lost from frame 100 ...
    pos[250, 0] = [70.0, 25.0]  # ... once reported beyond the halfway line ...
    pos[260:, 0] = [10.0, 30.0]  # ... and seen again from frame 260, somewhere else
    pos[:, 1] = [40.0, 20.0]  # H field players
    pos[100:200, 1] = [8.0, 25.0]  # one of them deep in the box, 22 m behind the next: the keeper
    pos[:, 2] = [30.0, 25.0]
    pos[:, 3] = [60.0, 25.0]  # A field players, right half at the kickoff
    pos[:, 4] = [70.0, 30.0]
    pos[150:, 4] = [97.0, 22.0]  # deep in A's box from frame 150
    team = np.array(["H", "H", "H", "A", "A", "A"])
    tracking = Tracking(
        ball=np.full((T, 2), np.nan, np.float32), ball_z=np.full(T, np.nan, np.float32),
        ball_speed=np.full(T, np.nan, np.float32), players=(pos / PITCH).astype(np.float32),
        speed=np.full((T, 6), np.nan, np.float32), speed_unit="", team=team,
        slot_ids=np.array(["24", "2", "3", "26", "27", "28"]), team_names=np.array(["Home", "Away"]),
        home_attacks_first_half=direction,
    )
    source = np.where(np.isnan(pos[:, :, 0]), NONE, OBSERVED).astype(np.uint8)
    return tracking, source


def at(xy, frames):
    return np.repeat([xy], frames, axis=0)


@pytest.fixture(scope="module")
def placed():
    tracking, source = made_up_half()
    return place_keepers(tracking, source, CONFIG, {"H": 0})


def test_goals_come_from_the_kickoff(capsys):
    tracking, _ = made_up_half()
    assert defended_goals(tracking, CONFIG) == {("first_half", "H"): "left", ("first_half", "A"): "right"}
    assert "note" not in capsys.readouterr().err
    wrong, _ = made_up_half(direction="right_to_left")  # metadata disagrees with the kickoff
    assert defended_goals(wrong, CONFIG)[("first_half", "H")] == "left"
    assert "the kickoff is taken" in capsys.readouterr().err


def test_every_team_has_one_keeper_slot(placed):
    keepers, source = placed
    assert list(keepers.slot_ids) == ["GKH", "2", "3", "26", "27", "28", "GKA"]
    assert not np.isnan(keepers.players[:, [0, 6], 0]).any()  # a keeper in every frame


def test_unseen_keeper_stays_where_last_seen(placed):
    keepers, source = placed
    h = keepers.players[:, 0] * PITCH
    np.testing.assert_allclose(h[:100], at([5.0, 25.0], 100), atol=1e-4)  # keeper track
    assert (source[:100, 0] == OBSERVED).all()
    np.testing.assert_allclose(h[100:200], at([8.0, 25.0], 100), atol=1e-4)  # the deepest player
    assert (source[100:200, 0] == KEEPER_DEEPEST).all()
    assert np.isnan(keepers.players[100:200, 1]).all()  # moved to the keeper slot, not counted twice
    # unseen from 200: held where last seen - frame 250 too, beyond his half he is no keeper
    np.testing.assert_allclose(h[200:260], at([8.0, 25.0], 60), atol=1e-4)
    assert (source[200:260, 0] == KEEPER_HELD).all()
    np.testing.assert_allclose(h[260:], at([10.0, 30.0], 40), atol=1e-4)  # seen again: his new position
    assert (source[260:, 0] == OBSERVED).all()


def test_before_the_first_sighting_he_stands_where_first_seen(placed):
    keepers, source = placed
    a = keepers.players[:, 6] * PITCH
    np.testing.assert_allclose(a, at([97.0, 22.0], T), atol=1e-4)
    assert (source[:150, 6] == KEEPER_HELD).all() and (source[150:, 6] == KEEPER_DEEPEST).all()
    table = keeper_table(keepers, source, CONFIG)
    assert list(table["keeper"]) == ["GKH", "GKA"]
    assert table.loc[0, ["seen (keeper track)", "seen (deepest player)", "held (last seen)"]].tolist() == [
        "46.7%", "33.3%", "20.0%"]


def test_never_seen_in_a_half_he_stands_in_front_of_his_goal():
    tracking, source = made_up_half()
    players = tracking.players.copy()
    players[150:, 4] = players[0, 4]  # A's deep man never comes
    keepers, source = place_keepers(replace(tracking, players=players), source,
                                    CONFIG, {"H": 0})
    np.testing.assert_allclose(keepers.players[:, 6] * PITCH, at([95.0, 25.0], T), atol=1e-4)
    assert (source[:, 6] == KEEPER_HELD).all()


def test_the_indicators_leave_held_keepers_out(placed):
    from src.pipeline import MatchData, measured

    keepers, source = placed
    data = MatchData(tracking=keepers, df=None, parts={}, dfs={}, ball=None, player_source=source)
    without = measured(data).players
    assert np.isnan(without[200:260, 0]).all() and np.isnan(without[:150, 6]).all()
    assert not np.isnan(without[100:200, 0]).any()  # the deepest player is a measurement
    assert measured(MatchData(tracking=keepers, df=None, parts={}, dfs={}, ball=None)) is keepers


def test_keeper_slots_from_the_metadata(tmp_path):
    tracking, _ = made_up_half()
    meta = tmp_path / "meta.xml"
    meta.write_text(
        "<main><Metadata><Players>"
        '<Player id="24" teamId="TH"><ProviderPlayerParameters><ProviderParameter><Name>position_type</Name>'
        "<Value>Goalkeeper</Value></ProviderParameter></ProviderPlayerParameters></Player>"
        '<Player id="26" teamId="TA"><ProviderPlayerParameters><ProviderParameter><Name>position_type</Name>'
        "<Value>Unknown</Value></ProviderParameter></ProviderPlayerParameters></Player>"
        "</Players></Metadata></main>"
    )
    assert keeper_slots(meta, tracking) == {"H": 0}


def test_keepers_are_squares_in_the_viewer(placed):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.visualization import PitchPlot

    keepers, source = placed
    fig, ax = plt.subplots()
    plot = PitchPlot(ax, keepers, CONFIG, player_source=source)
    plot.draw(50)  # H: keeper track seen, A: held (before his first sighting)
    assert len(plot.keepers["H"].get_offsets()) == 1 and len(plot.keepers["A"].get_offsets()) == 1
    assert len(plot.dots["H"].get_offsets()) == 2  # the keeper is not a dot as well
    assert tuple(plot.keepers["H"].get_facecolors()[0]) != (1.0, 1.0, 1.0, 1.0)  # seen: full
    assert tuple(plot.keepers["A"].get_facecolors()[0]) == (1.0, 1.0, 1.0, 1.0)  # held: hollow
    assert {t.get_text() for t in plot.slot_labels if t.get_visible()} >= {"GK"}
    plt.close(fig)
