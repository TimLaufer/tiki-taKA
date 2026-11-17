"""External labelled games (src.external) on synthetic files in Metrica's sample-data layout."""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_config
from src.external import (
    controlled_labels,
    degrade_ball,
    game_config,
    load_csv_tracking,
    load_events,
    periods,
    possession_labels,
)
from src.phases import check_phases


def events(*rows):
    return pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start", "end"])


ONE_PERIOD = events(
    ("H", "SET PIECE", "KICK OFF", 1, 10, 10),
    ("H", "PASS", "", 1, 12, 20),
    ("A", "RECOVERY", "INTERCEPTION", 1, 30, 30),
    ("A", "CHALLENGE", "GROUND-WON", 1, 35, 35),      # duels do not change possession
    ("A", "BALL OUT", "", 1, 40, 45),                 # out of bounds at 45
    ("H", "SET PIECE", "THROW IN", 1, 60, 60),
    ("H", "FAULT RECEIVED", "", 1, 70, 70),           # free kick awarded at once
    ("A", "CARD", "YELLOW", 1, 75, 75),               # no restart
    ("H", "SET PIECE", "FREE KICK", 1, 80, 80),
    ("H", "SHOT", "ON TARGET-GOAL", 1, 90, 92),       # goal at 92
    ("A", "CHALLENGE", "AERIAL-LOST", 1, 99, 99),
)


def test_event_labels_follow_metricas_definitions():
    labels = possession_labels(ONE_PERIOD, 110)
    expected = "X" * 10 + "H" * 20 + "A" * 15 + "D" * 15 + "H" * 10 + "D" * 10 + "H" * 12 + "D" * 8 + "X" * 10
    assert "".join(labels) == expected


def test_a_goal_kick_pass_is_not_a_goal():
    labels = possession_labels(events(("H", "PASS", "GOAL KICK", 1, 0, 30), ("H", "PASS", "", 1, 40, 50)), 51)
    assert set(labels) == {"H"}


def test_a_ball_lost_keeps_its_team_until_the_opponent_plays_the_ball():
    labels = possession_labels(events(("H", "PASS", "", 1, 0, 5), ("H", "BALL LOST", "INTERCEPTION", 1, 6, 9),
                                      ("A", "RECOVERY", "INTERCEPTION", 1, 12, 12), ("A", "PASS", "", 1, 20, 25)), 26)
    assert "".join(labels) == "H" * 12 + "A" * 14


def test_touches_without_control_go_back_to_the_previous_team():
    labels = np.array(list("D" + "H" * 100 + "A" * 30 + "H" * 100 + "D" + "A" * 10 + "D"))
    assert "".join(controlled_labels(labels, events(), min_s=2.0)) == "D" + "H" * 230 + "D" + "A" * 10 + "D"
    passed = events(("A", "PASS", "", 1, 105, 115))  # a completed pass is control, however short
    assert "".join(controlled_labels(labels, passed, min_s=2.0)) == "".join(labels)
    assert "".join(controlled_labels(labels, events(), min_s=1.0)) == "".join(labels)  # 1.2 s is long enough


def test_periods_become_a_complete_phase_table():
    config = game_config(load_config(), periods(events(("H", "PASS", "", 1, 10, 99), ("A", "PASS", "", 2, 150, 199))),
                         250)
    check_phases(config, 250)
    graded = [(p["name"], p["first"], p["last"]) for p in config["phases"] if p["graded"]]
    assert graded == [("first_half", 10, 99), ("second_half", 150, 199)]
    assert not config["sync"]["enabled"] and not config["smart_tagging"]["enabled"]


def write_csv_game(root, game=1, frames=4):
    folder = root / "data" / f"Sample_Game_{game}"
    folder.mkdir(parents=True)
    for side, first in (("Home", 1), ("Away", 15)):
        head = [",,," + ",".join([side, ""] * 2) + ",,",
                ",,," + ",".join([str(first), "", str(first + 1), ""]) + ",,",
                "Period,Frame,Time [s],Player%d,,Player%d,,Ball," % (first, first + 1)]
        rows = [f"1,{f},{f * 0.04:.2f},0.{f}0,0.50,NaN,NaN,0.50,0.5{f}" for f in range(1, frames + 1)]
        (folder / f"Sample_Game_{game}_RawTrackingData_{side}_Team.csv").write_text("\n".join(head + rows) + "\n")
    (folder / f"Sample_Game_{game}_RawEventsData.csv").write_text(
        "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,Start X,Start Y,End X,End Y\n"
        "Home,PASS,,1,1,0.04,3,0.12,Player1,Player2,0.1,0.5,0.3,0.5\n"
        "Away,ball lost,Interception,1,4,0.16,4,0.16,Player15,,0.4,0.5,,\n")
    return root


def test_csv_tracking_and_events_load_onto_the_frame_index_timeline(tmp_path):
    tracking = load_csv_tracking(write_csv_game(tmp_path), 1)
    assert tracking.n_frames == 4 and list(tracking.team) == ["H", "H", "A", "A"]
    assert list(tracking.slot_ids) == ["H1", "H2", "A15", "A16"]
    assert tracking.players[0, 0, 0] == pytest.approx(0.1) and np.isnan(tracking.players[:, 1]).all()
    assert tracking.ball[2, 1] == pytest.approx(0.53)
    table = load_events(tmp_path, 1)
    assert table[["team", "type", "start", "end"]].values.tolist() == [["H", "PASS", 0, 2], ["A", "BALL LOST", 3, 3]]


def test_csv_frames_must_be_complete(tmp_path):
    root = write_csv_game(tmp_path)
    path = root / "data" / "Sample_Game_1" / "Sample_Game_1_RawTrackingData_Away_Team.csv"
    path.write_text(path.read_text().replace("\n1,3,", "\n1,7,"))
    with pytest.raises(ValueError, match="frames"):
        load_csv_tracking(root, 1)


def test_challenge_like_ball_noise_is_reproducible_and_of_the_requested_size(tmp_path):
    tracking = load_csv_tracking(write_csv_game(tmp_path, frames=9), 1)
    from dataclasses import replace
    long = replace(tracking, ball=np.tile(np.array([[0.5, 0.5]], np.float32), (20000, 1)))
    a, b = degrade_ball(long, 1.0, 3.0, seed=1), degrade_ball(long, 1.0, 3.0, seed=1)
    assert np.array_equal(a.ball, b.ball) and not np.array_equal(a.ball, degrade_ball(long, 1.0, 3.0, seed=2).ball)
    error = (a.ball.astype(float) - 0.5) * [105.0, 68.0]
    assert error.std(axis=0) == pytest.approx([1.0, 1.0], abs=0.25)
    gap = replace(long, ball=np.where(np.arange(20000)[:, None] < 10, np.nan, long.ball).astype(np.float32))
    assert np.isnan(degrade_ball(gap, 1.0, 3.0, seed=1).ball[:10]).all()  # a missing ball stays missing
