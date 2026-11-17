"""The smart-tagging check on the made-up half of test_restarts (H = Barcelona defends the left
goal; still balls: kickoff H, corner H, throw-in A, goal kick H, free kick A, keeper A) with
made-up tags: a true corner, a throw-in without team, a corner where nothing happens, a free
kick with the wrong team, and a build-up phase."""
from dataclasses import replace

import numpy as np
import pytest

from src.restarts import candidates
from src.smart_tagging import (
    check_set_pieces,
    load_tags,
    signatures,
    smart_tagging_path,
    untagged,
)
from tests.test_restarts import CONFIG as RESTART_CONFIG
from tests.test_restarts import made_up_half

CONFIG = {**RESTART_CONFIG, "smart_tagging": {"margin_s": 3.0, "smooth_s": 1.0, "line_band_m": 2.0,
                                              "confirm_pct": 0.9, "contradict_pct": 0.5}}
TEAMS = np.array(["Barcelona", "Atletic Club"])


def instance(tag, start, end, code, **labels):
    groups = "".join(f"<label><group>{g}</group><text>{t}</text></label>" for g, t in labels.items())
    return f"<instance><ID>{tag}</ID><start>{start}</start><end>{end}</end><code>{code}</code>{groups}" \
           f"<label><text>no group</text></label></instance>"


XML = ("<file><ALL_INSTANCES>"
       + instance(5, 24.0, 27.2, "SET PIECES", Team="Barcelona", Type="Free Kick")
       + instance(1, 6.0, 9.2, "SET PIECES", Team="Barcelona", Type="Corner Kick", Half="1st Half")
       + instance(2, 12.0, 15.2, "SET PIECES", Team="N/A", Type="Throw-in")
       + instance(3, 35.2, 38.0, "SET PIECES", Team="Atletic Club", Type="Corner Kick")
       + instance(4, 0.0, 5.0, "Build Up", Team="Barcelona")
       + "</ALL_INSTANCES></file>")


@pytest.fixture(scope="module")
def match(tmp_path_factory):
    tracking, ball = made_up_half()
    tracking = replace(tracking, team_names=TEAMS)
    path = tmp_path_factory.mktemp("tags") / "TEST_tactical_smart_tagging.xml"
    path.write_text(XML)
    tags = load_tags(path, CONFIG, TEAMS)
    still_balls = candidates(tracking, ball, CONFIG)
    table = check_set_pieces(tags, still_balls, signatures(tracking, ball, CONFIG), CONFIG)
    return tracking, ball, tags, still_balls, table


def test_load_tags(match):
    _, _, tags, _, _ = match
    assert list(tags["tag"]) == ["4", "1", "2", "5", "3"]  # in time order
    assert list(tags["code"]) == ["BUILD UP", "SET PIECES", "SET PIECES", "SET PIECES", "SET PIECES"]
    assert tags.loc[1, ["first", "last", "type", "half"]].tolist() == [150, 230, "Corner Kick", "1st Half"]
    assert list(tags["team_side"]) == ["H", "H", "", "H", "A"]  # N/A: no team


def test_set_pieces_checked_against_the_tracking(match):
    *_, table = match
    assert list(table["kind"]) == ["corner", "throw-in", "free kick", "corner"]
    assert list(table["verdict"]) == ["confirmed", "confirmed", "confirmed", "contradicted"]
    assert list(table["ours"]) == ["restart (corner H)", "restart (throw-in A)", "restart (free kick A)", "-"]
    assert list(table["team"]) == ["H", "A", "H", "A"]  # the throw-in's team is ours
    assert list(table["team_agrees"]) == ["yes", "-", "no", "-"]
    assert table.loc[3, "fit"] == 0.0  # nobody more in a box than at any other moment
    assert "A (ours, the tag says N/A)" in table.loc[1, "summary"]
    assert "the team differs from ours" in table.loc[2, "summary"]


def test_the_marks(match):
    tracking, ball, *_ = match
    marks = signatures(tracking, ball, CONFIG)
    assert marks.loc[340, "touchline"] == 1 and marks.loc[120, "touchline"] == 0  # A2 behind the line
    assert marks.loc[490, "goal_area"] == 1 and marks.loc[120, "goal_area"] == 0
    assert marks.loc[40, "own_half"] == 1.0  # everybody in his own half at the kickoff
    assert marks.loc[40, "slow"] == 0.0  # nobody moves


def test_our_restarts_without_a_tag(match):
    _, _, tags, still_balls, _ = match
    assert list(untagged(still_balls, tags, CONFIG)["kind"]) == ["kickoff", "goal kick"]


def test_the_file_is_found_in_the_data_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    (tmp_path / "tactical").mkdir()
    (tmp_path / "tactical" / "TEST_tactical_smart_tagging.xml").write_text(XML)
    config = {**CONFIG, "data": {"dir": str(tmp_path), "game": "TEST"}}
    assert smart_tagging_path(config).name == "TEST_tactical_smart_tagging.xml"


def test_tags_in_the_viewer(match):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.visualization import PitchPlot

    tracking, _, _, _, table = match
    fig, ax = plt.subplots()
    plot = PitchPlot(ax, tracking, CONFIG, episodes=table, segment_name="tag")
    plot.draw(160)
    assert "tag 1/4" in plot.caption.get_text() and "corner H - confirmed" in plot.caption.get_text()
    plt.close(fig)
