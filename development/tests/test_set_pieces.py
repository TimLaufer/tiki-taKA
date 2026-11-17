"""Smart Tagging parsing on a synthetic SportsCode file in the layout of the export."""
import pandas as pd
import pytest

from src.smart_tagging import load_set_pieces, parse_set_pieces

XML = """<?xml version="1.0" ?>
<file><ALL_INSTANCES>
  <instance><ID>1</ID><start>10.0</start><end>45.0</end><code>SET PIECES</code>
    <label><group>Team</group><text>home fc</text></label>
    <label><group>Type</group><text>Corner Kick</text></label></instance>
  <instance><ID>2</ID><start>20.04</start><end>55.04</end><code>Set Pieces</code>
    <label><group>Team</group><text>N/A</text></label>
    <label><group>Type</group><text>Throw-in</text></label></instance>
  <instance><ID>3</ID><start>30</start><end>40</end><code>BUILD UP</code>
    <label><group>Team</group><text>Away FC</text></label></instance>
  <instance><ID>4</ID><start>5.0</start><end>10.0</end><code>SET PIECES</code>
    <label><group>Team</group><text>Away FC</text></label>
    <label><group>Type</group><text>Kick Off (Start)</text></label></instance>
</ALL_INSTANCES></file>
"""


def write(tmp_path, text=XML):
    path = tmp_path / "TEST_tactical_smart_tagging.xml"
    path.write_text(text)
    return path


def test_only_set_pieces_on_the_video_timeline_sorted_with_teams(tmp_path):
    table = parse_set_pieces(write(tmp_path), 25, ("Home FC", "Away FC"))
    assert table.to_dict("records") == [
        {"first": 125, "last": 250, "type": "Kick Off (Start)", "team": "A"},
        {"first": 250, "last": 1125, "type": "Corner Kick", "team": "H"},  # team names matched case-insensitively
        {"first": 501, "last": 1376, "type": "Throw-in", "team": ""},  # "N/A" is no team
    ]


def test_an_interval_ending_before_it_starts_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="invalid interval"):
        parse_set_pieces(write(tmp_path, XML.replace("<end>45.0</end>", "<end>9.0</end>")), 25)


def test_missing_or_disabled_tagging_weakens_the_evidence_instead_of_failing(epts_config, tmp_path):
    with pytest.warns(UserWarning, match="Smart Tagging not found"):
        table, path = load_set_pieces(epts_config)
    assert table.empty and path is None
    write(tmp_path / "data" / "tactical")
    table, path = load_set_pieces(epts_config, ("Home FC", "Away FC"))
    assert len(table) == 3 and path.name == "TEST_tactical_smart_tagging.xml"
    epts_config["smart_tagging"] = {"enabled": False}
    table, path = load_set_pieces(epts_config)
    assert isinstance(table, pd.DataFrame) and table.empty and path is None
