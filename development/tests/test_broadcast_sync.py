"""The synchronisation on a made-up match with a known answer.

Tactical: 22 players moving smoothly for 4000 frames. Broadcast: 2900 frames showing the
players near the action, in other track slots and with 0.3 m of noise. Broadcast frame b
shows tactical frame b + 600 before the cut at b = 1500 and b + 1100 after it (halftime cut
out); frames 800-899 are a replay of a moment 400 frames earlier; 1000-1049 have no homography.
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.broadcast_sync import (  # noqa: E402
    _metres_on_pitch,
    _visible,
    estimate_offsets,
    frame_cost,
    match_samples,
    matching_arrays,
    plot_offsets,
    report,
)
from src.data_loader import Tracking  # noqa: E402

PITCH = [105.0, 68.0]
T, B = 4000, 2900
FIRST_OFFSET, SECOND_OFFSET, CUT = 600, 1100, 1500
REPLAY, REPLAY_SHIFT = range(800, 900), 400
INVALID = range(1000, 1050)
# tactical frames where video 1 loses the ball for 8 s: broadcast 400-599 is live there;
# broadcast 800-999 is the replay (800-899), then live again. And 12 s in 100-399, before the
# broadcast begins (broadcast frame = tactical - 600 < 0): a long gap nothing can fill.
BALL_HOLES = (slice(100, 400), slice(1000, 1200), slice(1400, 1600))
GOALKEEPER = {"margin_m": 5.0, "fallback_depth_m": 5.0}
RESTARTS = {"still_speed_ms": 1.5, "smooth_s": 0.4, "min_still_s": 1.0, "corner_radius_m": 2.5, "spot_radius_m": 1.5,
            "line_margin_m": 1.5, "out_margin_m": 3.0, "max_player_out_m": 5.0,
            "min_duration_s": {"throw-in": 2.0, "free kick": 3.0, "corner": 3.0, "goal kick": 3.0, "penalty": 3.0,
                               "kickoff": 1.0},
            "opponents_back_m": {"throw-in": 1.5, "free kick": 3.0, "corner": 4.0, "goal kick": 6.0, "penalty": 6.0,
                                 "kickoff": 6.0},
            "stoppage_speed_ms": 2.0, "check_s": 2.0, "thrower_behind_m": 0.3, "ball_near_m": 5.0,
            "before_s": 10.0, "duel_m": 2.0, "penalty_check_s": 2.0, "penalty_box_max": 1, "penalty_spread_ms": 1.0,
            "back_on_s": 3.0}
PLAYERS = {"max_gap_s": 10.0, "max_speed_ms": 7.0, "slack_m": 2.0, "switch_penalty_m": 2.0, "from_video_2": True,
           "per_team": 11, "extra_slots": 5, "match_radius_m": 3.0}

CONFIG = {
    "fps": 25,
    "pitch": PITCH,
    "phases": [
        {"name": "first_half", "first": 0, "last": 2099, "half": 1, "graded": True, "clock_start_min": 0},
        {"name": "second_half", "first": 2100, "last": 3999, "half": 2, "graded": True, "clock_start_min": 45},
    ],
    "sync": {"samples": 60, "sequence_s": [0, 1], "min_players": 6, "max_reprojection_error": 3.0,
             "coarse_step": 5, "cap_m": 5.0, "drop_worst": 0.05, "tolerance_s": 0.5},
}


def tracking(players, team, ball=None):
    n, slots = players.shape[:2]
    return Tracking(
        ball=(np.full((n, 2), np.nan) if ball is None else ball).astype(np.float32),
        ball_z=np.full(n, np.nan, np.float32),
        ball_speed=np.full(n, np.nan, np.float32), players=players.astype(np.float32),
        speed=np.full((n, slots), np.nan, np.float32), speed_unit="", team=np.array(team),
        slot_ids=np.array([str(i + 1) for i in range(slots)]), team_names=np.array(["Home", "Away"]),
        home_attacks_first_half="left_to_right",
    )


@pytest.fixture(scope="module")
def match():
    return make_match()


def make_match():
    rng = np.random.default_rng(0)
    pos = np.empty((T, 22, 2))
    pos[0] = rng.uniform([10, 5], [95, 63], (22, 2))
    velocity = np.zeros((22, 2))
    for t in range(1, T):
        velocity = 0.92 * velocity + rng.normal(0, 0.04, (22, 2))
        pos[t] = np.clip(pos[t - 1] + velocity, [1, 1], [104, 67])

    tac = np.full((T, 30, 2), np.nan)  # home in slots 0-10, away in 15-25, the rest unused
    tac[:, 0:11], tac[:, 15:26] = pos[:, :11] / PITCH, pos[:, 11:] / PITCH

    source = np.where(np.arange(B) < CUT, np.arange(B) + FIRST_OFFSET, np.arange(B) + SECOND_OFFSET)
    source[list(REPLAY)] -= REPLAY_SHIFT
    slot_of = np.r_[rng.choice(20, 11, replace=False), 20 + rng.choice(20, 11, replace=False)]
    bro = np.full((B, 40, 2), np.nan)
    for b in range(B):
        p = pos[source[b]]
        seen = np.hypot(*(p - p.mean(axis=0)).T) < 30  # the camera shows the players near the action
        bro[b, slot_of[seen]] = (p[seen] + rng.normal(0, 0.3, (seen.sum(), 2))) / PITCH

    quality = pd.DataFrame({"valid": True, "reprojection_error": 0.5}, index=pd.RangeIndex(B))
    quality.loc[list(INVALID), "valid"] = False

    # the ball sits at player 0's feet; video 1 loses it in BALL_HOLES, video 2 shows it always
    ball = pos[:, 0] / PITCH
    for hole in BALL_HOLES:
        ball[hole] = np.nan
    ball_2 = (pos[source, 0] + rng.normal(0, 0.1, (B, 2))) / PITCH
    return (tracking(tac, ["H"] * 15 + ["A"] * 15, ball), tracking(bro, ["H"] * 20 + ["A"] * 20, ball_2),
            quality)


@pytest.fixture(scope="module")
def synced(match):
    tactical, broadcast, quality = match
    matches = match_samples(tactical, broadcast, quality, CONFIG, 25, 25)
    return estimate_offsets(matches, CONFIG, 25)


def test_frame_cost_ignores_track_numbers(match):
    tactical = matching_arrays(match[0], PITCH)
    rng = np.random.default_rng(1)
    shuffled = {side: slots[:, rng.permutation(slots.shape[1])] for side, slots in tactical.items()}
    points = {side: _visible(slots[100]) for side, slots in _metres_on_pitch(match[0], PITCH).items()}
    candidates = np.arange(0, 500)
    np.testing.assert_allclose(frame_cost(tactical, points, candidates, 5.0),
                               frame_cost(shuffled, points, candidates, 5.0))
    assert frame_cost(tactical, points, np.array([100]), 5.0)[0] == 0


def test_one_offset_per_half_is_found(synced):
    _, table = synced
    per_half = table[table["hypothesis"] == "one offset per half"].set_index("part")
    assert abs(per_half.loc["first_half", "offset"] - FIRST_OFFSET) <= 1
    assert abs(per_half.loc["second_half", "offset"] - SECOND_OFFSET) <= 1
    assert per_half["share"].min() >= 0.85
    assert per_half["slope"].sub(1).abs().max() < 0.002  # no stretching


def test_a_single_offset_does_not_fit_a_cut_halftime(synced):
    _, table = synced
    assert table.loc[table["hypothesis"] == "one offset", "share"].iloc[0] < 0.7


def test_replays_match_an_earlier_moment(synced):
    matches, _ = synced
    replays = matches[matches["broadcast_frame"].isin(REPLAY)]
    assert len(replays) >= 1
    assert (replays["status"] == "earlier").all()
    assert (replays["offset"] - (FIRST_OFFSET - REPLAY_SHIFT)).abs().max() <= 1


def test_frames_without_homography_are_never_used(synced):
    matches, _ = synced
    # neither the frame itself nor the one 1 s later may lie in the invalid stretch
    assert not matches["broadcast_frame"].between(INVALID.start - 25, INVALID.stop - 1).any()


def write_epts(folder, stem, players, team, ball=None):
    """The made-up tracking as files in the layout of the real export (as in tests/conftest.py)."""
    n, slots = players.shape[:2]
    ids = range(1, slots + 1)
    team_id = {"H": "TH", "A": "TA"}
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<main><Metadata>
  <GlobalConfig><FrameRate>25.0</FrameRate><ProviderName>Metrica Sports</ProviderName><ProviderGlobalParameters>
    <ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>
    <ProviderParameter><Name>first_half_end</Name><Value>{n}</Value></ProviderParameter>
  </ProviderGlobalParameters></GlobalConfig>
  <Sessions><Session id="s1"><MatchParameters><Score idLocalTeam="TH" idVisitingTeam="TA">
    <LocalTeamScore/><VisitingTeamScore/></Score></MatchParameters>
    <FieldSize><Width>105</Width><Height>68</Height></FieldSize></Session></Sessions>
  <Teams>
    <Team id="TH"><Name>Home</Name><ProviderTeamsParameters><ProviderParameter>
      <Name>attack_direction_first_half</Name><Value>left_to_right</Value></ProviderParameter></ProviderTeamsParameters></Team>
    <Team id="TA"><Name>Away</Name><ProviderTeamsParameters/></Team>
  </Teams>
  <Players>{"".join(f'<Player id="{i}" teamId="{team_id[team[i - 1]]}"><Name>Track {i}</Name>'
                    f'<ShirtNumber>{i}</ShirtNumber><ProviderPlayerParameters/></Player>' for i in ids)}</Players>
  <Devices><Device id="d1"><Sensors>
    <Sensor id="position"><Name>Position</Name><Channels>
      <Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>
      <Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel></Channels></Sensor>
    <Sensor id="speed"><Name>Speed in kilometers per hour</Name><Channels>
      <Channel id="s"><Name>speed_kph</Name><Unit>kph</Unit></Channel></Channels></Sensor>
  </Sensors></Device></Devices>
  <PlayerChannels>{"".join(f'<PlayerChannel id="p{i}_{c}" playerId="{i}" channelId="{c}"/>'
                           for i in ids for c in "xys")}</PlayerChannels>
</Metadata>
<DataFormatSpecifications><DataFormatSpecification startFrame="1" endFrame="{n}" separator=":">
  <StringRegister name="frameCount"/>
  <SplitRegister separator=";">{"".join(
      '<SplitRegister separator=",">' + "".join(f'<PlayerChannelRef playerChannelId="p{i}_{c}"/>' for c in "xys")
      + "</SplitRegister>" for i in ids)}</SplitRegister>
  <SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>
    <BallChannelRef channelId="s"/></SplitRegister>
</DataFormatSpecification></DataFormatSpecifications></main>"""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.xml").write_text(xml)
    with open(folder / f"{stem}.txt", "w") as f:
        for frame, row in enumerate(players, start=1):
            tracks = ";".join(",," if np.isnan(x) else f"{x:.6f},{y:.6f},5.0" for x, y in row)
            bx, by = (np.nan, np.nan) if ball is None else ball[frame - 1]
            f.write(f"{frame}:{tracks}:" + ("NaN,NaN,NaN" if np.isnan(bx) else f"{bx:.6f},{by:.6f},NaN") + "\n")


def write_match(match, folder):
    """Tracking of both videos and the broadcast homography quality as files; returns the config."""
    tactical, broadcast, quality = match
    write_epts(folder / "data" / "tactical", "TEST_tactical_atd", tactical.players, tactical.team, tactical.ball)
    write_epts(folder / "data" / "broadcast", "TEST_broadcast_atd", broadcast.players, broadcast.team,
               broadcast.ball)
    homography = pd.DataFrame({"frame": range(B), **{f"h{r}{c}": 0.0 for r in range(3) for c in range(3)},
                               "reprojection_error": quality["reprojection_error"],
                               "valid": quality["valid"].astype(int)})
    homography.to_csv(folder / "data" / "broadcast" / "TEST_broadcast_homography.csv", index=False)
    return {**CONFIG, "data": {"dir": str(folder / "data"), "cache_dir": str(folder / "cache"),
                               "game": "TEST", "source": "tactical"}}


def test_ball_data_from_both_videos_end_to_end(match, tmp_path, monkeypatch):
    """Files on disk -> main.load_match: video 2 fills the 8 s holes of video 1 where it shows
    the match live, not where it shows a replay; the analysis interpolates what is left."""
    from src.pipeline import load_match

    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    config = {**write_match(match, tmp_path),
              "broadcast_offsets": {"first_half": FIRST_OFFSET, "second_half": SECOND_OFFSET},
              "players": PLAYERS, "goalkeeper": GOALKEEPER,
              "merge": {"min_gap_s": 5.0, "check_players": True, "max_player_rms_m": 3.0, "min_players": 4},
              "ball": {"max_gap_s": 5.0, "max_speed_ms": 40.0, "velocity_window": 2}}
    ball = load_match(config).ball
    origin = ball["ball_from"]
    assert (origin.loc[1000:1199] == "second").all()  # live broadcast fills the first hole
    assert (origin.loc[1400:1499] == "rejected").all()  # the replay is not taken ...
    assert (ball.loc[1400:1499, "source"] == "interpolated").all()  # ... the 4 s left are interpolated
    assert (origin.loc[1500:1599] == "second").all()  # live again
    # and the positions there are the broadcast's (broadcast frame = tactical frame - 600)
    np.testing.assert_allclose(ball.loc[1000:1199, "x"].to_numpy(), match[1].ball[400:600, 0], atol=1e-5)


def test_main_command_end_to_end(match, tmp_path, monkeypatch):
    """`python main.py` from files on disk: merged ball, long gaps judged, the plot written."""
    import sys

    import yaml

    import main

    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    config = {**write_match(match, tmp_path),
              "broadcast_offsets": {"first_half": FIRST_OFFSET, "second_half": SECOND_OFFSET},
              "players": PLAYERS, "goalkeeper": GOALKEEPER,
              "merge": {"min_gap_s": 5.0, "check_players": True, "max_player_rms_m": 3.0, "min_players": 4},
              "ball": {"max_gap_s": 5.0, "max_speed_ms": 40.0, "velocity_window": 2},
              "dead_ball": {"radius_m": 10.0, "speed_window_s": 0.4, "smooth_s": 1.0, "crowd_min_players": 12,
                            "min_duration_s": 3.0, "top": 25, "slow_speed_ms": 1.5, "line_margin_m": 1.5}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(tmp_path / "config.yaml"),
                                      "--save-plot", str(tmp_path / "ball.png")])
    data, dead, balls = main.main()
    # the 12 s hole before the broadcast begins is the one long gap left; the players run there
    assert list(dead.gaps[["first", "last"]].itertuples(index=False, name=None)) == [(100, 399)]
    assert dead.gaps.loc[0, "verdict"] == "unclear"
    for half in ("first_half", "second_half"):  # one plot per half
        assert (tmp_path / f"ball_{half}.png").stat().st_size > 0
    matplotlib.pyplot.close("all")


def test_command_line_end_to_end(match, tmp_path, monkeypatch):
    """`python -m src.broadcast_sync` from files on disk, as on the real data."""
    import sys

    import yaml

    from src import broadcast_sync

    config = write_match(match, tmp_path)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))

    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["broadcast_sync", "--config", str(tmp_path / "config.yaml"),
                                      "--save-plot", str(tmp_path / "sync.png")])
    matches, table = broadcast_sync.main()
    per_half = table[table["hypothesis"] == "one offset per half"].set_index("part")
    assert abs(per_half.loc["first_half", "offset"] - FIRST_OFFSET) <= 1
    assert abs(per_half.loc["second_half", "offset"] - SECOND_OFFSET) <= 1
    assert (tmp_path / "sync.png").stat().st_size > 0
    assert (tmp_path / "cache" / "TEST_broadcast_tracking.npz").exists()  # parsed once, then cached
    matplotlib.pyplot.close("all")


def test_live_check_tells_live_play_from_a_replay(match):
    from src.broadcast_sync import broadcast_shows_live

    tactical, broadcast, _ = match
    live = np.arange(1000, 1100)  # broadcast frames 400-499: live
    replay = np.arange(1400, 1500)  # broadcast frames 800-899: a replay of 400 frames earlier
    rms_live, players = broadcast_shows_live(tactical, broadcast, live, FIRST_OFFSET, CONFIG)
    rms_replay, _ = broadcast_shows_live(tactical, broadcast, replay, FIRST_OFFSET, CONFIG)
    assert np.nanmax(rms_live) < 1.0  # 0.3 m of noise
    assert np.nanmin(rms_replay) > 3.0
    assert players.min() >= 6
    rms_outside, compared = broadcast_shows_live(tactical, broadcast, np.array([100]), FIRST_OFFSET, CONFIG)
    assert compared[0] == 0 and np.isnan(rms_outside[0])  # frame 100 - 600 is before the broadcast


def test_broadcast_ball_is_moved_onto_the_tactical_timeline():
    from dataclasses import replace

    from src.broadcast_sync import broadcast_ball_halves

    n = 10
    broadcast = replace(tracking(np.full((n, 2, 2), np.nan), ["H", "A"]),
                        ball=np.column_stack([np.arange(n) / 10, np.full(n, 0.5)]).astype(np.float32),
                        ball_speed=np.full(n, 30.0, np.float32))
    quality = pd.DataFrame({"valid": True, "reprojection_error": 0.5}, index=pd.RangeIndex(n))
    quality.loc[3, "valid"] = False  # e.g. a close-up: no ball from this frame
    quality.loc[4, "reprojection_error"] = 100.0
    config = {"phases": [{"name": "first_half", "first": 4, "last": 16, "half": 1, "graded": True}]}
    half = broadcast_ball_halves(broadcast, quality, {"first_half": 5}, config)["first_half"]
    assert list(half.index) == list(range(4, 17))  # tactical frames
    assert list(half["broadcast_frame"]) == [-1] + list(range(10)) + [-1, -1]  # frame - 5, inside 0..9
    np.testing.assert_allclose(half.loc[[5, 6, 14], "ball_x"], [0.0, 0.1, 0.9])
    assert half.loc[[4, 8, 9, 15, 16], "ball_x"].isna().all()  # outside, invalid, or poor homography
    assert (half["phase"] == "first_half").all() and half.attrs["speed_unit"] == ""


def test_report_and_plot(synced):
    matches, table = synced
    text = report(matches, table, CONFIG, 25, 25, B)
    assert "one offset per half" in text and "first_half" in text and "replay?" in text
    ax = plot_offsets(matches, table, CONFIG)
    assert ax.get_xlabel() == "broadcast frame"
    matplotlib.pyplot.close(ax.figure)


def test_restarts_command_end_to_end(match, tmp_path, monkeypatch, capsys):
    """`python -m src.restarts` from files on disk runs through and lists what it finds."""
    import sys

    import yaml

    from src import restarts

    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    config = {**write_match(match, tmp_path),
              "broadcast_offsets": {"first_half": FIRST_OFFSET, "second_half": SECOND_OFFSET},
              "players": PLAYERS, "goalkeeper": GOALKEEPER,
              "merge": {"min_gap_s": 5.0, "check_players": True, "max_player_rms_m": 3.0, "min_players": 4},
              "ball": {"max_gap_s": 5.0, "max_speed_ms": 40.0, "velocity_window": 2},
              "restarts": RESTARTS, "dead_ball": {"speed_window_s": 0.4}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["restarts", "--config", str(tmp_path / "config.yaml")])
    table = restarts.main()
    assert list(table.columns[:4]) == ["id", "half", "match", "video"]
    assert "times the ball lies still in the halves" in capsys.readouterr().out


def test_smart_tagging_command_end_to_end(match, tmp_path, monkeypatch, capsys):
    """`python -m src.smart_tagging` from files on disk: the tags read, checked, listed."""
    import sys

    import yaml

    from src import smart_tagging

    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    config = {**write_match(match, tmp_path),
              "broadcast_offsets": {"first_half": FIRST_OFFSET, "second_half": SECOND_OFFSET},
              "players": PLAYERS, "goalkeeper": GOALKEEPER,
              "merge": {"min_gap_s": 5.0, "check_players": True, "max_player_rms_m": 3.0, "min_players": 4},
              "ball": {"max_gap_s": 5.0, "max_speed_ms": 40.0, "velocity_window": 2},
              "restarts": RESTARTS, "dead_ball": {"speed_window_s": 0.4},
              "smart_tagging": {"margin_s": 3.0, "smooth_s": 1.0, "line_band_m": 2.0, "confirm_pct": 0.9,
                                "contradict_pct": 0.5}}
    (tmp_path / "data" / "tactical" / "TEST_tactical_smart_tagging.xml").write_text(
        "<file><ALL_INSTANCES><instance><ID>1</ID><start>2.0</start><end>4.0</end><code>SET PIECES</code>"
        "<label><group>Type</group><text>Throw-in</text></label></instance></ALL_INSTANCES></file>")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["smart_tagging", "--config", str(tmp_path / "config.yaml")])
    table = smart_tagging.main()
    assert "1 smart-tagging instances" in capsys.readouterr().out
    assert len(table) <= 1 and list(table.columns[:2]) == ["id", "tag"]


def test_all_situations_in_one_list(match, tmp_path, monkeypatch):
    """`python -m src.visualization --all`: restarts, long gaps, crowds and tags in one list, in
    time order - here the one long gap of the ball (frames 100-399), without a tag file."""
    from src.pipeline import all_situations, load_match
    from src.visualization import PitchPlot

    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    config = {**write_match(match, tmp_path),
              "broadcast_offsets": {"first_half": FIRST_OFFSET, "second_half": SECOND_OFFSET},
              "players": PLAYERS, "goalkeeper": GOALKEEPER,
              "merge": {"min_gap_s": 5.0, "check_players": True, "max_player_rms_m": 3.0, "min_players": 4},
              "ball": {"max_gap_s": 5.0, "max_speed_ms": 40.0, "velocity_window": 2},
              "restarts": RESTARTS,
              "dead_ball": {"radius_m": 10.0, "speed_window_s": 0.4, "smooth_s": 1.0, "crowd_min_players": 12,
                            "min_duration_s": 3.0, "top": 25, "slow_speed_ms": 1.5, "line_margin_m": 1.5},
              "smart_tagging": {"margin_s": 3.0, "smooth_s": 1.0, "line_band_m": 2.0, "confirm_pct": 0.9,
                                "contradict_pct": 0.5}}
    data = load_match(config)
    table = all_situations(data, config)
    assert list(table.columns) == ["id", "source", "first", "last", "duration_s", "match", "summary"]
    assert list(table["id"]) == list(range(1, len(table) + 1)) and table["first"].is_monotonic_increasing
    gap = table[table["source"] == "gap"].iloc[0]
    assert (gap["first"], gap["last"]) == (100, 399) and gap["summary"].startswith("gap: unclear")
    fig, ax = matplotlib.pyplot.subplots()
    plot = PitchPlot(ax, data.tracking, config, episodes=table, segment_name="situation")
    plot.draw(200)
    assert "situation" in plot.caption.get_text() and "gap: unclear" in plot.caption.get_text()
    matplotlib.pyplot.close(fig)


def test_segments_command_end_to_end(match, tmp_path, monkeypatch, capsys):
    """`python -m src.segments` from files on disk: the events, the possession after them, the live stretches."""
    import sys

    import yaml

    from src import segments

    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    config = {**write_match(match, tmp_path),
              "broadcast_offsets": {"first_half": FIRST_OFFSET, "second_half": SECOND_OFFSET},
              "players": PLAYERS, "goalkeeper": GOALKEEPER,
              "merge": {"min_gap_s": 5.0, "check_players": True, "max_player_rms_m": 3.0, "min_players": 4},
              "ball": {"max_gap_s": 5.0, "max_speed_ms": 40.0, "velocity_window": 2},
              "restarts": RESTARTS, "segments": {"merge_s": 1.0, "after_s": 3.0, "control_m": 2.0, "before_s": 3.0, "catch_m": 2.0, "catch_look_s": 0.24, "catch_pass_min_ms": 5.0, "catch_back_m": 60.0,
                                         "max_lookback_s": 60.0, "out_margin_m": 1.5, "out_min_s": 0.4, "rejoin_s": 3.0,
                                         "max_out_gap_s": 15.0, "slow_speed_ms": 1.5, "reaction_s": 1.0,
                                         "smooth_s": 1.0},
              "dead_ball": {"radius_m": 10.0, "speed_window_s": 0.4, "smooth_s": 1.0, "crowd_min_players": 12,
                            "min_duration_s": 3.0, "top": 25, "slow_speed_ms": 1.5, "line_margin_m": 1.5}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["segments", "--config", str(tmp_path / "config.yaml")])
    events, live = segments.main()
    assert "dead-ball events in the halves" in capsys.readouterr().out
    assert list(live.columns) == ["half", "first", "last", "team", "end_team", "duration_s"]


def test_the_submission_is_stefans_chain_and_restart_based_runs_too(match, tmp_path, monkeypatch):
    """src.baseline.predict with baseline.dead = gap is exactly preprocess -> features -> model ->
    postprocess (run_pipeline.py's chain); restart and union run on the same data."""
    from src.baseline import predict
    from src.data_loader import load_config
    from src.feature_extraction import extract_features
    from src.pipeline import load_match
    from src.possession_model import PossessionModel
    from src.postprocessing import postprocess
    from src.preprocessing import preprocess

    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    base = load_config()
    config = {**base, **write_match(match, tmp_path), "smart_tagging": {**base["smart_tagging"], "enabled": False}}
    data = load_match(config, use_video_2=False)
    chain = postprocess(PossessionModel(config).predict(extract_features(preprocess(data, config), config)), config)
    assert (predict(data, {**config, "baseline": {**config["baseline"], "dead": "gap"}}) == chain).all()
    for logic in ("restart", "union", "evidence"):
        labels = predict(data, {**config, "baseline": {**config["baseline"], "dead": logic}})
        assert len(labels) == len(chain) and set(labels) <= set("HADX")
