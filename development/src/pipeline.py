"""The pipeline as functions, to call from main.py, the viewer, a notebook or anywhere else.

    from src.data_loader import load_config
    from src.pipeline import load_match, dead_ball_analysis
    config = load_config()
    data = load_match(config)                 # tracking, DataFrames per phase, the merged ball
    dead = dead_ball_analysis(data, config)   # crowding, the crowd episodes, the long gaps judged

What load_match builds:

    data.tracking               whole match as arrays, index = video frame; the players cleaned:
                                lost ones bridged, missing ones from video 2 (src/player_cleaning.py)
    data.raw_tracking           the tracking as delivered
    data.player_source          per frame and slot where each player position comes from (SOURCE_NAMES)
    data.df                     whole match as one DataFrame, one row per frame, column "phase"
    data.parts["first_half"]    one phase as arrays: .players (frames, 46, 2), .speed, .ball, ...
    data.dfs["first_half"]      one phase as DataFrame, the index is still the video frame
    data.ball                   THE BALL DATA: video 1, plus video 2 in the gaps of video 1 longer
                                than merge.min_gap_s (players checked), then analysed: false
                                detections dropped, gaps up to ball.max_gap_s interpolated,
                                velocity (src/ball_analysis.py). Column ball_from: which video.
    data.tactical_ball          the tactical ball alone, analysed the same way: its long gaps are
                                the stoppage evidence of the prediction (src/stoppages.py)
    data.video_2                the broadcast tracking and its ball per half (None without it)
    data.merged[half]           the merged ball per half, before the analysis

and dead_ball_analysis (src/dead_ball.py):

    dead.indicators             per frame: crowd at the densest spot, players' median speed, ...
    dead.episodes               the crowd episodes, by id
    dead.gaps                   the gaps of data.ball still longer than ball.max_gap_s, in time
                                order, with the evidence and the verdict dead / unclear
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.ball_analysis import (
    analyze_ball,
    ball_dataframe,
    gap_stats,
    gaps_to_fill,
    insert_halves,
    merge_ball,
    phase_table,
    speed_check,
)
from src.broadcast_sync import broadcast_ball_halves, broadcast_shows_live
from src.data_loader import PROJECT_ROOT, Tracking, atd_paths, load_homography_quality, load_tracking
from src.dead_ball import episodes, indicators, long_gaps
from src.goalkeeper import keeper_slots, keeper_table, place_keepers
from src.inspect_data import to_dataframe
from src.phases import overview, split_dataframe, split_tracking
from src.player_cleaning import KEEPER_HELD, clean_players, player_table
from src.progress import Steps, say

# The halves from kickoff to whistle. second_half_pre_kickoff is left out: no play, all D,
# and the broadcast offsets only hold inside the halves.
HALVES = ["first_half", "second_half"]
VIDEO_1, VIDEO_2, BOTH, MERGED = "video 1 (tactical)", "video 2 (broadcast)", "video 1 + 2", "merged"
DEAD_BALLS = "dead balls"


@dataclass
class Video2:
    tracking: Tracking  # the broadcast tracking, on its own timeline
    offsets: dict[str, int]  # per half: tactical frame = broadcast frame + offset
    balls: dict[str, pd.DataFrame]  # per half: the broadcast ball on the tactical timeline


@dataclass
class MatchData:
    tracking: Tracking  # whole match, players cleaned (src/player_cleaning.py), index = video frame
    df: pd.DataFrame  # whole match, one row per video frame
    parts: dict[str, Tracking]  # per phase: the arrays (views into tracking, no copy)
    dfs: dict[str, pd.DataFrame]  # per phase: the rows of df
    ball: pd.DataFrame  # THE ball data: merged, analysed, one row per video frame
    video_2: Video2 | None = None  # the broadcast, if there is one
    merged: dict[str, pd.DataFrame] | None = None  # per half: the merged ball before the analysis
    raw_tracking: Tracking | None = None  # the tracking as delivered
    player_source: np.ndarray | None = None  # (frames, slots): where each player position comes from
    tactical_ball: pd.DataFrame | None = None  # the tactical ball alone, analysed: its gaps judge stoppages


@dataclass
class DeadBall:
    indicators: pd.DataFrame  # per frame: crowd, crowd_share, crowd_x/_y, players, speed_ms
    episodes: pd.DataFrame  # the crowd episodes, by id
    gaps: pd.DataFrame  # the long gaps of data.ball, in time order, with verdict and reasons


def ball_halves(dfs: dict[str, pd.DataFrame], names: list[str] = HALVES) -> dict[str, pd.DataFrame]:
    """The ball columns (phase, ball_x, ball_y, ball_s) of each half, as delivered."""
    return {name: ball_dataframe(dfs[name]) for name in names}


def load_video_2(config: dict, names: list[str] = HALVES, tactical: Tracking | None = None) -> Video2 | None:
    """Automatically align broadcast tracking; None when disabled or assets are absent.

    An unreliable computed alignment is an error, never a silent guessed offset.
    """
    if not config.get("sync", {}).get("enabled", True):
        say("broadcast synchronization disabled: video 2 is left out")
        return None
    try:
        broadcast = load_tracking(config, "broadcast")
        quality = load_homography_quality(config, "broadcast")
    except FileNotFoundError as error:
        say(f"no broadcast data, video 2 is left out ({error})")
        return None
    from src.synchronization import synchronize
    offsets = synchronize(tactical if tactical is not None else load_tracking(config, "tactical"),
                          broadcast, quality, config)
    offsets = {name: offsets[name] for name in names}
    valid = (quality["valid"].to_numpy() &
             (quality["reprojection_error"].to_numpy() <= config["sync"]["max_reprojection_error"]))
    # The same quality gate applies to player completion, not only to ball merging.
    broadcast = replace(broadcast, players=np.where(valid[:, None, None], broadcast.players, np.nan))
    return Video2(tracking=broadcast, offsets=offsets,
                  balls=broadcast_ball_halves(broadcast, quality, offsets, config))


def merge_videos(tracking: Tracking, dfs: dict[str, pd.DataFrame], video_2: Video2,
                 config: dict) -> dict[str, pd.DataFrame]:
    """Per half: the ball of video 1, plus the ball of video 2 in the gaps of video 1 longer
    than merge.min_gap_s. With merge.check_players, a broadcast frame is only used where its
    players match the tactical ones at that moment, which keeps replays out."""
    settings = config["merge"]
    min_gap = round(settings["min_gap_s"] * config["fps"])
    merged = {}
    for half, first in ball_halves(dfs, list(video_2.balls)).items():
        second = video_2.balls[half]
        usable = None
        if settings["check_players"]:
            wanted = gaps_to_fill(first, second, min_gap)
            rms, players = broadcast_shows_live(tracking, video_2.tracking, first.index.to_numpy()[wanted],
                                                video_2.offsets[half], config)
            usable = np.zeros(len(first), dtype=bool)
            usable[wanted] = (players >= settings["min_players"]) & (rms <= settings["max_player_rms_m"])
        merged[half] = merge_ball(first, second, min_gap, usable)
    return merged


def _ball_only(tracking: Tracking) -> Tracking:
    """The tracking without players: to_dataframe of it has just time, phase and the ball."""
    return replace(tracking, players=tracking.players[:, :0], speed=tracking.speed[:, :0],
                   team=tracking.team[:0], slot_ids=tracking.slot_ids[:0])


def load_match(config: dict, steps: Steps | None = None, use_video_2: bool = True,
               raw: Tracking | None = None, keepers: dict[str, int] | None = None) -> MatchData:
    """Load, clean the players, merge the ball of both videos and analyse it, place the
    goalkeepers, convert and split; each step shows up in the terminal with its duration.

    raw, keepers: tracking already loaded (e.g. another game, src.external) and its marked
    goalkeeper slots; by default the configured tracking and its metadata."""
    steps = steps or Steps(7)
    with steps("Loading the tracking"):
        if raw is None:
            raw = load_tracking(config)
            keepers = keeper_slots(atd_paths(config, config["data"]["source"])[0], raw)
        keepers = keepers or {}
    with steps("Video 2: the broadcast tracking and its ball, moved onto the tactical timeline"):
        video_2 = load_video_2(config, tactical=raw) if use_video_2 else None
    players = config["players"]
    with steps(f"Cleaning the players: lost ones bridged up to {players['max_gap_s']:g} s"
               + (", missing ones from video 2" if video_2 is not None and players.get("from_video_2") else "")):
        tracking, player_source = clean_players(raw, video_2, config, alone=tuple(keepers.values()))
    with steps(f"Merging the ball: video 2 in the gaps of video 1 longer than {config['merge']['min_gap_s']} s"):
        ball_frames = to_dataframe(_ball_only(raw), config)
        # the check whether the broadcast shows the moment live uses the players as delivered:
        # players taken from video 2 would match the broadcast by construction
        merged = (merge_videos(raw, split_dataframe(ball_frames, config), video_2, config)
                  if video_2 is not None else None)
        if merged is None:
            say("nothing to merge: the ball data comes from video 1 alone")
    with steps(f"Ball analysis of the merged ball: false detections, gaps up to {config['ball']['max_gap_s']} s, "
               "motion"):
        ball = analyze_ball(insert_halves(ball_dataframe(ball_frames), merged), config)
        # The gaps of the tactical ball itself are the stoppage evidence; the broadcast
        # ball may partly fill a stoppage (a keeper placing the ball) and must not hide it.
        tactical_ball = ball if merged is None else analyze_ball(ball_dataframe(ball_frames), config)
    with steps("Goalkeepers: the keeper track or the deepest player, else held where last seen - one per team, always"):
        tracking, player_source = place_keepers(tracking, player_source, config, keepers)
        say("keeper tracks in the metadata: "
            + (", ".join(f"{side} slot {raw.slot_ids[i]}" for side, i in keepers.items()) or "none"))
    with steps("One DataFrame row per video frame, split by match phase"):
        df = to_dataframe(tracking, config)
        parts, dfs = split_tracking(tracking, config), split_dataframe(df, config)
    return MatchData(tracking=tracking, df=df, parts=parts, dfs=dfs, ball=ball, video_2=video_2, merged=merged,
                     raw_tracking=raw, player_source=player_source, tactical_ball=tactical_ball)


def measured(data: MatchData) -> Tracking:
    """data.tracking without the keepers held at their last position: no measurements, so the
    indicators (crowding, speed, block shifts) leave them out - a keeper standing still where he
    was last seen would slow the players' median speed down."""
    if data.player_source is None:
        return data.tracking
    players = data.tracking.players.copy()
    players[data.player_source == KEEPER_HELD] = np.nan
    return replace(data.tracking, players=players)


def dead_ball_analysis(data: MatchData, config: dict) -> DeadBall:
    """Crowding and speed of the players, the crowd episodes, and the long gaps of the ball
    data judged dead / unclear (src/dead_ball.py)."""
    tracking = measured(data)
    ind = indicators(tracking, config)
    crowd_episodes = episodes(ind, tracking, config)
    return DeadBall(indicators=ind, episodes=crowd_episodes,
                    gaps=long_gaps(data.ball, ind, crowd_episodes, config))


def all_situations(data: MatchData, config: dict, dead: DeadBall | None = None) -> pd.DataFrame:
    """Everything we find that may interrupt play, in one list in time order (columns id, source,
    first, last, duration_s, match, summary): the restarts (src/restarts.py), the long gaps of the
    ball judged dead or unclear and the crowd episodes (src/dead_ball.py), and Metrica's set-piece
    tags checked against the tracking (src/smart_tagging.py) if the file is there.
    dead: the dead-ball analysis if already done."""
    from src.restarts import RESTART, candidates
    from src.smart_tagging import check_set_pieces, load_tags, signatures, smart_tagging_path

    dead = dead if dead is not None else dead_ball_analysis(data, config)
    tracking = measured(data)
    still = candidates(tracking, data.ball, config)
    crowd = dead.episodes.copy()
    crowd["summary"] = [f"{zone}, up to {peak} players" for zone, peak in zip(crowd["zone"], crowd["peak"])]
    parts = [("restart", still[still["verdict"] == RESTART]), ("gap", dead.gaps), ("crowd", crowd)]
    try:
        tags = load_tags(smart_tagging_path(config), config, data.tracking.team_names)
        parts.append(("tag", check_set_pieces(tags, still, signatures(tracking, data.ball, config), config)))
    except FileNotFoundError:
        say("no smart-tagging file: the list goes without Metrica's tags")
    columns = ["first", "last", "duration_s", "match", "summary"]
    table = pd.concat([part[columns].assign(source=name) for name, part in parts if len(part)]
                      or [pd.DataFrame(columns=columns + ["source"])], ignore_index=True)
    table = table.sort_values(["first", "last"], kind="stable", ignore_index=True)
    table["summary"] = [f"{source}: {summary}" for source, summary in zip(table["source"], table["summary"])]
    table.insert(0, "id", np.arange(1, len(table) + 1))
    return table[["id", "source"] + columns]


def verdict_halves(data: MatchData, gaps: pd.DataFrame, names: list[str] = HALVES) -> dict[str, pd.DataFrame]:
    """Per half, for the plot: the analysed ball (x, y as ball_x, ball_y) and, per frame, the
    verdict of the long gap it lies in ("" outside the long gaps)."""
    verdict = np.full(len(data.ball), "", dtype=object)
    frames = data.ball.index.to_numpy()
    for gap in gaps.itertuples():
        verdict[np.searchsorted(frames, gap.first) : np.searchsorted(frames, gap.last, "right")] = gap.verdict
    verdict = pd.Series(verdict, index=data.ball.index)
    halves = {}
    for name in names:
        part = data.ball.loc[data.dfs[name].index]
        halves[name] = pd.DataFrame({"phase": part["phase"], "ball_x": part["x"], "ball_y": part["y"],
                                     "verdict": verdict.loc[part.index]})
    return halves


def ball_sources(data: MatchData) -> dict[str, dict[str, pd.DataFrame]]:
    """For comparing: the ball per half from video 1, video 2, video 1 + 2 (every gap) and merged."""
    balls = {VIDEO_1: ball_halves(data.dfs)}
    if data.video_2 is not None:
        balls[VIDEO_2] = data.video_2.balls
        balls[BOTH] = {half: merge_ball(ball, data.video_2.balls[half]) for half, ball in balls[VIDEO_1].items()}
    if data.merged is not None:
        balls[MERGED] = data.merged
    return balls


def load_public_gt(config: dict, n_frames: int) -> np.ndarray | None:
    """The public 5-minute ground truth as labels per frame, or None if it is not there.
    For understanding only: tuning on it is against the rules (RULES.md)."""
    from src.visualization import read_labels

    path = PROJECT_ROOT / config["public_gt"] if config.get("public_gt") else None
    if path is None or not path.is_file():
        say(f"no public ground truth at {path}")
        return None
    return read_labels(path, n_frames)


def _player_table(data: MatchData, config: dict) -> pd.DataFrame:
    raw = data.raw_tracking if data.raw_tracking is not None else data.tracking
    source = data.player_source
    if source is None:  # nothing cleaned: every position is observed
        source = (~np.isnan(data.tracking.players[:, :, 0])).astype(np.uint8)
    return player_table(raw, data.tracking, source, config)


def describe(data: MatchData, config: dict) -> str:
    shapes = pd.DataFrame(
        {
            name: {
                "players": str(part.players.shape),
                "speed": str(part.speed.shape),
                "ball": str(part.ball.shape),
                "ball_speed": str(part.ball_speed.shape),
                "DataFrame": f"{len(data.dfs[name])} rows x {data.dfs[name].shape[1]} cols",
            }
            for name, part in data.parts.items()
        }
    ).T
    unit = data.tracking.speed_unit or "-"
    home, away = data.tracking.team_names
    return "\n".join([
        "Match phases:",
        overview(data.tracking, config).to_string(),
        "",
        "Per phase: arrays in data.parts[<phase>], DataFrame in data.dfs[<phase>]:",
        shapes.to_string(),
        "",
        f"players: x, y per track (0-1) | speed: per track ({unit}) | ball: x, y (0-1) | ball_speed ({unit})",
        f"{len(data.tracking.slot_ids)} tracks, H = {home}, A = {away}",
        "",
        "The players, cleaned (data.tracking; as delivered: data.raw_tracking), on the pitch per frame "
        "and where the positions come from:",
        _player_table(data, config).to_string(),
        "",
        "The goalkeepers (slots GKH, GKA), where their positions come from:",
        (keeper_table(data.tracking, data.player_source, config).to_string(index=False)
         if data.player_source is not None else "-"),
        "",
        "The ball data, data.ball (merged, then analysed):",
        phase_table(data.ball).to_string(),
        speed_check(data.ball),
    ])


def gap_tables(balls: dict[str, dict[str, pd.DataFrame]], config: dict) -> str:
    """Per half one table: the gap statistics of every source side by side."""
    parts = []
    for half in balls[VIDEO_1]:
        table = pd.DataFrame({source: gap_stats(balls[source][half], config) for source in balls})
        parts += [f"{half}:", table.to_string(), ""]
    return "\n".join(parts)


def merge_table(merged: dict[str, pd.DataFrame], config: dict) -> str:
    """Per half: where the merged ball comes from, in frames."""
    names = {"first": "from video 1", "second": "from video 2",
             "rejected": "video 2 rejected (players do not match: replay?)", "none": "no ball"}
    table = pd.DataFrame({half: ball["ball_from"].value_counts().reindex(list(names), fill_value=0)
                          for half, ball in merged.items()})
    table.index = [names[i] for i in table.index]
    return (f"Merged ball (video 2 only in gaps of video 1 longer than {config['merge']['min_gap_s']} s), "
            f"in frames:\n{table.to_string()}")
