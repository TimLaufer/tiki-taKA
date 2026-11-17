"""Pitch view of the tracking: interactive viewer, single frame or clip.

    python -m src.visualization                        # viewer, opens at the first kickoff
    python -m src.visualization --frame 20000          # viewer at frame 20000
    python -m src.visualization --frame 20000 --out frame.png
    python -m src.visualization --frame 20000 --to 20250 --out clip.gif   # .mp4 needs ffmpeg
    python -m src.visualization --labels my_submission.csv --labels gt_public.csv
    python -m src.visualization --episodes             # the crowded moments of src.dead_ball, by id
    python -m src.visualization --episode 7            # ... starting at episode 7
    python -m src.visualization --gaps                 # the long gaps of the ball data, judged dead/unclear
    python -m src.visualization --gap 3                # ... starting at gap 3
    python -m src.visualization --blocks               # the block shifts of src.block_shift, with arrows
    python -m src.visualization --block 12             # ... starting at block shift 12
    python -m src.visualization --restarts             # the restarts of src.restarts: kind, team, taker
    python -m src.visualization --restart 5            # ... starting at restart 5
    python -m src.visualization --restarts --rejected  # the still balls judged in play, with the reasons
    python -m src.visualization --tags                 # Metrica's set-piece tags, checked by src.smart_tagging
    python -m src.visualization --tag 4                # ... starting at tag 4
    python -m src.visualization --all                  # all of them in one list: restarts, gaps, crowds, tags
    python -m src.visualization --segments             # the dead-ball events, with who has the ball after each
    python -m src.visualization --dead                 # Stefan's and Tim's dead balls with their kinds, n / p
    python -m src.visualization --situation 30         # ... starting at situation 30

Viewer: type a frame number into the box at the bottom right and press Enter,
or drag the slider. Keys: left/right 1 frame, down/up 1 s, pagedown/pageup 10 s,
space play/pause; with episodes, gaps, blocks or restarts also n/p for the next/previous one.
Without --labels the viewer shows ours (outputs/<game>_baseline.csv, from python -m
src.baseline) and the public ground truth (its five minutes) above the pitch and in the box
at the top right: green frame = the same, red = they differ, grey = no truth here.
--labels takes any frame,label CSVs instead; the first counts as ours, the second as truth.
--episodes and --gaps list them with their ids in the terminal and draw a circle of
dead_ball.radius_m around the densest spot, red where dead_ball.crowd_min_players
stand in it. --gaps runs the whole pipeline (src/pipeline.py) and shows its ball
data, the merged ball, instead of the tactical ball alone. --blocks does the same and
draws each team's common shift c over block.window_s as an arrow from the team's centre:
bold where the team shifts as a block, faint elsewhere. --restarts also runs the pipeline
and opens each restart where the ball starts to lie still; the line above the pitch says
the kind, the team and the taker.

The y axis points down because the 0-1 coordinates start top-left, so the
picture should match the tactical camera. Check that once against the video.
The plots show real match data: do not commit or share them.
"""
from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, patches
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, TextBox

from src.data_loader import DEFAULT_CONFIG, PROJECT_ROOT, Tracking, load_config, load_tracking
from src.phases import first_graded_frame, match_clock, phase_at, video_time
from src.progress import say

COLORS = {"H": "#1d4ed8", "A": "#dc2626", "ball": "#facc15", "pitch": "#2f6b3a", "line": "#e8f0e9",
          "crowd": "#9ca3af", "episode": "#ef4444"}
LABEL_COLORS = {"H": COLORS["H"], "A": COLORS["A"], "D": "#4b5563", "X": "#6b7280"}
TRAIL = 25  # frames of ball trail
CAPTION_WIDTH = 80  # characters per caption line: longer lines would run under the legend
# matplotlib's own bindings for keys the viewer uses (back, forward, pan)
FREED_KEYS = {"keymap.back": ("left",), "keymap.forward": ("right",), "keymap.pan": ("p",)}


def draw_pitch(ax, length: float, width: float) -> None:
    """Standard pitch markings in metres, origin top-left."""
    line = dict(color=COLORS["line"], lw=1.2, zorder=1)
    mid = width / 2
    ax.set_facecolor(COLORS["pitch"])
    ax.add_patch(patches.Rectangle((0, 0), length, width, fill=False, **line))
    ax.plot([length / 2] * 2, [0, width], **line)
    ax.add_patch(patches.Circle((length / 2, mid), 9.15, fill=False, **line))
    arc = np.degrees(np.arccos(5.5 / 9.15))  # where the penalty arc leaves the box
    for goal_line, inward in ((0, 1), (length, -1)):
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):
            x = goal_line if inward > 0 else goal_line - depth
            ax.add_patch(patches.Rectangle((x, mid - half), depth, 2 * half, fill=False, **line))
        x = goal_line - 2 if inward > 0 else goal_line
        ax.add_patch(patches.Rectangle((x, mid - 3.66), 2, 7.32, fill=False, **line))
        spot = goal_line + inward * 11
        start = -arc if inward > 0 else 180 - arc
        ax.add_patch(patches.Arc((spot, mid), 18.3, 18.3, theta1=start, theta2=start + 2 * arc, **line))
        ax.plot([spot], [mid], marker="o", ms=2, color=COLORS["line"], zorder=1)
    ax.plot([length / 2], [mid], marker="o", ms=2, color=COLORS["line"], zorder=1)
    ax.set_xlim(-4, length + 4)
    ax.set_ylim(width + 3, -3)  # y points down
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def read_labels(path, n_frames: int) -> np.ndarray:
    labels = np.full(n_frames, "", dtype="<U1")
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        frame_col, label_col = header.index("frame"), header.index("label")
        for row in reader:
            frame = int(row[frame_col])
            if 0 <= frame < n_frames:
                labels[frame] = row[label_col].strip().upper()
    return labels


class PitchPlot:
    """Draws one frame of the tracking; call draw(frame) to move to another.

    crowd: optionally the indicators of src.dead_ball, for the circle around the densest spot.
    episodes: optionally a table of stretches (columns id, first, last, duration_s, match and
    either summary or zone and peak) - the crowd episodes or the long gaps - for the line in
    the title; segment_name is what they are called there.
    """

    def __init__(self, ax, tracking: Tracking, config: dict, label_sets: dict | None = None,
                 crowd: pd.DataFrame | None = None, episodes: pd.DataFrame | None = None,
                 segment_name: str = "episode", blocks: pd.DataFrame | None = None,
                 player_source: np.ndarray | None = None, dead_kinds: dict | None = None):
        self.segment_name = segment_name
        self.player_source = player_source  # src.player_cleaning: bridged players faint, video 2 hollow
        self.blocks = blocks  # the indicators of src.block_shift, for the shift arrows
        self.arrows = {}
        if blocks is not None:
            self.arrows = {side: ax.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1,
                                           color=COLORS[side], width=0.006, zorder=6, visible=False)
                           for side in "HA"}
        self.t = tracking
        self.scale = np.asarray(config["pitch"], dtype=float)
        self.fps = config["fps"]
        self.config = config
        self.label_sets = label_sets or {}
        self.dead_kinds = dead_kinds or {}  # name -> per frame the kind of its dead ball, "" in live play
        draw_pitch(ax, *self.scale)

        self.episodes = {} if episodes is None else {int(e.id): e for e in episodes.itertuples()}
        self.episode_of = np.zeros(tracking.n_frames, dtype=int)  # episode id per frame, 0 = none
        for episode in self.episodes.values():
            self.episode_of[episode.first : episode.last + 1] = episode.id
        self.circle = None
        if crowd is not None:
            self.radius = float(config["dead_ball"]["radius_m"])
            self.crowd_min = int(config["dead_ball"]["crowd_min_players"])
            self.crowd_xy = crowd[["crowd_x", "crowd_y"]].to_numpy(dtype=float)
            self.crowd_n = crowd["crowd"].to_numpy()
            self.circle = patches.Circle((0, 0), self.radius, fill=False, ls="--", zorder=2, visible=False)
            ax.add_patch(self.circle)
            self.circle_label = ax.text(0, 0, "", fontsize=8, color="white", ha="center", va="bottom", zorder=6)

        self.dots = {
            side: ax.scatter([], [], s=150, c=COLORS[side], edgecolors="white", linewidths=1, zorder=3)
            for side in "HA"
        }
        self.is_keeper = np.char.startswith(tracking.slot_ids.astype(str), "GK")  # src.goalkeeper's slots
        self.keepers = {
            side: ax.scatter([], [], s=190, marker="s", c=COLORS[side], edgecolors="black", linewidths=1.5, zorder=3)
            for side in "HA"
        }
        self.slot_labels = [
            ax.text(0, 0, "GK" if keeper else sid, fontsize=6, color="white", ha="center", va="center", zorder=4,
                    visible=False)
            for sid, keeper in zip(tracking.slot_ids, self.is_keeper)
        ]
        (self.trail,) = ax.plot([], [], color=COLORS["ball"], lw=1.5, alpha=0.7, zorder=4)
        self.ball = ax.scatter([], [], s=70, c=COLORS["ball"], edgecolors="black", linewidths=1, zorder=5)
        self.caption = ax.set_title("", loc="left", fontsize=10, family="monospace")
        # our label and, where there is one, the truth, big in the corner: green frame = same, red = differs
        self.badge = ax.text(0.995, 0.985, "", transform=ax.transAxes, ha="right", va="top", fontsize=13,
                             family="monospace", fontweight="bold", zorder=10, visible=False,
                             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#9ca3af", linewidth=2.5))

        home, away = tracking.team_names
        handles = [
            Line2D([], [], ls="", marker="o", ms=9, color=COLORS["H"], label=f"H  {home}"),
            Line2D([], [], ls="", marker="o", ms=9, color=COLORS["A"], label=f"A  {away}"),
            Line2D([], [], ls="", marker="o", ms=7, color=COLORS["ball"], mec="black", label="ball"),
        ]
        if player_source is not None:
            handles += [
                Line2D([], [], ls="", marker="o", ms=9, color=COLORS["H"], alpha=0.4, label="bridged"),
                Line2D([], [], ls="", marker="o", ms=9, mfc="white", mec=COLORS["H"], label="from video 2"),
            ]
        if self.is_keeper.any():
            handles += [
                Line2D([], [], ls="", marker="s", ms=9, color=COLORS["H"], mec="black", label="keeper"),
                Line2D([], [], ls="", marker="s", ms=9, mfc="white", mec=COLORS["H"], label="keeper held (last seen)"),
            ]
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 1), ncol=3, frameon=False, fontsize=9)
        self._rgba = {side: np.array(matplotlib.colors.to_rgba(COLORS[side])) for side in "HA"}

    def draw(self, frame: int) -> None:
        t = self.t
        xy = t.players[frame] * self.scale
        seen = ~np.isnan(xy[:, 0])
        for side, dots in self.dots.items():
            mask = seen & (t.team == side) & ~self.is_keeper
            dots.set_offsets(xy[mask])
            if self.player_source is not None:
                self._style_dots(dots, side, self.player_source[frame][mask])
            keeper = seen & (t.team == side) & self.is_keeper
            self.keepers[side].set_offsets(xy[keeper])
            if self.player_source is not None:
                self._style_dots(self.keepers[side], side, self.player_source[frame][keeper], edge="black")
        for label, pos, ok in zip(self.slot_labels, xy, seen):
            label.set_visible(bool(ok))
            if ok:
                label.set_position(pos)

        trail = t.ball[max(0, frame - TRAIL) : frame + 1] * self.scale
        self.trail.set_data(trail[:, 0], trail[:, 1])
        ball = t.ball[frame] * self.scale
        self.ball.set_offsets(ball[None] if not np.isnan(ball[0]) else np.empty((0, 2)))
        if self.circle is not None:
            self._draw_crowd(frame)
        if self.arrows:
            self._draw_arrows(frame)
        self.caption.set_text(self._caption(frame, int(seen.sum())))
        self._draw_badge(frame)

    def _style_dots(self, dots, side: str, source: np.ndarray, edge: str = "white") -> None:
        """Observed players full, bridged ones faint, the ones from video 2 and keepers held at their last position hollow."""
        from src.player_cleaning import KEEPER_HELD, OTHER_TRACK, SAME_TRACK, VIDEO_2

        face = np.tile(self._rgba[side], (len(source), 1))
        edge = np.tile(np.array(matplotlib.colors.to_rgba(edge)), (len(source), 1))
        face[np.isin(source, [SAME_TRACK, OTHER_TRACK]), 3] = 0.4
        hollow = np.isin(source, [VIDEO_2, KEEPER_HELD])
        face[hollow] = [1.0, 1.0, 1.0, 1.0]
        edge[hollow] = self._rgba[side]
        dots.set_facecolors(face)
        dots.set_edgecolors(edge)

    def _draw_arrows(self, frame: int) -> None:
        """Each team's common shift c, from its centre: bold in a block shift, faint elsewhere."""
        row = self.blocks.loc[frame]
        for side, arrow in self.arrows.items():
            x, y = row[f"{side}_centre_x"], row[f"{side}_centre_y"]
            u, v = row[f"{side}_shift_x"], row[f"{side}_shift_y"]
            shown = not (np.isnan(x) or np.isnan(u)) and np.hypot(u, v) >= 0.5
            arrow.set_visible(bool(shown))
            if shown:
                arrow.set_offsets([[x, y]])
                arrow.set_UVC([u], [v])
                arrow.set_alpha(1.0 if row[f"{side}_block"] else 0.35)

    def _draw_crowd(self, frame: int) -> None:
        """The circle around the densest spot: red where at least crowd_min_players stand in it."""
        x, y = self.crowd_xy[frame]
        shown = not np.isnan(x)
        self.circle.set_visible(shown)
        self.circle_label.set_visible(shown)
        if shown:
            crowded = self.crowd_n[frame] >= self.crowd_min
            self.circle.set_center((x, y))
            self.circle.set_edgecolor(COLORS["episode"] if crowded else COLORS["crowd"])
            self.circle.set_linewidth(2.2 if crowded else 1.2)
            self.circle_label.set_position((x, y - self.radius - 0.5))  # y points down: just above the circle
            self.circle_label.set_text(f"{self.crowd_n[frame]} within {self.radius:g} m")

    def _compared(self, frame: int) -> tuple[str, str]:
        """(ours, truth) at a frame: the first and the second label set ("" where there is none)."""
        sets = list(self.label_sets.values())
        ours = sets[0][frame] if sets else ""
        truth = sets[1][frame] if len(sets) > 1 else ""
        return ours, truth

    def _differs(self, frame: int) -> bool:
        ours, truth = self._compared(frame)
        return bool(ours and truth and ours != truth)

    def _draw_badge(self, frame: int) -> None:
        if not self.label_sets:
            return
        names = list(self.label_sets)
        ours, truth = self._compared(frame)
        lines = [f"{names[0]:>5s} {ours or '-'}"]
        if len(names) > 1:
            lines.append(f"{names[1]:>5s} {truth or '-'}")
        edge = "#9ca3af" if not (ours and truth) else ("#dc2626" if ours != truth else "#16a34a")
        self.badge.set_text("\n".join(lines))
        self.badge.set_color(LABEL_COLORS.get(ours, "black"))
        self.badge.get_bbox_patch().set_edgecolor(edge)
        self.badge.set_visible(True)

    def _caption(self, frame: int, n_tracked: int) -> str:
        phase = phase_at(frame, self.config)
        clock = match_clock(frame, self.config)
        ball = "" if not np.isnan(self.t.ball[frame, 0]) else "   BALL MISSING"
        text = (
            f"frame {frame}   video {video_time(frame, self.config)}   "
            f"{phase}{'  ' + clock if clock else ''}   {n_tracked} tracked{ball}"
        )
        if self.label_sets:
            text += "\n" + "   ".join(f"{name}: {labels[frame] or '-'}" for name, labels in self.label_sets.items())
            if self._differs(frame):
                text += "   <- differs"
        if self.dead_kinds:
            text += "\n" + "   ".join(f"{name}: {kinds[frame] or 'live'}" for name, kinds in self.dead_kinds.items())
        if self.episodes:
            name = self.segment_name
            segment = self.episodes.get(int(self.episode_of[frame]))
            if segment is None:
                text += f"\nno {name} here  -  n / p: next / previous {name}"
            else:
                what = getattr(segment, "summary", None) or f"{segment.zone}, up to {segment.peak} players"
                line = f"{name} {segment.id}/{len(self.episodes)} ({segment.duration_s} s from {segment.match}): {what}"
                text += "\n" + textwrap.fill(line, CAPTION_WIDTH, subsequent_indent="    ")  # clear of the legend
        return text


def run_viewer(fig, plot: PitchPlot, start: int, episode_starts: list[int] | None = None) -> None:
    """The interactive viewer. episode_starts: first frame of each episode, by id, for the keys n / p."""
    n = plot.t.n_frames
    slider = Slider(fig.add_axes([0.08, 0.035, 0.6, 0.03]), "frame", 0, n - 1, valinit=start, valstep=1, valfmt="%d")
    slider.on_changed(lambda value: (plot.draw(int(value)), fig.canvas.draw_idle()))
    frame_box = TextBox(fig.add_axes([0.84, 0.03, 0.1, 0.04]), "go to frame ", initial=str(start))

    def go_to(text: str) -> None:
        try:
            slider.set_val(int(np.clip(int(text.strip()), 0, n - 1)))
        except ValueError:
            pass  # not a number: ignore

    frame_box.on_submit(go_to)

    def move(delta: int) -> None:
        slider.set_val(int(np.clip(slider.val + delta, 0, n - 1)))

    timer = fig.canvas.new_timer(interval=1000 // plot.fps)
    timer.add_callback(move, 1)
    playing = False
    steps = {"right": 1, "left": -1, "up": plot.fps, "down": -plot.fps, "pageup": 10 * plot.fps, "pagedown": -10 * plot.fps}
    starts = list(episode_starts or [])
    current = starts.index(start) if start in starts else -1  # the episode shown, as an index into starts

    def jump(step: int) -> None:
        nonlocal current
        if starts:
            current = int(np.clip(current + step, 0, len(starts) - 1))
            slider.set_val(starts[current])

    def on_key(event) -> None:
        nonlocal playing
        if frame_box.capturekeystrokes:  # typing into the box, not navigating
            return
        if event.key in steps:
            move(steps[event.key])
        elif event.key == " ":
            playing = not playing
            timer.start() if playing else timer.stop()
        elif event.key in ("n", "p"):
            jump(1 if event.key == "n" else -1)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plot.draw(start)
    plt.show()


def save(fig, plot: PitchPlot, out: Path, start: int, end: int | None, step: int) -> None:
    if end is None:
        plot.draw(start)
        fig.savefig(out, dpi=120)
        return
    fps = max(1, round(plot.fps / step))
    writer = animation.PillowWriter(fps=fps) if out.suffix.lower() == ".gif" else animation.FFMpegWriter(fps=fps)
    clip = animation.FuncAnimation(fig, plot.draw, frames=range(start, end + 1, step))
    clip.save(out, writer=writer, dpi=100)


def main():
    parser = argparse.ArgumentParser(description="Pitch view of the tracking data.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source", help="tactical or broadcast (default: config)")
    parser.add_argument("--frame", type=int, help="video frame to start at (default: first kickoff)")
    parser.add_argument("--to", type=int, help="last frame of a clip (needs --out)")
    parser.add_argument("--step", type=int, default=1, help="frame step of a clip")
    parser.add_argument("--labels", action="append", default=[], help="frame,label CSV to show (repeatable)")
    parser.add_argument("--out", type=Path, help=".png for one frame, .gif or .mp4 for a clip")
    parser.add_argument("--episodes", action="store_true",
                        help="list the crowded moments of src.dead_ball and step through them (keys n / p)")
    parser.add_argument("--episode", type=int, help="start at this episode id (implies --episodes)")
    parser.add_argument("--gaps", action="store_true",
                        help="list the long gaps of the ball data, judged dead / unclear, and step through them")
    parser.add_argument("--gap", type=int, help="start at this gap id (implies --gaps)")
    parser.add_argument("--blocks", action="store_true",
                        help="list the block shifts of src.block_shift and step through them, with arrows")
    parser.add_argument("--block", type=int, help="start at this block shift id (implies --blocks)")
    parser.add_argument("--restarts", action="store_true",
                        help="list the restarts of src.restarts (who takes them) and step through them")
    parser.add_argument("--restart", type=int, help="start at this restart id (implies --restarts)")
    parser.add_argument("--tags", action="store_true",
                        help="list the set-piece tags of Metrica's smart tagging, checked (src.smart_tagging)")
    parser.add_argument("--tag", type=int, help="start at this tag id (implies --tags)")
    parser.add_argument("--segments", action="store_true",
                        help="the dead-ball events of src.segments, with who has the ball after each")
    parser.add_argument("--segment", type=int, help="start at this event id (implies --segments)")
    parser.add_argument("--all", action="store_true",
                        help="every situation that may interrupt play in one list: restarts, long gaps, "
                             "crowds and smart-tagging set pieces")
    parser.add_argument("--situation", type=int, help="start at this situation id (implies --all)")
    parser.add_argument("--dead", action="store_true",
                        help="Stefan's and Tim's dead balls with their kinds (src.dead_ball_compare), n / p through them")
    parser.add_argument("--dead-ball", type=int, help="start at this dead ball id (implies --dead)")
    parser.add_argument("--rejected", action="store_true",
                        help="with --restarts: step through the still balls judged in play instead")
    args = parser.parse_args()
    if args.to is not None and args.out is None:
        parser.error("--to needs --out")
    with_episodes = args.episodes or args.episode is not None
    with_gaps = args.gaps or args.gap is not None
    with_blocks = args.blocks or args.block is not None
    with_restarts = args.restarts or args.restart is not None
    with_tags = args.tags or args.tag is not None
    with_all = args.all or args.situation is not None
    with_segments = args.segments or args.segment is not None
    with_dead = args.dead or args.dead_ball is not None
    if args.rejected and not with_restarts:
        parser.error("--rejected goes with --restarts")
    pipeline_modes = with_gaps or with_blocks or with_restarts or with_tags or with_all or with_segments or with_dead
    if with_episodes + with_gaps + with_blocks + with_restarts + with_tags + with_all + with_segments + with_dead > 1:
        parser.error("only one of --episodes, --gaps, --blocks, --restarts, --tags, --all and --segments")
    if (with_episodes or pipeline_modes) and args.source == "broadcast":
        parser.error("--episodes, --gaps, --blocks, --restarts, --tags, --all and --segments work on the "
                     "tactical tracking")

    config = load_config(args.config)
    crowd = table = blocks = player_source = dead_kinds = None
    segment_name, wanted = "episode", None
    if pipeline_modes:
        from dataclasses import replace

        from src.pipeline import dead_ball_analysis, load_match, measured

        data = load_match(config)
        ball = data.ball[["x", "y"]].to_numpy(dtype=np.float32)
        tracking = replace(data.tracking, ball=ball)  # show the ball data, the merged ball
        player_source = data.player_source  # and the cleaned players: bridged faint, video 2 hollow
        if with_gaps:
            say("judging the long gaps, as main.py does (a few seconds) ...")
            dead = dead_ball_analysis(data, config)
            crowd, table, segment_name, wanted = dead.indicators, dead.gaps, "gap", args.gap
        elif with_dead:
            from src.dead_ball_compare import dead_ball_table, dead_balls_of_match, kinds_per_frame

            say("finding Stefan's and Tim's dead balls, as src.dead_ball_compare does (a few seconds) ...")
            bars = dead_balls_of_match(data, config)
            dead_kinds = {name.split(" ")[0]: kinds_per_frame(spans, tracking.n_frames) for name, spans in bars.items()}
            table, segment_name, wanted = dead_ball_table(bars, config), "dead ball", args.dead_ball
        elif with_segments:
            from src.segments import find_segments

            say("cutting the match at the dead balls, as src.segments does (a few seconds) ...")
            dead = dead_ball_analysis(data, config)
            crowd, (table, _) = dead.indicators, find_segments(data, config, dead)
            table = table.assign(first=table["start"], duration_s=table["dead_s"])  # from where play stopped
            segment_name, wanted = "event", args.segment
        elif with_all:
            from src.pipeline import all_situations

            say("collecting every situation: restarts, long gaps, crowds, smart tagging (a few seconds) ...")
            dead = dead_ball_analysis(data, config)
            crowd, table = dead.indicators, all_situations(data, config, dead)
            segment_name, wanted = "situation", args.situation
        elif with_tags:
            from src.restarts import candidates
            from src.smart_tagging import check_set_pieces, load_tags, signatures, smart_tagging_path

            say("checking the smart tagging, as src.smart_tagging does (a few seconds) ...")
            measured_tracking = measured(data)
            tags = load_tags(smart_tagging_path(config), config, data.tracking.team_names)
            table = check_set_pieces(tags, candidates(measured_tracking, data.ball, config),
                                     signatures(measured_tracking, data.ball, config), config)
            segment_name, wanted = "tag", args.tag
        elif with_restarts:
            from src.restarts import IN_PLAY, RESTART, find_restarts

            say("finding the restarts, as src.restarts does (a few seconds) ...")
            verdict, segment_name = (IN_PLAY, "rejected") if args.rejected else (RESTART, "restart")
            table, wanted = find_restarts(measured(data), data.ball, config, verdict), args.restart
        else:
            from src.block_shift import block_episodes, block_indicators

            say("finding the block shifts, as src.block_shift does (a few seconds) ...")
            blocks = block_indicators(measured(data), data.ball, config)
            table, segment_name, wanted = block_episodes(blocks, data.tracking, config), "block", args.block
    else:
        tracking = load_tracking(config, args.source)
    if with_episodes:
        from src.dead_ball import episodes as find_episodes
        from src.dead_ball import indicators

        say("finding the crowded moments, as src.dead_ball does (a few seconds) ...")
        crowd = indicators(tracking, config)
        table, wanted = find_episodes(crowd, tracking, config), args.episode

    start = first_graded_frame(config) if args.frame is None else args.frame
    if not 0 <= start < tracking.n_frames:
        parser.error(f"--frame must be in 0..{tracking.n_frames - 1}")
    if args.labels:
        labels = {Path(p).stem: read_labels(p, tracking.n_frames) for p in args.labels}
    else:  # by default ours (python -m src.baseline) and the public ground truth, where there are
        labels = {}
        ours = PROJECT_ROOT / "outputs" / f"{config['data']['game']}_baseline.csv"
        truth = PROJECT_ROOT / config["public_gt"] if config.get("public_gt") else None
        if ours.is_file():
            labels["ours"] = read_labels(ours, tracking.n_frames)
        else:  # an empty column, so the truth stays the second one
            say(f"no labels of ours yet at {ours}: run python -m src.baseline")
            labels["ours"] = np.full(tracking.n_frames, "", dtype="<U1")
        if truth is not None and truth.is_file():
            labels["truth"] = read_labels(truth, tracking.n_frames)
    if table is not None:
        if table.empty:
            parser.error(f"no {segment_name}s with the current settings in config.yaml")
        wanted = 1 if wanted is None else wanted
        if not 1 <= wanted <= len(table):
            parser.error(f"--{segment_name} must be in 1..{len(table)}")
        print(table.drop(columns="summary", errors="ignore").to_string(index=False))
        print(f"\nopening {segment_name} {wanted}; keys: n / p next / previous {segment_name}, "
              "arrows step, space plays")
        if args.frame is None:
            start = int(table.loc[table["id"] == wanted, "first"].iloc[0])

    if args.out is None:  # free the keys the viewer uses from matplotlib's own bindings
        for key, freed in FREED_KEYS.items():
            plt.rcParams[key] = [k for k in plt.rcParams[key] if k not in freed]
    fig, ax = plt.subplots(figsize=(12, 8.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.09)
    plot = PitchPlot(ax, tracking, config, labels, crowd=crowd, episodes=table, segment_name=segment_name,
                     blocks=blocks, player_source=player_source, dead_kinds=dead_kinds)

    if args.out is None:
        run_viewer(fig, plot, start, None if table is None else list(table["first"]))
    else:
        end = None if args.to is None else min(args.to, tracking.n_frames - 1)
        save(fig, plot, args.out, start, end, args.step)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
