"""Entry point of the possession pipeline - the command; the pipeline itself is src/pipeline.py.

    python main.py                        # load, merge the ball of both videos, judge the long gaps, plot
    python main.py --save-plot ball.png   # the same, but write ball_first_half.png and ball_second_half.png
    python -i main.py                     # the same, then stay in Python with `data`, `dead` and `balls`

    data     src.pipeline.MatchData: tracking, DataFrames per phase, data.ball = THE ball data
    dead     src.pipeline.DeadBall: crowding and speed per frame, crowd episodes, long gaps judged
    balls    the ball per half and source, for the comparison table and the plot

The long gaps can be looked at on the pitch: python -m src.visualization --gaps (keys n / p).

For prediction, CSV export and the official public check, run from the repository root:
    .venv/bin/python run_pipeline.py --evaluate
This command remains the detailed two-camera diagnostic viewer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.ball_analysis import plot_gap_grid
from src.data_loader import DEFAULT_CONFIG, load_config
from src.dead_ball import gap_summary, ground_truth_share
from src.pipeline import (
    DEAD_BALLS,
    HALVES,
    MERGED,
    VIDEO_1,
    VIDEO_2,
    ball_sources,
    dead_ball_analysis,
    describe,
    gap_tables,
    load_match,
    load_public_gt,
    merge_table,
    verdict_halves,
)
from src.progress import Steps, say


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--save-plot", type=Path, help="write the ball plot to this file instead of a window")
    args = parser.parse_args()

    config = load_config(args.config)
    steps = Steps(10)
    data = load_match(config, steps)
    with steps(f"Dead balls: crowding and speed of the players, the gaps still longer than "
               f"{config['ball']['max_gap_s']} s judged"):
        dead = dead_ball_analysis(data, config)
        gt = load_public_gt(config, data.tracking.n_frames)
        if gt is not None:
            dead.gaps["public_gt_D"] = ground_truth_share(dead.gaps, gt)
    with steps(f"Ball gaps per half ({', '.join(HALVES)}) and video"):
        balls = ball_sources(data)
        tables = gap_tables(balls, config)
    with steps("Drawing the plots, one window per half"):
        plotted = {VIDEO_1: (balls[VIDEO_1], "tactical", {})}
        if data.video_2 is not None:
            plotted[VIDEO_2] = (data.video_2.balls, "broadcast", data.video_2.offsets)
        if data.merged is not None:
            plotted[MERGED] = (data.merged, "tactical", {})
        plotted[DEAD_BALLS] = (verdict_halves(data, dead.gaps), "tactical", {})
        figures = {half: plot_gap_grid(plotted, config, halves=[half]) for half in HALVES}
        for half, fig in figures.items():
            if fig.canvas.manager is not None:
                fig.canvas.manager.set_window_title(f"Ball gaps: {half}")

    print()
    print(describe(data, config))
    print("\nBall per half and video, before the analysis (a gap = frames in a row without the ball):\n")
    print(tables)
    if data.merged is not None:
        print(merge_table(data.merged, config))
    print(f"\nThe gaps of data.ball still longer than {config['ball']['max_gap_s']} s, judged "
          "(public_gt_D: for understanding only, never to tune on):")
    print(dead.gaps.drop(columns="summary").to_string(index=False))
    print("\n" + gap_summary(dead.gaps).to_string(index=False))
    print("\nlook at them on the pitch: .venv/bin/python -m src.visualization --gap <id>   (then n / p)")
    if args.save_plot:
        out = args.save_plot
        for half, fig in figures.items():
            path = out.with_name(f"{out.stem}_{half}{out.suffix}")
            fig.savefig(path, dpi=120)
            print(f"wrote {path}")
    else:
        say("two plot windows are open, one per half - close them to finish")
        plt.show()
    return data, dead, balls


if __name__ == "__main__":
    data, dead, balls = main()
