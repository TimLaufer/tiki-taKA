"""The tracking as pandas DataFrames, to look at and explore.

    python -m src.inspect_data --frame 20000              # rows from frame 20000 + that frame as a table
    python -m src.inspect_data --frame 20000 --full-row   # also every column of that row
    python -m src.inspect_data --describe                 # what the entries of the raw .txt mean

In Python or a notebook:

    from src.data_loader import load_config, load_tracking
    from src.inspect_data import to_dataframe, frame_table
    config = load_config()
    tracking = load_tracking(config)
    df = to_dataframe(tracking, config)       # one row per video frame
    frame_table(tracking, config, 20000)      # one row per object in frame 20000

This prints real match data: look at it, do not commit or share it.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import DEFAULT_CONFIG, Tracking, atd_paths, load_config, load_tracking
from src.phases import first_graded_frame, half_of_frames, phase_labels


def describe_metadata(meta_path) -> str:
    """What the entries of the raw ATD .txt mean, as declared in its metadata .xml."""
    root = ET.parse(meta_path).getroot()
    for el in root.iter():
        el.tag = el.tag.rsplit("}", 1)[-1]  # drop XML namespaces
    meta = root.find("Metadata")

    def text(el, tag):
        return (el.findtext(tag) or "").strip() if el is not None else ""

    def params(el):
        return ", ".join(f"{text(p, 'Name')}={text(p, 'Value')}" for p in el.iter("ProviderParameter")) if el is not None else ""

    lines = [
        f"metadata file  {Path(meta_path).name}",
        f"frame rate     {text(meta.find('GlobalConfig'), 'FrameRate')} fps",
        f"parameters     {params(meta.find('GlobalConfig/ProviderGlobalParameters'))}",
        "",
        "teams:",
    ]
    players = list(meta.iter("Player"))
    for team in meta.iter("Team"):
        n_tracks = sum(p.get("teamId") == team.get("id") for p in players)
        lines.append(f"  {team.get('id'):<10} {text(team, 'Name'):<24} {n_tracks} tracks   {params(team)}")

    lines += ["", "sensors (what is measured per track):"]
    for sensor in meta.iter("Sensor"):
        for channel in sensor.iter("Channel"):
            lines.append(
                f"  {sensor.get('id'):<10} channel {channel.get('id'):<4} "
                f"unit {text(channel, 'Unit'):<12} {text(channel, 'Name')}"
            )

    channel_of = {pc.get("id"): pc.get("channelId") for pc in meta.iter("PlayerChannel")}
    player_of = {pc.get("id"): pc.get("playerId") for pc in meta.iter("PlayerChannel")}

    def render(el) -> str:
        sep = el.get("separator", "")
        kids = list(el)
        if el.tag == "StringRegister":
            return f"<{el.get('name')}>"
        if kids and all(k.tag == "PlayerChannelRef" for k in kids):
            track = player_of.get(kids[0].get("playerChannelId"), "?")
            return f"[track {track}: " + sep.join(channel_of.get(k.get("playerChannelId"), "?") for k in kids) + "]"
        if kids and all(k.tag == "BallChannelRef" for k in kids):
            return "[ball: " + sep.join(k.get("channelId") for k in kids) + "]"
        parts = [render(k) for k in kids]
        if len(parts) > 4:
            parts = parts[:2] + [f"... {len(parts)} tracks in total ..."] + parts[-1:]
        return f" {sep} ".join(parts)

    specs = root.findall("DataFormatSpecifications/DataFormatSpecification")
    lines += ["", f"layout of one line in the .txt ({len(specs)} layout(s), first one shown):"]
    if specs:
        lines.append(f"  frames {specs[0].get('startFrame')}-{specs[0].get('endFrame')}:  {render(specs[0])}")
    return "\n".join(lines)


def to_dataframe(tracking: Tracking, config: dict) -> pd.DataFrame:
    """One row per video frame; the index is the frame, i.e. the submission row.

    Columns: time_s, phase (see src/phases.py), half (1/2, 0 outside play),
    ball_x, ball_y, ball_s, then per
    track slot <slot>_x, <slot>_y, <slot>_s. Positions are normalised 0-1,
    speeds (_s) are in df.attrs["speed_unit"] as declared by the metadata,
    NaN means not observed. Columns the export does not fill are left out.
    df.attrs["team"] maps each slot to "H" or "A". Every field: docs/DATEN_LEGENDE.md.
    """
    n = tracking.n_frames
    columns = {
        "time_s": np.arange(n) / config["fps"],
        "phase": phase_labels(n, config),
        "half": half_of_frames(n, config),
        "ball_x": tracking.ball[:, 0],
        "ball_y": tracking.ball[:, 1],
    }
    if not np.isnan(tracking.ball_z).all():
        columns["ball_z"] = tracking.ball_z
    if not np.isnan(tracking.ball_speed).all():
        columns["ball_s"] = tracking.ball_speed
    has_speed = not np.isnan(tracking.speed).all()
    for k, slot in enumerate(tracking.slot_ids):
        columns[f"{slot}_x"] = tracking.players[:, k, 0]
        columns[f"{slot}_y"] = tracking.players[:, k, 1]
        if has_speed:
            columns[f"{slot}_s"] = tracking.speed[:, k]
    df = pd.DataFrame(columns, index=pd.RangeIndex(n, name="frame"))
    df.attrs["team"] = dict(zip(tracking.slot_ids.tolist(), tracking.team.tolist()))
    df.attrs["speed_unit"] = tracking.speed_unit
    return df


def frame_table(tracking: Tracking, config: dict, frame: int) -> pd.DataFrame:
    """One row per object seen in `frame`: the ball first, then players nearest the ball first.

    The speed column is named after the unit the metadata declares, e.g. speed_kph.
    """
    scale = np.asarray(config["pitch"], dtype=float)
    seen = ~np.isnan(tracking.players[frame, :, 0])
    ball = tracking.ball[frame].astype(float)
    speed_column = f"speed_{tracking.speed_unit}" if tracking.speed_unit else "speed"

    xy = np.vstack([ball, tracking.players[frame, seen].astype(float)])
    table = pd.DataFrame({
        "object": np.concatenate([["ball"], tracking.slot_ids[seen]]),
        "team": np.concatenate([["-"], tracking.team[seen]]),
        "x": xy[:, 0],
        "y": xy[:, 1],
        "x_m": xy[:, 0] * scale[0],
        "y_m": xy[:, 1] * scale[1],
        speed_column: np.concatenate([[tracking.ball_speed[frame]], tracking.speed[frame, seen]]).astype(float),
        "dist_to_ball_m": np.hypot(*((xy - ball) * scale).T),
    })
    players = table.iloc[1:].sort_values(["dist_to_ball_m", "team"], na_position="last")
    return pd.concat([table.iloc[:1], players], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Show the tracking as pandas DataFrames.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source", help="tactical or broadcast (default: config)")
    parser.add_argument("--frame", type=int, help="video frame (default: first kickoff)")
    parser.add_argument("--rows", type=int, default=5, help="rows of the wide DataFrame to show")
    parser.add_argument("--full-row", action="store_true", help="also print every column of the row")
    parser.add_argument("--describe", action="store_true", help="explain the raw file from its metadata, then stop")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.describe:
        meta_path, _ = atd_paths(config, args.source or config["data"]["source"])
        print(describe_metadata(meta_path))
        return
    tracking = load_tracking(config, args.source)
    frame = first_graded_frame(config) if args.frame is None else args.frame
    if not 0 <= frame < tracking.n_frames:
        parser.error(f"--frame must be in 0..{tracking.n_frames - 1}")

    df = to_dataframe(tracking, config)
    n_slots = len(tracking.slot_ids)
    per_slot = [c.split("_", 1)[1] for c in df.columns if c.startswith(f"{tracking.slot_ids[0]}_")]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.precision", 3)

    print(f"DataFrame: {len(df)} rows (one per video frame) x {df.shape[1]} columns")
    fixed = [c for c in df.columns if c.rsplit("_", 1)[0] not in df.attrs["team"]]
    print(f"columns: {', '.join(fixed)}, "
          f"then <slot>_{'/_'.join(per_slot)} for each of {n_slots} track slots")
    print(f"speeds (_s) in: {tracking.speed_unit or '-'} (as declared in the metadata)\n")
    print(f"Rows {frame}-{frame + args.rows - 1} (middle columns hidden):")
    print(df.iloc[frame : frame + args.rows])

    if args.full_row:
        print(f"\nRow {frame}, every column (team of each slot in brackets):")
        row = df.iloc[frame]
        team = df.attrs["team"]
        for column, value in row.items():
            slot = column.rsplit("_", 1)[0]
            print(f"  {column:<12} {value:>10.4f}   {f'({team[slot]})' if slot in team else ''}")

    table = frame_table(tracking, config, frame)
    print(f"\nFrame {frame} as a table, one object per line ({len(table) - 1} of {n_slots} slots seen):")
    print(table.round(3).round({"x_m": 1, "y_m": 1, "dist_to_ball_m": 1, table.columns[6]: 1}).to_string(index=False))


if __name__ == "__main__":
    main()
