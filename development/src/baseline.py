"""A first labelling of every frame: dead balls D, possession from the restarts and ball control.

    python -m src.baseline                          # writes outputs/<game>_baseline.csv (frame,label)
    python -m src.visualization --labels outputs/BAR-ATH_baseline.csv --labels ../evaluation/gt/BAR-ATH_possession_gt_public.csv

The rule (settings under `baseline` in config.yaml):

    D          the dead balls of src/segments.py, from where play stopped to where the ball is
               played again; the graded phase before the second kickoff; everything outside
               the halves (not graded)
    H / A      in each live stretch, the team that has the ball after the event before it (at
               the start of a half: the first team in control). It changes when the other team
               controls the ball for longer than switch_s without a break: one of its players
               within control_m of the ball, nearer than the team in possession. The change
               counts from where that control began.

The public ground truth is scored for understanding only: tuning on it is against the rules.
The official spellF1 comes from the scorer of the competition repository.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.dead_ball import _runs
from src.progress import Steps, say
from src.restarts import _team_distances


def label_frames(n_frames: int, events: pd.DataFrame, live: pd.DataFrame, tracking, ball: pd.DataFrame,
                 config: dict) -> np.ndarray:
    """One label per video frame, see the module docstring. tracking: without the held keepers."""
    settings, fps = config["baseline"], config["fps"]
    labels = np.full(n_frames, "D", dtype="<U1")
    near = _team_distances(tracking, ball, config)
    offset = tracking.first_frame
    with np.errstate(invalid="ignore"):  # NaN: no ball or no player - nobody controls it
        control = {side: (near[side] <= settings["control_m"]) & ~(near[other] < near[side])
                   for side, other in (("H", "A"), ("A", "H"))}
    longer = int(round(settings["switch_s"] * fps))
    for stretch in live.itertuples():
        a, b = stretch.first - offset, stretch.last - offset + 1
        team = stretch.team
        if not team:  # the start of a half: the first team in control
            first = {side: np.flatnonzero(control[side][a:b]) for side in "HA"}
            starts = {side: f[0] for side, f in first.items() if len(f)}
            team = min(starts, key=starts.get) if starts else "H"
        changes = []  # (relative frame, new team)
        for side in "HA":
            changes += [(start, side) for start, stop in _runs(control[side][a:b]) if stop - start > longer]
        current, at = team, 0
        for start, side in sorted(changes):
            if side != current:
                labels[a + at : a + start] = current
                current, at = side, start
        labels[a + at : b] = current
    return labels


def stefan_labels(n_frames: int, events: pd.DataFrame, live: pd.DataFrame, tracking, ball: pd.DataFrame,
                  config: dict) -> np.ndarray:
    """Our dead balls and restart teams, Stefan's possession model (src/possession_model.py) in
    the live stretches: control = the observed ball at most control_max_speed_ms, a player within
    control_radius_m and more than control_margin_m nearer than every opponent; a change after
    confirmation_s of it, dated back; otherwise the owner keeps the ball. Settings under `model`.
    Outside the halves, and in our dead balls: D. Frames no team is known for yet: X."""
    from src.phases import half_of_frames
    from src.positions import players_on_pitch_m
    from src.possession_model import PossessionModel

    frames = np.arange(n_frames)
    halves = [p for p in config["phases"] if "clock_start_min" in p]
    inside = np.zeros(n_frames, dtype=bool)
    for half in halves:
        inside[half["first"] : half["last"] + 1] = True
    dead = ~inside
    for event in events.itertuples():
        dead[event.start : event.played + 1] = True
    xy = ball[["x_m", "y_m"]].reindex(frames).to_numpy(dtype=float)
    players = players_on_pitch_m(tracking, config["pitch"])
    features = pd.DataFrame(index=pd.RangeIndex(n_frames, name="frame"))
    for side in "HA":
        with np.errstate(invalid="ignore"):
            distance = np.linalg.norm(players[:, tracking.team == side] - xy[:, None], axis=2)
        distance[~np.isfinite(distance)] = np.inf
        features[f"nearest_{side}_m"] = distance.min(axis=1) if distance.shape[1] else np.inf
    seeds = np.full(n_frames, "", dtype="<U1")
    for stretch in live.itertuples():
        if stretch.team in ("H", "A"):
            seeds[stretch.first] = stretch.team
    features["dead"], features["graded"] = dead, np.ones(n_frames, dtype=bool)
    features["half"] = half_of_frames(n_frames, config)
    features["ball_source"] = ball["source"].reindex(frames).astype(str).to_numpy()
    features["ball_speed_ms"] = ball["speed_ms"].reindex(frames).to_numpy(dtype=float)
    holds = np.full(n_frames, "", dtype="<U1")  # the keeper holds the unseen ball: his team (a catch)
    for catch in events.attrs.get("catches", pd.DataFrame(columns=["first", "last", "team"])).itertuples():
        holds[catch.first : catch.last + 1] = catch.team
    holds[dead] = ""
    features["restart_team"], features["hold_team"] = seeds, holds
    return PossessionModel(config).predict(features)


def restart_based(features: pd.DataFrame, events: pd.DataFrame, live: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Stefan's feature table (src.feature_extraction) with our dead balls in place of his: D from
    where play stopped to where the restart is played (src.segments), outside the halves and
    before a kickoff; the team at the start of each live stretch; the keeper's team while he holds
    the unseen ball (catches). His control evidence and possession model are left as they are."""
    n = len(features)
    inside = np.zeros(n, dtype=bool)
    for half in (p for p in config["phases"] if "clock_start_min" in p):
        inside[half["first"] : half["last"] + 1] = True
    dead = ~inside
    for event in events.itertuples():
        dead[event.start : event.played + 1] = True
    seeds = np.full(n, "", dtype="<U1")
    for stretch in live.itertuples():
        if stretch.team in ("H", "A") and stretch.first < n:
            seeds[stretch.first] = stretch.team
    holds = np.full(n, "", dtype="<U1")
    for catch in events.attrs.get("catches", pd.DataFrame(columns=["first", "last", "team"])).itertuples():
        holds[catch.first : catch.last + 1] = catch.team
    holds[dead] = ""
    out = features.copy()
    out["dead"], out["restart_team"], out["hold_team"] = dead, seeds, holds
    return out


DEAD_LOGICS = ("gap", "restart", "union", "combined", "evidence")


def prepare(data, config: dict, set_pieces: pd.DataFrame | None = None):
    """(prepared, features) for the possession model: Stefan's preprocessing and features
    (src.preprocessing, src.feature_extraction), with the dead balls of baseline.dead:

        gap       Stefan's (src.stoppages): long ball gaps judged by the players' pace - the
                  submission, chosen on the labelled sample games of other matches (src.benchmark)
        restart   Tim's (src.segments): restarts, where play stopped, keeper catches
        union     dead where either says so, Tim's restart teams
        combined  both judged per kind of restart (src.dead_ball_compare)
        evidence  per stoppage the logic with the richer data there (choose_by_evidence)

    data: src.pipeline.load_match; set_pieces: Smart Tagging (None: load as configured)."""
    from src.feature_extraction import extract_features
    from src.preprocessing import preprocess

    logic = config.get("baseline", {}).get("dead", "gap")
    if logic not in DEAD_LOGICS:
        raise ValueError(f"baseline.dead must be one of {DEAD_LOGICS}, not {logic!r}")
    prepared = preprocess(data, config, set_pieces=set_pieces)
    features = extract_features(prepared, config)
    if logic == "gap":
        return prepared, features
    if logic == "combined":
        from src.dead_ball_compare import COMBINED, judged_dead_balls, live_after

        bars, tim_events, stefan_seeds = judged_dead_balls(data, config)
        spans = bars[COMBINED]
        events = pd.DataFrame({"start": [f for f, _, _ in spans], "played": [s - 1 for _, s, _ in spans]})
        live = live_after(spans, tim_events, stefan_seeds, config)
    else:
        from src.segments import find_segments

        events, live = find_segments(data, config)
    if logic == "evidence":
        return prepared, choose_by_evidence(features, events)
    gap_dead = features["dead"].to_numpy(dtype=bool)
    features = restart_based(features, events, live, config)
    if logic == "union":
        features["dead"] = features["dead"].to_numpy(dtype=bool) | gap_dead
    return prepared, features


def choose_by_evidence(features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Both dead-ball logics, one chosen per stoppage by what the data shows there. For every
    stretch either calls dead:

        the tracked ball is seen crossing a line (a Tim event starting "ball out")
            -> Tim's span: the start is observed; Stefan's waits until the tracker loses the ball
        Tim sees a restart (ball still, then played) where Stefan sees no stoppage at all
            -> Tim's span: a placed ball is direct evidence
        otherwise, Stefan has the stretch dead
            -> Stefan's span: the ball is not seen, only the players' pace can tell
        otherwise (only Tim, from walking pace or a merged gap alone) -> live

    The team that restarts play comes from the same side: Tim's after_team at the end of his
    span, else Stefan's restart seed. Stefan's keeper holds are kept; Tim's catches are not used.
    features: Stefan's (src.feature_extraction); events: Tim's (src.segments, with start_by,
    clear, start, played, after_team)."""
    n = len(features)
    stefan = features["dead"].to_numpy(dtype=bool)
    tim = np.zeros(n, dtype=bool)
    for event in events.itertuples():
        tim[event.start : event.played + 1] = True
    chosen = np.zeros(n, dtype=bool)
    seeds = features["restart_team"].to_numpy().astype("<U1").copy()
    for a, b in _runs(stefan | tim):
        mine = events[(events["start"] < b) & (events["played"] >= a)]
        observed_out = (mine["start_by"] == "ball out").any() if len(mine) else False
        placed_alone = bool(mine["clear"].any()) and not stefan[a:b].any() if len(mine) else False
        if observed_out or placed_alone:
            chosen[a:b] = tim[a:b]
            for event in mine.itertuples():
                if event.played + 1 < n and event.after_team in ("H", "A"):
                    seeds[event.played + 1] = event.after_team
        elif stefan[a:b].any():
            chosen[a:b] = stefan[a:b]
    out = features.copy()
    holds = out["hold_team"].to_numpy().astype("<U1").copy() if "hold_team" in out else np.full(n, "", "<U1")
    holds[chosen] = ""
    out["dead"], out["restart_team"], out["hold_team"] = chosen, seeds, holds
    return out


def decode(prepared, features: pd.DataFrame, config: dict) -> np.ndarray:
    """Stefan's possession model on the features, the optional team-shape stage, and the
    postprocessing (ungraded frames X): one label per frame."""
    from src.possession_model import PossessionModel
    from src.postprocessing import postprocess

    model = PossessionModel(config)
    labels = model.predict(features)
    if config.get("team_shape", {}).get("enabled", False):
        from src.team_shape import refine

        labels, _ = refine(labels, prepared.tracking, model.evidence(features), features["graded"].to_numpy(), config)
    return postprocess(labels, config)


def predict(data, config: dict) -> np.ndarray:
    """The submission: one label per frame of the match of src.pipeline.load_match."""
    return decode(*prepare(data, config), config)


def score(labels: np.ndarray, gt: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Frame accuracy and per-label F1 over the frames the public ground truth grades (understanding only)."""
    graded = np.isin(gt, ["H", "A", "D"])
    truth, ours = gt[graded], labels[graded]
    f1 = {}
    for label in "HAD":
        tp = np.sum((ours == label) & (truth == label))
        precision = tp / max(np.sum(ours == label), 1)
        recall = tp / max(np.sum(truth == label), 1)
        f1[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {"graded frames": int(graded.sum()), "accuracy": f"{np.mean(truth == ours):.1%}",
               **{f"F1 {label}": f"{value:.3f}" for label, value in f1.items()},
               "macro F1": f"{np.mean(list(f1.values())):.3f}"}
    table = pd.crosstab(pd.Series(truth, name="ground truth"), pd.Series(ours, name="ours"))
    return summary, table


def main():
    from src.data_loader import DEFAULT_CONFIG, PROJECT_ROOT, load_config
    from src.pipeline import dead_ball_analysis, load_match, load_public_gt, measured
    from src.segments import find_segments

    parser = argparse.ArgumentParser(description="A first labelling of every frame.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, help="the CSV to write (default outputs/<game>_baseline.csv)")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["baseline"]
    out = args.out or PROJECT_ROOT / "outputs" / f"{config['data']['game']}_baseline.csv"
    steps = Steps(8)

    data = load_match(config, steps)
    logic = settings.get("dead", "gap")
    with steps(f"Every frame labelled: dead balls ({logic}), Stefan's possession model in live play"):
        if settings.get("possession", "stefan") == "ours":  # Tim's first rule, on Tim's dead balls
            events, live = find_segments(data, config, dead_ball_analysis(data, config))
            labels = label_frames(data.tracking.n_frames, events, live, measured(data), data.ball, config)
        else:
            labels = predict(data, config)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(out, index=False)
        say(f"wrote {out} - match data, do not commit or share it")

    halves = [p for p in config["phases"] if "clock_start_min" in p]
    inside = np.zeros(len(labels), dtype=bool)
    for half in halves:
        inside[half["first"] : half["last"] + 1] = True
    shares = pd.Series(labels[inside]).value_counts(normalize=True).reindex(list("HAD"), fill_value=0)
    spells = sum(len(_runs(labels[half["first"] : half["last"] + 1] == side)) for half in halves for side in "HAD")
    print("\nIn the halves: " + ", ".join(f"{k} {v:.1%}" for k, v in shares.items()) + f"; {spells} spells")
    gt = load_public_gt(config, len(labels))
    if gt is not None:
        summary, table = score(labels, gt)
        print("\nAgainst the public ground truth (understanding only, never tune on it):")
        for name, value in summary.items():
            print(f"  {name:14s} {value}")
        print(table.to_string())
        from evaluation.metrics import spell_f1

        spell = spell_f1(gt, labels, config["baseline"].get("spell_min_iou", 0.5))
        print(f"\nspell F1 (our approximation: a spell is a hit at IoU >= "
              f"{config['baseline'].get('spell_min_iou', 0.5):g} with a true spell of its label):")
        for name in ("H", "A", "D", "all"):
            s = spell[name]
            print(f"  {name:4s} F1 {s['f1']:.3f}   true spells {s['spells_true']:4d}, ours {s['spells_pred']:4d}, "
                  f"hits {s['hits']:4d}")
        print(f"  macro over H, A, D: {spell['macro_f1']:.3f}")
    print(f"\nthe official spellF1: run the competition's scorer on {out}")
    print(f"look at it: .venv/bin/python -m src.visualization --labels {out} --labels {config.get('public_gt', '')}")
    return labels


if __name__ == "__main__":
    labels = main()
