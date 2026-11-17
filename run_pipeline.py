"""Generate and validate a full-match submission; optionally check public scores.

Run from the repository root: .venv/bin/python run_pipeline.py --evaluate
Prediction never reads labels. Evaluation is a separate step after the CSV is saved.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "development"))
sys.path.insert(0, str(ROOT / "evaluation"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))

import numpy as np
import pandas as pd
import possession_metrics as official

from games import GAMES
from src.data_loader import (DEFAULT_CONFIG, atd_paths, load_config, atd_frame_rate,
                             check_frame_ids, _find_file, get_data_dir)
from src.feature_extraction import extract_features
from src.phases import check_phases, graded_mask
from src.pipeline import load_match
from src.possession_model import PossessionModel
from src.postprocessing import postprocess
from src.preprocessing import preprocess
from starter.baseline import nearest_player_labels, smooth, team_of_each_track, write_submission


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict(config: dict, mode: str):
    """Return labels and diagnostics without access to public or private labels."""
    expected = GAMES[config["data"]["game"]]["n_video_frames"]
    check_phases(config, expected)
    if config["data"]["source"] != "tactical":
        raise ValueError("Submissions must use the tactical timeline")
    if atd_frame_rate(config, "tactical") != config["fps"] or config["fps"] != GAMES[config["data"]["game"]]["fps"]:
        raise ValueError("Configured fps disagrees with tactical metadata or the submission contract")
    check_frame_ids(atd_paths(config, "tactical")[1], expected)
    if mode == "baseline":
        meta, raw = atd_paths(config, "tactical")
        codes = nearest_player_labels(raw, team_of_each_track(meta), expected, config["pitch"])
        # Reproduce the published reference exactly, independently of new model settings.
        return smooth(codes, 50), {"algorithm": "published baseline_smooth50"}
    if mode != "current":
        raise ValueError(f"Unknown mode: {mode}")
    data = load_match(config, use_video_2=config.get("sync", {}).get("enabled", True))
    print("Judging stoppages and restarts ...", flush=True)
    prepared = preprocess(data, config)
    features = extract_features(prepared, config)
    model = PossessionModel(config)
    labels = model.predict(features)
    graded = graded_mask(len(labels), config)
    shape = {"enabled": False}
    if config.get("team_shape", {}).get("enabled", False):
        from src.team_shape import refine
        labels, shape = refine(labels, prepared.tracking, model.evidence(features), graded, config)
        shape["enabled"] = True
    labels = postprocess(labels, config)
    gaps = prepared.gaps
    return labels, {
        "algorithm": "stoppages from tactical ball gaps and player pace + restart possessor"
                     " + confirmed control retained until the other team's control",
        "raw_ball_coverage_graded": float(np.isfinite(data.raw_tracking.ball[graded]).all(axis=1).mean()),
        "cleaned_ball_coverage_graded": float(np.isfinite(data.ball.loc[graded, ["x", "y"]]).all(axis=1).mean()),
        "ball_sources_graded": {str(k): int(v) for k, v in features.loc[graded, "ball_source"].value_counts().items()},
        "long_tactical_gaps": len(gaps),
        "long_gap_verdicts": {str(k): int(v) for k, v in gaps["verdict"].value_counts().sort_index().items()},
        "long_gap_dead_reasons": {str(k): int(v) for k, v in gaps.loc[gaps["verdict"] == "dead", "why"]
                                  .str.replace(r" \(.*\)", "", regex=True).value_counts().sort_index().items()},
        "placed_ball_restarts": len(prepared.restarts),
        "live_stretches": len(prepared.seeds),
        "restart_team_found_by": {str(k): int(v) for k, v in prepared.seeds["how"].value_counts().sort_index().items()},
        "smart_tagging_set_pieces": len(prepared.set_pieces),
        "team_shape": shape,
        "dead_frames_graded": int(prepared.dead[graded].sum()),
        "broadcast_used": data.video_2 is not None,
        "broadcast_offsets": data.video_2.offsets if data.video_2 is not None else {},
        "smart_tagging_file": str(prepared.smart_tagging) if prepared.smart_tagging else None,
    }


def run(config: dict, output: Path, mode="current", evaluate=False, bootstrap=1000) -> dict:
    started = time.perf_counter()
    labels, diagnostics = predict(config, mode)
    table = pd.DataFrame({"frame": np.arange(len(labels)), "label": labels})
    error = official.validate_submission(table, GAMES[config["data"]["game"]]["n_video_frames"])
    if error:
        raise ValueError(f"Invalid prediction: {error}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(output), labels)
    error = official.validate_submission(official.read_label_csv(output), len(labels))
    if error:
        raise ValueError(f"Invalid saved submission: {error}")
    graded = graded_mask(len(labels), config)
    input_paths = list(atd_paths(config, "tactical"))
    if diagnostics.get("broadcast_used"):
        input_paths += list(atd_paths(config, "broadcast"))
        input_paths.append(_find_file(get_data_dir(config),
                                     f"{config['data']['game']}_broadcast_homography.csv", "broadcast", None))
    if diagnostics.get("smart_tagging_file"):
        input_paths.append(Path(diagnostics["smart_tagging_file"]))
    report = {
        "mode": mode,
        "submission": str(output.resolve()),
        "submission_sha256": sha256(output),
        "frames": len(labels),
        "label_counts_graded": {label: int((labels[graded] == label).sum()) for label in "HADX"},
        "config": config,
        "inputs_sha256": {str(p): sha256(p) for p in input_paths},
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p)
                        for p in [Path(__file__).resolve(), ROOT / "games.py", ROOT / "starter/baseline.py",
                                  ROOT / "evaluation/possession_metrics.py",
                                  *sorted((ROOT / "development/src").glob("*.py"))]},
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "pandas", "kloppy", "matplotlib", "PyYAML")},
        "diagnostics": diagnostics,
        "prediction_seconds": round(time.perf_counter() - started, 3),
    }
    print(f"Validated {len(labels):,} rows: {output}", flush=True)
    report_path = output.with_suffix(".report.json")
    # Persist the exact method/input/output fingerprint before any public label is read.
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if evaluate:
        # Deliberately fixed to the public slice; never search for private ground truth.
        truth_path = ROOT / "evaluation" / "gt" / f"{config['data']['game']}_possession_gt_public.csv"
        truth = official.read_label_csv(truth_path)
        error = official.validate_submission(truth, len(labels))
        if error:
            raise ValueError(f"Invalid public ground truth: {error}")
        scores = official.evaluate(official.label_codes(truth), official.label_codes(labels),
                                   bootstrap_n=bootstrap)
        report["public_evaluation"] = scores
        report["evaluation_scope"] = "Public five minutes only; correctness check, not tuning or full-match ranking."
        print(f"Public check: spellF1={scores['spell_f1_mean']:.4f}, "
              f"macroF1={scores['macro_f1']:.4f}, "
              f"possessions={scores['n_pred_spells']} (truth {scores['n_gt_spells']})")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, help="override configuration and MTAH_DATA_DIR")
    parser.add_argument("--mode", choices=("current", "baseline"), default="current")
    parser.add_argument("--output", type=Path, help="default: outputs/<mode>/submission.csv")
    parser.add_argument("--evaluate", action="store_true", help="run the official public scorer after prediction")
    parser.add_argument("--bootstrap", type=int, default=1000, help="public confidence-interval draws; 0 disables")
    args = parser.parse_args(argv)
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    if args.data_dir is not None:
        os.environ["MTAH_DATA_DIR"] = str(args.data_dir.resolve())
    config = load_config(args.config)
    output = args.output or ROOT / "outputs" / args.mode / "submission.csv"
    try:
        return run(config, output, args.mode, args.evaluate, args.bootstrap)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Pipeline failed: {error}\nFor downloaded ZIPs, run python prepare_data.py first.\n")


if __name__ == "__main__":
    main()
