"""Benchmark the possession model on labelled matches from other games (src.external).

    python -m src.benchmark                      # the configured model on Metrica's sample games
    python -m src.benchmark --variants           # ... every variant in VARIANTS, and the oracles
    python -m src.benchmark --clean              # the corrected sample tracking as delivered
    python -m src.benchmark --labels controlled  # CHALLENGE.md's control proxies on the labels

Each game runs through the unchanged prediction pipeline (no broadcast, no Smart Tagging: the
sample games have neither) and is scored by the official metric library against labels derived
from Metrica's human-annotated events (src.external). By default the ball gets the smooth
challenge-like error of src.external.CHALLENGE_LIKE_NOISE, because the sample tracking is
corrected and far cleaner than the challenge input. Nothing about the challenge match is read
here, so this is where model choices may legitimately be compared (RULES.md: labelled data of
other games). The data: https://github.com/metrica-sports/sample-data, in data/external/metrica_sample.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import PROJECT_ROOT, load_config
from src.external import CHALLENGE_LIKE_NOISE, controlled_labels, degrade_ball, load_game
from src.baseline import decode, prepare
from src.pipeline import load_match

REPO = PROJECT_ROOT.parent
DEFAULT_ROOT = REPO / "data" / "external" / "metrica_sample"
GAMES = (1, 2, 3)
NO_TAGS = pd.DataFrame(columns=["first", "last", "type", "team"])
# name -> configuration overrides ("section.key": value); "configured" is config.yaml as is.
VARIANTS = {
    "configured": {},
    "previous (0.4 s, bridging)": {"model.confirmation_s": 0.4, "stoppages.bridge_s": 5.0},
    "confirm 0.2 s": {"model.confirmation_s": 0.2},
    "confirm 0.4 s": {"model.confirmation_s": 0.4},
    "confirm 0.8 s": {"model.confirmation_s": 0.8},
    "confirm 1.2 s": {"model.confirmation_s": 1.2},
    "radius 1.5 m": {"model.control_radius_m": 1.5},
    "radius 2 m": {"model.control_radius_m": 2.0},
    "radius 2.5 m": {"model.control_radius_m": 2.5},
    "radius 4 m": {"model.control_radius_m": 4.0},
    "margin 0.5 m": {"model.control_margin_m": 0.5},
    "margin 2 m": {"model.control_margin_m": 2.0},
    "walking 1.2 m/s": {"stoppages.walking_speed_ms": 1.2},
    "walking 1.8 m/s": {"stoppages.walking_speed_ms": 1.8},
    "bridging 5 s": {"stoppages.bridge_s": 5.0},
    "no restart team": {"stoppages.seed_window_s": 0.0},
    "team shape": {"team_shape.enabled": True},
    "team shape, veto 0.1": {"team_shape.enabled": True, "team_shape.veto_below": 0.1},
    "team shape, veto 0.3": {"team_shape.enabled": True, "team_shape.veto_below": 0.3},
    # Tim's dead balls (src.segments: restarts, where play stopped, catches) under the same model
    "restart-based D": {"baseline.dead": "restart"},
    "restart-based D, no catches": {"baseline.dead": "restart", "segments.catches": False},
    "restart-based D, confirm 0.4 s": {"baseline.dead": "restart", "model.confirmation_s": 0.4},
    "restart-based D, confirm 0.8 s": {"baseline.dead": "restart", "model.confirmation_s": 0.8},
    "restart-based D, radius 2 m": {"baseline.dead": "restart", "model.control_radius_m": 2.0},
    "restart-based D, radius 4 m": {"baseline.dead": "restart", "model.control_radius_m": 4.0},
    "restart-based D, margin 0.5 m": {"baseline.dead": "restart", "model.control_margin_m": 0.5},
    "restart-based D, margin 2 m": {"baseline.dead": "restart", "model.control_margin_m": 2.0},
    # one building block of the restart-based logic switched off at a time
    "restart-based D, no stoppage start": {"baseline.dead": "restart", "segments.max_lookback_s": 0.0},
    "restart-based D, no duel check": {"baseline.dead": "restart", "restarts.duel_m": None},
    "restart-based D, no penalty scene": {"baseline.dead": "restart", "restarts.penalty_box_max": 99,
                                          "restarts.penalty_spread_ms": 99.0},
    "restart-based D, no thrower": {"baseline.dead": "restart", "restarts.thrower_behind_m": 99.0},
    # both logics together: dead where either says so, our restart teams
    "union of both D": {"baseline.dead": "union"},
    # Stefan's dead balls with Tim's added, without the parts that cost on these games
    "union, no stoppage start": {"baseline.dead": "union", "segments.max_lookback_s": 0.0},
    "union, no stoppage start, no catches": {"baseline.dead": "union", "segments.max_lookback_s": 0.0,
                                             "segments.catches": False},
    "union, start only at a ball out": {"baseline.dead": "union", "segments.slow_speed_ms": -1.0},
    "restart-based D, start only at a ball out": {"baseline.dead": "restart", "segments.slow_speed_ms": -1.0},
    # both judged per kind of restart (src.dead_ball_compare)
    "combined per kind": {"baseline.dead": "combined"},
    # per stoppage the logic with the richer data there (src.baseline.choose_by_evidence)
    "chosen by evidence": {"baseline.dead": "evidence"},
}
# Every variant runs the submission's code (src.baseline.prepare / decode); "baseline.dead" picks the
# dead balls: "gap" (Stefan's, src.stoppages, chosen), "restart" (Tim's, src.segments) or "union".
# Diagnostics, not candidate models: one component replaced by the truth, to see what limits the score.
ORACLES = ("oracle stoppages", "oracle team")


def _metrics():
    sys.path.insert(0, str(REPO / "evaluation"))
    import possession_metrics

    return possession_metrics


def with_overrides(config: dict, overrides: dict) -> dict:
    out = copy.deepcopy(config)
    for key, value in overrides.items():
        section, name = key.split(".", 1)
        out.setdefault(section, {})[name] = value
    return out


class Game:
    """One sample game, loaded once; prepared again only for preprocessing overrides."""

    def __init__(self, root: Path, number: int, config: dict, noise: tuple[float, float] | None = None,
                 labels: str = "touch"):
        self.number = number
        self.tracking, touch, self.config, self.events = load_game(root, number, config)
        # touch: Metrica's event definitions as they are; controlled: CHALLENGE.md's control proxies
        self.truth = touch if labels == "touch" else controlled_labels(touch, self.events)
        if noise is not None:  # challenge-like ball positions; the seed makes it reproducible
            self.tracking = degrade_ball(self.tracking, *noise, seed=number)
        self.keepers = {}
        if number == 3:  # EPTS metadata marks the goalkeepers, as in the challenge export
            from src.goalkeeper import keeper_slots
            self.keepers = keeper_slots(Path(root) / "data" / "Sample_Game_3" / "Sample_Game_3_metadata.xml",
                                        self.tracking)
        self._prepared = {}

    def prepare(self, config: dict):
        key = repr(sorted((k, repr(v)) for k, v in config.items() if k in ("stoppages", "dead_ball", "restarts",
                                                                           "ball", "players", "goalkeeper",
                                                                           "segments", "baseline")))
        if key not in self._prepared:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                    warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = load_match(config, use_video_2=False, raw=self.tracking, keepers=self.keepers)
                prepared, features = prepare(data, config, set_pieces=NO_TAGS)  # as the submission does
                self._prepared[key] = (prepared, features)
        return self._prepared[key]

    def predict(self, overrides: dict | None = None) -> np.ndarray:
        config = with_overrides(self.config, overrides or {})
        prepared, features = self.prepare(config)
        return self._decode(prepared, features, config)

    @staticmethod
    def _decode(prepared, features, config) -> np.ndarray:
        return decode(prepared, features, config)

    def oracle(self, kind: str) -> np.ndarray:
        """The configured model with the true stoppages, or with the true team on live frames."""
        prepared, features = self.prepare(self.config)
        if kind == "oracle team":
            labels = self._decode(prepared, features, self.config)
            fix = np.isin(labels, ["H", "A"]) & np.isin(self.truth, ["H", "A"])
            labels[fix] = self.truth[fix]
            return labels
        from src.stoppages import restart_seeds
        dead = (self.truth == "D")
        seeds = restart_seeds(dead, prepared.ball, prepared.tracking, prepared.restarts, self.config)
        features = features.copy()
        features["dead"] = dead
        restart = np.full(len(features), "", dtype="<U1")
        for seed in seeds.itertuples():
            if seed.team in ("H", "A"):
                restart[int(seed.start)] = seed.team
        features["restart_team"] = restart
        return self._decode(prepared, features, self.config)


def score(truth: np.ndarray, labels: np.ndarray) -> dict:
    pm = _metrics()
    result = pm.evaluate(pm.label_codes(truth), pm.label_codes(labels))
    return {"spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
            "alive/dead F1": result["alive_dead_f1"], "team acc": result["team_acc_both_alive"],
            "poss": result["n_pred_spells"], "true poss": result["n_gt_spells"]}


def run(root: Path = DEFAULT_ROOT, variants: dict | None = None, games=GAMES, oracles=(),
        noise: tuple[float, float] | None = None, labels: str = "touch") -> pd.DataFrame:
    config = load_config()
    loaded = [Game(root, number, config, noise, labels) for number in games]
    rows = []
    for name, overrides in (variants or {"configured": {}}).items():
        for game in loaded:
            rows.append({"variant": name, "game": game.number, **score(game.truth, game.predict(overrides))})
    for name in oracles:
        for game in loaded:
            rows.append({"variant": name, "game": game.number, **score(game.truth, game.oracle(name))})
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per variant: the mean over games, and the spellF1 of each game."""
    means = table.groupby("variant", sort=False).mean(numeric_only=True).drop(columns="game")
    per_game = table.pivot(index="variant", columns="game", values="spellF1").add_prefix("spellF1 g")
    return means.join(per_game.reindex(means.index)).round(4)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_ROOT, help="the metrica-sports/sample-data folder")
    parser.add_argument("--variants", action="store_true", help="also run every variant in VARIANTS")
    parser.add_argument("--clean", action="store_true",
                        help="score the corrected sample tracking as delivered (default: challenge-like ball noise)")
    parser.add_argument("--labels", choices=("touch", "controlled"), default="touch",
                        help="possession from every significant touch (Metrica events) or controlled (CHALLENGE.md)")
    args = parser.parse_args()
    table = run(args.data, VARIANTS if args.variants else None, oracles=ORACLES if args.variants else (),
                noise=None if args.clean else CHALLENGE_LIKE_NOISE, labels=args.labels)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary(table).to_string())


if __name__ == "__main__":
    main()
