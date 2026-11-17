"""Persistent controlled possession between stoppages.

Within each live stretch (graded, ball in play, never across a half or a stoppage):

* Play restarts with the restarting team (feature restart_team at the stretch's first frame,
  from src.stoppages), when it is known.
* Control evidence on a frame: the observed (not interpolated) ball, at most
  control_max_speed_ms fast where its speed is known, with a player of one team within
  control_radius_m and more than control_margin_m nearer than every opponent.
* A team gains possession after confirmation_s of uninterrupted evidence. The prediction is
  offline, so the change is dated back to the first supporting frame.
* Otherwise the owner keeps the ball - through passes, loose and contested balls, deflections,
  brief touches and occlusions of any length - until the other team's control is confirmed or
  play stops (CHALLENGE.md: possession stays with the last team that had it).
* hold_team: a goalkeeper holds the unseen ball, so his team has it.
* A stretch whose restarting team is unknown belongs, from its start, to the first team whose
  control is confirmed in it; until then, and if that never happens, the label is X.
"""
import numpy as np

TEAMS = ("H", "A")


class PossessionModel:
    def __init__(self, config: dict):
        self.config = config.get("model") or {}
        self.fps = config.get("fps", 25)
        self.radius = self.config.get("control_radius_m", 3.0)
        self.margin = self.config.get("control_margin_m", 1.0)
        self.max_speed = self.config.get("control_max_speed_ms", 12.0)
        self.confirm_s = self.config.get("confirmation_s", 0.4)
        values = (self.fps, self.radius, self.margin, self.max_speed, self.confirm_s)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Control settings and fps must be finite and positive")

    def fit(self, features, labels=None):
        """This rule-based model has no learned parameters."""
        if labels is not None:
            raise ValueError("This model does not train on labels")
        return self

    def evidence(self, features) -> np.ndarray:
        """Per frame: the team with control evidence ("H", "A") or ""."""
        home = np.nan_to_num(features["nearest_H_m"].to_numpy(dtype=float), nan=np.inf,
                             posinf=np.inf, neginf=np.inf)
        away = np.nan_to_num(features["nearest_A_m"].to_numpy(dtype=float), nan=np.inf,
                             posinf=np.inf, neginf=np.inf)
        observed = features["ball_source"].to_numpy() == "observed"
        speed = features["ball_speed_ms"].to_numpy(dtype=float)
        controllable = observed & (~np.isfinite(speed) | (speed <= self.max_speed))
        evidence = np.full(len(features), "", dtype="<U1")
        evidence[controllable & (home <= self.radius) & (home + self.margin < away)] = "H"
        evidence[controllable & (away <= self.radius) & (away + self.margin < home)] = "A"
        return evidence

    def predict(self, features):
        n = len(features)
        labels = np.full(n, "X", dtype="<U1")
        dead = features["dead"].to_numpy(dtype=bool)
        graded = features["graded"].to_numpy(dtype=bool)
        halves = features["half"].to_numpy()
        seeds = features["restart_team"].to_numpy() if "restart_team" in features else np.full(n, "")
        holds = features["hold_team"].to_numpy() if "hold_team" in features else np.full(n, "")
        evidence = self.evidence(features)
        needed = max(1, round(self.confirm_s * self.fps))
        owner, candidate, count, first, start = "", "", 0, 0, 0
        for i in range(n):
            if not graded[i]:
                continue
            if dead[i]:
                labels[i] = "D"
                continue
            if i == 0 or not graded[i - 1] or dead[i - 1] or halves[i] != halves[i - 1]:
                # A live stretch begins: play restarts, possibly with a known team.
                owner = seeds[i] if seeds[i] in TEAMS else ""
                candidate, count, start = "", 0, i
            if holds[i] in TEAMS:
                keeper = holds[i]
                if owner != keeper:
                    # The catch: from the keeper's evidence just before, or the stretch start.
                    since = start if not owner else (first if candidate == keeper else i)
                    labels[since:i] = keeper
                    owner = keeper
                candidate, count = "", 0
                labels[i] = keeper
                continue
            team = evidence[i]
            if team:
                if team == candidate:
                    count += 1
                else:
                    candidate, count, first = team, 1, i
                if count == needed and team != owner:
                    # Offline confirmation dates the change back to its first supporting frame;
                    # the first possessor of a stretch without a known restart has it from the start.
                    labels[(first if owner else start) : i + 1] = team
                    owner = team
            else:
                candidate, count = "", 0
            labels[i] = owner or "X"
        return labels
