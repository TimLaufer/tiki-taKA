# Runnable possession framework

A deterministic path from the supplied match data to a validated submission of all
199,319 tactical frames. `run_pipeline.py` (repository root) is the prediction and
evaluation command; `development/main.py` remains a diagnostic viewer.

## The submission

The submitted labels are written by one command, run in `development/` (data found as described
in `config/config.yaml`, or via `MTAH_DATA_DIR`):

```bash
cd development
.venv/bin/python -m src.baseline        # -> development/outputs/BAR-ATH_baseline.csv
cd ..
development/.venv/bin/python evaluation/validate_submission.py development/outputs/BAR-ATH_baseline.csv
```

Submitted configuration (`development/config/config.yaml`): `baseline.dead: gap` (dead ball from
the long tactical ball gaps judged by the players' pace, `src/stoppages.py`, with the restart
recognition of `src/restarts.py` and the crowd episodes of `src/dead_ball.py` as cues) and
`baseline.possession: stefan` (`src/possession_model.py`: control within 3 m and 1 m nearer than
every opponent, a change after 0.6 s), team shape off. This is the chain of `run_pipeline.py`.

Further configurations, built and compared on Metrica's labelled sample matches
(`python -m src.benchmark --variants`) but not submitted: `baseline.dead: evidence` (per stoppage
the logic with the richer data there; level with `gap` on the sample matches), `restart`
(`src/segments.py`), `union`, `combined`. The technical report is `development/report/report.tex`.

## Setup and run

Run from the repository root. The checks below used Python 3.14.6 on Linux; every
installed dependency is pinned in `requirements-dev.lock`. The dependency-light scorer
requirements are unchanged.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python prepare_data.py          # unpack the downloaded ZIPs into data/
.venv/bin/python run_pipeline.py          # predict + validate (never reads labels)
.venv/bin/python run_pipeline.py --evaluate   # ... then the official metrics on the public 5 minutes
```

`prepare_data.py` extracts the tracking, homography and tagging files from the download
ZIPs into `data/tactical/` and `data/broadcast/`, verifies files that already exist, and
links complete `.mp4` files (partial downloads are ignored). Videos are not used. For
assets elsewhere, set an absolute `MTAH_DATA_DIR` or pass `--data-dir`. Configuration
paths are relative to `development/`; CLI output paths to the working directory.

Outputs (all git-ignored):

- `outputs/current/submission.csv` — one label per tactical frame, checked with the
  official validation function before and after writing.
- `outputs/current/submission.report.json` — configuration, SHA-256 of every input and
  source file, package versions, label-free diagnostics, and (with `--evaluate`) the
  public scores. The fingerprint is written before any label is read.
- `competition_data/cache/` — parsed tracking (`.npz`) and the computed broadcast
  alignment (`.json`). Both are keyed on input contents, parser/algorithm source and
  settings; a cold run (about 3 minutes, mostly synchronization) and a warm run (about
  15 s) produce byte-identical submissions.

Useful commands:

```bash
.venv/bin/python run_pipeline.py --mode baseline --evaluate      # published baseline_smooth50, exactly
.venv/bin/python evaluation/validate_submission.py outputs/current/submission.csv
.venv/bin/python evaluation/run_evaluation.py --results outputs/current --out outputs/current/official --bootstrap 1000
.venv/bin/python -m pytest -c development/pyproject.toml development/tests -q   # synthetic, no match data
cd development && ../.venv/bin/python -m src.benchmark --variants   # validation on other matches (below)
```

## Method

No model is trained on labels. Every threshold is a physical or football quantity, follows
from label-free analysis of the inputs below, or was checked on labelled data of *other*
matches (see *Validation on other matches*). The optional team-shape stage fits a small
model per match on the pipeline's own pseudo-labels; it is off for the submission.

**Inputs.** Tactical tracking (players and ball; frame `i` is ATD `frame_id - 1`), the
published phase table, broadcast tracking and homography (aligned algorithmically), and
Smart Tagging set pieces as fallible corroboration. Videos are not used.

**Preprocessing.** The loader rejects missing, repeated or unordered frames; incomplete
coordinates are missing, never invented. Lost player tracks are bridged for up to 10 s
where the player could have run (7 m/s + 2 m); missing players are completed from the
broadcast only where its players match the tactical ones; one goalkeeper per team is kept
in a fixed slot. Single false ball detections (jumps above 40 m/s) are dropped.

**Synchronization.** One offset per half from 400 evenly spaced broadcast frames, each
matched to the tactical frame whose same-team player layout fits best over 0, 1 and 2 s.
Accepted only with at least 10 matches, 85% of them within 0.5 s of the median, and a
clock slope within 0.0005 of 1 (this match: 99.5% and 98.4% agreement, slopes 0.99998).
The broadcast ball fills tactical gaps longer than 5 s only where the broadcast shows the
moment live (homography valid, reprojection error at most 3, players matching within
3 m RMS). No offset is typed in.

**What the data shows about dead ball** (unlabelled input analysis, full match). The
tactical ball track has practically no gaps between 1 s and 5 s; gaps of 5 s or more
cover about 40% of the graded frames, close to the out-of-play share of a professional
match. Inside them the players' median speed is about 1 m/s (87% of those frames at
walking pace) against 1.6 m/s while the ball is seen; the broadcast has a usable
homography in 46% of the walking gaps but 90% of the others (television cuts to replays
at stoppages); and Smart Tagging set pieces open shortly before such gaps end.

**Stoppages (D).** Each tactical ball gap of at least 5 s is judged on the tactical ball
itself (a broadcast ball glimpsed during a stoppage must not hide it):

- *keeper hold* — the same goalkeeper controls the ball where it vanishes and where it
  reappears, and it was not last seen at a line: the ball is in play (Law 12); his team.
- *dead* — the ball was last seen out of play; or the players' median speed over the gap
  is at most walking pace (1.5 m/s); or at most 2.0 m/s with one more cue: last seen at
  a line, reappearing at a restart spot, a crowd episode, or a tagged set piece around the
  restart (a tag never decides alone).
- *live* — otherwise: play continues unseen.

Also dead: the observed ball more than 1.5 m beyond a line for at least 0.4 s; the
waiting before a detected placed-ball restart; graded frames before a half's kickoff
(Law 8; the kickoff is where the published phase table starts the match clock). Joining
short walking-pace visible interruptions into the stoppages around them is available
(`stoppages.bridge_s`) but off: on the labelled matches they were mostly real play.

**Restarts.** Each live stretch starts with a restart. Its team is that of a detected
placed-ball restart; else, from the first ball seen within 1 s: at a corner flag the
attacking team, in a goal area the defending team (the laws), elsewhere the player in
control at the ball (the taker); else the first team whose control is later confirmed.

**Possession (H/A).** Control evidence on a frame: the observed (not interpolated) ball
at most 12 m/s, a player within 3 m, and more than 1 m nearer than every opponent. A
change needs 0.6 s of uninterrupted evidence and is dated back to its first frame. The
owner keeps the ball through passes, loose and contested balls, deflections and
occlusions of any length, until the other team's control is confirmed or play stops. No
smoothing is applied. `X` remains only where a live stretch has no known team yet; the
scorer repeats the previous label there.

**Optional stage: team shape** (`team_shape.enabled`, off). The team in possession spreads
out, advances and runs less than the team chasing it. A logistic model of these three
home-minus-away differences is fitted per match on the frames where the owning team's
player has the ball at his feet (pseudo-labels; in-sample AUC about 0.88 here). Shape
follows a change of possession about 3 s later, so it only judges: a possession won in
open play that lasts at least 5 s but whose part from 3 s on the model gives its winner a
probability below 0.2 is returned (veto), and unattributed frames take a clear shape's
team (fill). On this match it would veto 7 possessions (0.8% of graded frames); on the
labelled matches it gave no consistent gain, so it stays off.

## Validation on other matches

[Metrica's open sample data](https://github.com/metrica-sports/sample-data) (three
anonymised matches; same provider, formats and 25 fps; events annotated by Metrica) is the
labelled data, allowed by `RULES.md` because it is about other games. Download it into the
git-ignored `data/external/metrica_sample/` (the repository layout as is); `src/external.py`
reads it, and `python -m src.benchmark --variants` (in `development/`) runs the unchanged
pipeline on each match and scores it with the official metrics. Options: `--clean`,
`--labels controlled`.

- **Reference labels** follow Metrica's event definitions: the team of the latest on-ball
  event has the ball (*touch*); dead from a ball out, goal, offside, foul or referee stop
  until the next on-ball event. A *controlled* variant applies the proxies of
  `CHALLENGE.md`: a spell won in open play counts only if it lasts 2 s or contains a
  completed pass.
- **Realistic noise.** The sample tracking is corrected: its ball lies about 0.1 m from
  its owner, against a median of 1.8 m between ball and nearest player in the challenge's
  live play. The benchmark therefore adds a smooth ball error (1.15 m per axis, 3 s time
  constant) by default, which reproduces that distance and the ball speeds, both measured
  on the challenge input without labels.
- **Findings** (spellF1, mean of three matches):

| | clean, touch | clean, controlled | noisy, touch | noisy, controlled |
|---|---|---|---|---|
| previous settings (0.4 s, bridging) | 0.548 | 0.593 | 0.512 | 0.575 |
| **submitted (0.6 s, no bridging)** | **0.569** | **0.659** | **0.527** | **0.630** |
| submitted + team shape | 0.563 | 0.652 | 0.529 | 0.633 |
| oracle dead ball / oracle team | 0.608 / 0.919 | 0.711 / 0.900 | 0.554 / 0.933 | 0.663 / 0.924 |

The submitted settings beat the previous ones on all twelve match-condition pairs. Under
the realistic noise its 3 m radius, 1 m margin and 0.6 s confirmation are best: smaller
radii win only on the corrected data (1.5 m would lose 0.10 with noise), 0.2 s and 1.2 s
lose everywhere, the walking threshold is insensitive, and dropping restart teams costs
0.04-0.05. The oracles (one component replaced by the truth) locate the remaining error:
perfect stoppages would add 0.03-0.05, perfect team calls in live play 0.24-0.41.

## Validation

- 187 synthetic tests (no licensed data) cover loading and frame contracts, cache
  invalidation (data, parser and corruption), synchronization gates, gap verdicts,
  keeper holds, Smart Tagging parsing and fallibility, pre-kickoff and half boundaries,
  interruption bridging, restart teams, retention, the external-data loader and labels,
  the noise model, team shape, and end-to-end prediction.
- The official validator accepts the file; warm, repeated and cold-cache runs are
  byte-identical (SHA-256 `9664234e607003b89049400e8c3d152631e609ff2125ba37769bef61287cf956`).
- Label-free full-match diagnostics: 41.9% of graded frames dead in 143 stoppages (about
  61 min of play in 104), 348 predicted possessions, 230 graded `X` frames, the two
  kickoffs taken by opposite teams.

## Public correctness check (descriptive only)

Run once per version, after its settings and submission hash were fixed, with the
unchanged official scorer (`--bootstrap 1000`). The public window is five minutes with 17
possessions; at this size the scorer itself reports spellF1 gaps below about 0.30 as noise.
The numbers do not establish full-match quality, and no model change followed them.

| | public five minutes | bootstrap 5–95% |
|---|---|---|
| spellF1 (ranking metric) | 0.526 | 0.366 – 0.663 |
| macroF1 | 0.737 (F1 H 0.857, A 0.802, D 0.552) | 0.557 – 0.858 |
| alive/dead F1 | 0.916 | |
| possessions (truth) | 18 (17), mean 14.6 s (14.2 s) | |
| transition F1 at ±25 frames | 0.286 | |

The previous settings scored 0.544 spellF1 and 0.739 macroF1 here. The difference is far
inside the noise band, and by the rules this window may not choose between versions: the
choice rests on the three labelled matches above.

For context, the published baselines score 0.33 (`smooth50`) and 0.42 (`smooth125`)
spellF1 on the same window; the difference lies inside the noise band. Most frame errors
in the window are dead-ball frames labelled live (D recall 0.45), consistent with
stoppages that begin while the ball is still visible.

## Limitations

- Dead ball is inferred from when the tactical tracker loses the ball and how the players
  move; stoppages with the ball in view start late (at its disappearance), and slow live
  play the camera cannot follow can be mistaken for a stoppage.
- Control is a proximity heuristic: possession changes in live play are the main error
  (oracle above). Close duels, one-touch play and long passes between opponents produce
  wrong or late changes, and the best confirmation time depends on how control is defined.
- The validation labels are event-based proxies on three matches with corrected tracking
  and simulated ball error, not the challenge's own definition or noise.
- Restart teams from the nearest player fail when the ball reappears already in flight
  (then the first confirmed control decides) or when the wrong player stands nearest.
- One broadcast offset per half; local player checks reject inconsistent moments rather
  than modelling cuts. Smart Tagging team tags and phases are not used.
- Full-match quality cannot be established from the public five-minute slice.

The compact method description is `development/docs/method.tex`. To compile it without
writing build products next to the source:

```bash
mkdir -p outputs/method
pdflatex -interaction=nonstopmode -halt-on-error -output-directory outputs/method development/docs/method.tex
```
