# comp-psych

## Purpose

`comp_psych` is a Python analysis package for a longitudinal computational-psychiatry
study, for two of the behavioral decision-making tasks — **gain_loss** (a
probabilistic reward/loss-avoidance task) and **explore_exploit** (a multi-armed
"casino" bandit task) — alongside a battery of self-report **questionnaires**
(DASS-21, OCI-R, SPQ-Brief, and demographics), repeated across multiple sessions
("spells") per participant. Task and questionnaire data are collected via a web app
and stored in Firebase/Firestore.

This package exports that raw data out of Firestore, loads it into tidy pandas
DataFrames, and runs behavioral and computational-modeling analyses on it (win-stay/
lose-shift behavior, choice-outcome regressions, probability-reversal transitions,
Stan-fitted reinforcement-learning models, and questionnaire score/subscale/item
analyses, including correlations against task behavior).

The package name is `comp_psych` (snake_case, since Python imports can't contain 
hyphens); the repo/directory name is `comp-psych` (kebab-case).

## Directory structure

```
comp-psych/
├── requirements.txt
├── pyproject.toml
├── data/                         gitignored — see "Data requirements" below
└── comp_psych/
    ├── core/                     shared utilities
    │   ├── env.py                 paths & credentials (DATA_ROOT, TASK_ROOT, FB_CREDENTIALS_FILE, ...)
    │   ├── selection.py            subselect_data() — shared trial/subject filter
    │   ├── modeling.py             compute_map_estimates(), load_model_parameters()
    │   └── fb_export_all.py        CLI: run all 3 exports for one spell
    ├── gain_loss/                 task domain
    │   ├── config.py               task-specific paths (derived from core.env)
    │   ├── load.py                 load_gain_loss_data()
    │   ├── modeling.py             get_param_names()
    │   ├── export/fb_export.py     Firestore -> raw JSON -> parquet
    │   └── analyses/
    │       ├── behavior/           win-stay/lose-shift, performance, transitions, bonuses
    │       ├── comparison/         behavior/model params vs. questionnaire subscales
    │       └── modeling/           fit_stan_model.py + stan_models/*.stan
    ├── explore_exploit/            same shape as gain_loss (no modeling/ subfolder)
    └── questionnaires/             same shape, but load.py handles questionnaire CSVs
        ├── load.py                  load_scores(), load_subscales(), load_questions()
        ├── export/fb_export_questionnaires.py
        └── analyses/                distributions, correlations, cross-session consistency
```

## Installation

- **Python**: developed against **3.14** (a conda env named `comp_psych_env`); no
  hard floor is enforced, but the codebase uses no version-specific syntax beyond
  standard modern Python.
- Install the package in editable mode from the repo root — this also pulls in its
  declared dependencies (unpinned, `>=` floors):

  ```bash
  pip install -e .
  ```

  `pyproject.toml` lives at the repo root.

  For a reproducible environment matching the one this was developed/tested in, use
  the pinned versions in `requirements.txt` instead:

  ```bash
  pip install -e . --no-deps
  pip install -r requirements.txt
  ```

- **Stan toolchain**: `comp_psych/gain_loss/analyses/modeling/fit_stan_model.py` uses
  [`cmdstanpy`](https://mc-stan.org/cmdstanpy/), which needs a working CmdStan
  installation (`python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"` if you
  don't have one). Compiled Stan binaries (`.exe`) are **not** checked into the repo
  (gitignored) — CmdStanModel compiles the `.stan` source on demand.

## Data requirements

Data lives outside the repo, at paths defined in
[comp_psych/core/env.py](comp_psych/core/env.py) — `DATA_ROOT`, `TASK_ROOT`,
`DEMOGRAPHICS_DIR`, `QUESTIONNAIRE_DIR`, `FB_CREDENTIALS_FILE`. **These are hardcoded
absolute paths tied to the original development machine** (see Gotchas below) — update
them before running on a new machine.

- **Firebase credentials**: a service-account JSON key at `FB_CREDENTIALS_FILE`
  (gitignored, not included in the repo — request a fresh key from the Firebase
  project owner). The `fb_export_questionnaires.py` CLI can also read
  `GOOGLE_APPLICATION_CREDENTIALS` or auto-detect a `*firebase*.json` file in the
  current directory.
- **Inclusion lists**: `DEMOGRAPHICS_DIR/inclusion_list_group<X>.csv`, one column
  `participant_id`, listing approved/complete participants for group X. Both
  behavioral and questionnaire exports silently skip participants not in this file
  (pass `--all` to the questionnaire CLI, or bypass via `fb_export_all`'s check
  raising if the file is missing).
- **Task design files**: `TASK_ROOT/<task>/.../designs/csv/{practice,set}<N>.csv`,
  referenced by each trial's `designNo`. Required for the export step to attach
  win-probability/correct-choice fields to raw trial data.
- **Exported data layout** (written under `DATA_ROOT/<task>/`):
  - `<task>/raw_data/<participant_id>.json` — per-participant raw trial documents.
  - `<task>/data/all_data.parquet` — consolidated, design-joined trial data.
  - `<task>/stan_fits/<model_name>/<participant_id>/` — Stan fit outputs
    (`param_estimates.npz`, `samples.parquet`, `summary.csv`).
  - `questionnaire_data/q_<questionnaire>_<spell>.csv` — long-format questionnaire exports.
  - `<task>/bonuses/<spell>_<task>_bonuses.csv` — computed participant bonuses.

## How to run

All commands run as modules from the repo root, so `comp_psych.*` imports resolve.

1. **Export everything for one spell** (recommended entry point):

   ```bash
   python -m comp_psych.core.fb_export_all --spell s1_groupA [--force-refresh]
   ```

   Runs the questionnaire export, then the explore_exploit export, then the
   gain_loss export. Raises `FileNotFoundError` up front if no inclusion list exists
   for the spell's group. `--force-refresh` re-downloads raw task data even if local
   files already exist.

2. **Or export one task/questionnaire set at a time**:

   ```bash
   python -m comp_psych.questionnaires.export.fb_export_questionnaires --spell s1_groupA [--questionnaire dass21] [--all]
   python -m comp_psych.explore_exploit.export.fb_export
   python -m comp_psych.gain_loss.export.fb_export
   ```

3. **Fit a Stan model** (gain_loss only; requires `all_data.parquet` already exported):

   ```bash
   python -m comp_psych.gain_loss.analyses.modeling.fit_stan_model
   ```

   Edit the `__main__` block's `subselect`/`model_name`/`force_rerun` args to change
   what's fit — this and the analysis scripts below aren't argparse CLIs, they're
   run-as-scripts with parameters set at the bottom of the file.

4. **Run a behavioral or questionnaire analysis**, e.g.:

   ```bash
   python -m comp_psych.gain_loss.analyses.behavior.analyze_wsls
   python -m comp_psych.explore_exploit.analyses.behavior.choice_outcome_regression
   python -m comp_psych.gain_loss.analyses.comparison.compare_parameters_to_questionnaires
   python -m comp_psych.questionnaires.analyses.analyze_questionnaire_correlations
   ```

   Each analysis module's public function (e.g. `analyze_wsls(subselect=..., plot_flag=...)`)
   can also be imported and called directly from a notebook or script.

## Expected outputs

- Export scripts write to `DATA_ROOT/<task>/...` as described above.
- Analysis functions return `pandas.DataFrame`s (import and call them directly to get
  data back programmatically) and, when `plot_flag=True` (the default in most),
  display matplotlib figures — nothing is saved to disk automatically.
- `determine_bonuses` is the one analysis that writes output (`bonuses/*.csv`), since
  it's used operationally to pay participants.
- Stan fits write `param_estimates.npz` / `samples.parquet` / `summary.csv` per
  participant under `stan_fits/<model_name>/`.

## Key assumptions, exclusion criteria, and known gotchas

- **`core.selection.subselect_data`** is the single chokepoint for trial/subject
  filtering (drop-trial removal, practice-trial removal, session-count filtering,
  group/participant filtering) — extend it there rather than reimplementing per-task filters.
- **Session parsing**: session/group IDs are parsed from `session_id`/`spellId`
  strings like `s3_groupA` (session = 3, group = 'A') throughout the codebase.
- **Inclusion lists gate both exports**: a participant absent from
  `inclusion_list_group<X>.csv` is silently skipped during export (not an error) —
  pass `--all` to the questionnaire CLI to bypass.
- **Hardcoded absolute paths**: `comp_psych/core/env.py` hardcodes `DATA_ROOT`,
  `TASK_ROOT`, and `FB_CREDENTIALS_FILE` to the original development machine. If you
  move the repo or run on a new machine, **update these paths first** — nothing else
  in the codebase should need to change, since every task/questionnaire module derives
  its paths from these.
- **Firebase credentials**: a live service-account key exists at
  `comp_psych/core/fb_credentials.json` on the original development machine. It's
  correctly gitignored.

## Contact / authorship / citation

- Author: cgrossman (see module docstrings; contact info not otherwise recorded in-repo).
