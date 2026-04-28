# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository rename

This repo replaces the older `compPsych` repo. The naming convention is:

- Repo / directory name: `comp-psych` (kebab-case)
- Python package name: `comp_psych` (snake-case, since Python imports cannot contain hyphens)

Old code under `oLabAnalysis/comp_psych/...` and `compPsych/...` has been retired. When you see references to the old layout in comments, docstrings, or configuration, update them to the new layout.

Related infra changes that came with the rename:

- A repo-level `.gitignore` was added. It excludes `comp_psych/data/`, `comp_psych/core/fb_credentials.json`, any `**/fb_credentials.json`, and `*.credentials.json`. Do not commit data files or service-account JSON.
- Data paths in [comp_psych/core/env.py](comp_psych/core/env.py) were repointed at the new repo location (`DATA_ROOT`, `TASK_ROOT`, `DEMOGRAPHICS_DIR`, `QUESTIONNAIRE_DIR`, `FB_CREDENTIALS_FILE`). If you move the repo or run on a new machine, update these absolute paths — they are the single source of truth for data locations.
- A fresh Firebase service-account API key was generated and lives at [comp_psych/core/fb_credentials.json](comp_psych/core/fb_credentials.json) (gitignored). Anything that talks to Firestore reads `FB_CREDENTIALS_FILE` from `core.env`.

## Install

The package is defined by [comp_psych/pyproject.toml](comp_psych/pyproject.toml). Install in editable mode from the inner package directory so the `comp_psych.*` imports resolve:

```bash
pip install -e comp_psych
```

There is no `requirements.txt` at the repo root. Runtime deps (used in code): `pandas`, `numpy`, `scipy`, `firebase-admin`, `google-cloud-firestore`, `pyarrow` (for parquet).

## High-level architecture

The package is organized around **three task domains** plus a shared core. Each task domain follows the same internal shape, which is the most important thing to understand before editing:

```
comp_psych/
  core/                       shared utilities (env, selection, modeling helpers, credentials)
  gain_loss/                  task domain
    config.py                 task-specific paths, derived from core.env
    load.py                   load_<task>_data() -> DataFrame from parquet in DATA_DIR
    modeling.py               param-name lookups, model-specific helpers
    export/fb_export.py       Firestore -> raw JSON -> parquet pipeline
    analyses/{behavior,comparison,modeling}/
  explore_exploit/            same shape as gain_loss
  questionnaires/             same shape, but load.py handles questionnaire CSVs (scores/subscales/items)
    export/fb_export_questionnaires.py
    analyses/
```

### Data flow

1. **Export from Firestore.** `*/export/fb_export*.py` scripts pull task or questionnaire documents from Firebase using `FB_CREDENTIALS_FILE` and write them to disk under `DATA_ROOT/<task>/`. Questionnaires are exported as long-format CSVs (`q_<questionnaire>_<spell>.csv`); behavioral tasks are exported as JSON and consolidated into `all_data.parquet`.
2. **Load.** `<task>.load.load_*_data(...)` reads the parquet/CSV and returns a tidy `pandas.DataFrame`. By default it routes through `comp_psych.core.selection.subselect_data` to drop dropped/practice trials and optionally filter by participant or session count.
3. **Analyze / model.** Analysis scripts under `analyses/` consume the loaded DataFrame. Stan model fits land in `DATA_ROOT/<task>/stan_fits/`; the Stan source lives under `<task>/analyses/modeling/stan_models/`. `core.modeling.compute_map_estimates` and `<task>/modeling.py` handle parameter post-processing.

### Inclusion / filtering

Behavioral and questionnaire export scripts both honor a per-cohort inclusion list at `DEMOGRAPHICS_DIR/inclusion_list_<groupId>.csv` (column `participant_id`). Pass `--all` to bypass it. Participants not in the file are silently skipped during export.

### Cross-task conventions worth knowing

- Session id is parsed out of `session_id` (e.g. `s3_groupA` → session = `3`, group = `A`). Most loaders add a numeric `session` column.
- `core.selection.subselect_data` is the single chokepoint for trial/subject filtering — extend it there rather than reimplementing per-task filters.
- Every task imports from `comp_psych.core.env` for paths/credentials. Don't hardcode absolute paths inside `<task>/config.py`; derive them from `DATA_ROOT` / `TASK_ROOT`.

## Running export scripts

Export scripts are CLI entry points run as modules from the repo root (so `comp_psych.*` imports resolve):

```bash
# Questionnaire export — all four questionnaires for the spell
python -m comp_psych.questionnaires.export.fb_export_questionnaires \
    --spell s1_groupA

# Single questionnaire
python -m comp_psych.questionnaires.export.fb_export_questionnaires \
    --spell s1_groupA --questionnaire dass21

# Bypass inclusion-list filtering
python -m comp_psych.questionnaires.export.fb_export_questionnaires \
    --spell s1_groupA --all

# One participant only
python -m comp_psych.questionnaires.export.fb_export_questionnaires \
    --spell s1_groupA --prolific-id <ID>
```

`--questionnaire` is optional; when omitted the script loops over all supported questionnaires (`dass21`, `ocir`, `spq`, `demography`). These are hard-coded both in `get_questionnaire_config` and in the default loop in `main()` — adding a new one means updating both places.

`--credentials` overrides `FB_CREDENTIALS_FILE`; otherwise the script falls back to `GOOGLE_APPLICATION_CREDENTIALS` then to auto-detected `*firebase*.json` in CWD.

The export scripts depend on `firebase-admin` and `google-cloud-firestore`. These are not declared in `pyproject.toml`, so install them explicitly (`pip install firebase-admin google-cloud-firestore`). Import-time errors from these libraries are silently swallowed by the `suppress_stderr()` context, so a missing dependency presents as the script exiting 1 with no output — install the packages before debugging further.

## Tests

There is no test suite in this repo.
