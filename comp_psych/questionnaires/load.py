"""
created 25.12.2

utilities for questionnaire data

@author: cgrossman
"""

import pandas as pd
import numpy as np
import os
from comp_psych.core.env import QUESTIONNAIRE_DIR
from comp_psych.core.selection import subselect_data

def load_scores(questionnaire, subselect=None):
    """Load per-participant total scores for a questionnaire, across all exported sessions.

    The "real" total score is computed as the catch-trial-corrected total
    (`catch_total - last_catch_value`); see the export pipeline's `total`
    and `catch` items.

    Parameters
    ----------
    questionnaire : str
        Questionnaire ID (e.g. 'dass21'); matched as a substring against
        exported CSV filenames in `QUESTIONNAIRE_DIR`.
    subselect : dict, optional
        Filter criteria passed to `comp_psych.core.selection.subselect_data`
        (with `defaults=False`, so no trial-level defaults are applied).

    Returns
    -------
    pandas.DataFrame
        One row per (participant, session), with `participant_id`, `score`,
        `session`, and `group` columns.

    Raises
    ------
    FileNotFoundError
        If no exported CSVs match `questionnaire` in `QUESTIONNAIRE_DIR`.
    """

    q_filenames = []
    for q in os.listdir(QUESTIONNAIRE_DIR):
        if questionnaire in q:
            q_filenames.append(q)

    if not q_filenames:
        raise FileNotFoundError(
            f"No exported CSVs matching questionnaire '{questionnaire}' found in {QUESTIONNAIRE_DIR}. "
            f"Run `python -m comp_psych.questionnaires.export.fb_export_questionnaires --questionnaire {questionnaire}` first."
        )

    qd = []
    for q in q_filenames:
        session_group = q.split('_')
        session = session_group[2]
        group = session_group[3].split('.')[0]

        q_data = pd.read_csv(os.path.join(QUESTIONNAIRE_DIR, q))

        # Find catch trial value and compute real score
        q_data['last_catch_value'] = (
            q_data['value']
            .where(q_data['item'] == 'catch')
            .ffill()
            .astype(float)
        )
        q_data['catch_total'] = (
            q_data['value']
            .where(q_data['item'] == 'total')
            .astype(float)
        )
        q_data['score'] = (
            q_data['catch_total'] - q_data['last_catch_value']
        ).where(q_data['item'] == 'total')

        # Extract total scores and prolific IDs only
        q_data = (
            q_data
            .loc[q_data['item'] == 'total',
                ['prolificId', 'score']]
            .rename(columns={'prolificId': 'participant_id'})
            .reset_index(drop=True)
        )

        q_data['session'] = session[1:]
        q_data['group'] = group[-1]

        qd.append(q_data)

    qd = pd.concat(qd, ignore_index=True)

    # Convert only numeric-compatible columns to int
    for col in qd.columns:
        tmp = pd.to_numeric(qd[col], errors="coerce")       # Try converting to numeric (floats allowed first)
        if not tmp.isna().any():                            # Only convert if all values could be coerced to numbers (no NaNs introduced)
            qd[col] = tmp.astype(int)                       # Convert to int

    if subselect is not None:
        qd = subselect_data(qd, subselect, defaults=False)

    return qd

def load_subscales(questionnaire, subselect=None):
    """Load per-participant subscale sums for a questionnaire, across all exported sessions.

    Parameters
    ----------
    questionnaire : str
        Questionnaire ID (e.g. 'dass21'); matched as a substring against
        exported CSV filenames in `QUESTIONNAIRE_DIR`.
    subselect : dict, optional
        Filter criteria passed to `comp_psych.core.selection.subselect_data`
        (with `defaults=False`, so no trial-level defaults are applied).

    Returns
    -------
    pandas.DataFrame
        One row per (participant, session), with `participant_id`, one
        column per subscale code (e.g. 'A', 'S', 'D' for dass21), `session`,
        and `group`.

    Raises
    ------
    FileNotFoundError
        If no exported CSVs match `questionnaire` in `QUESTIONNAIRE_DIR`.
    """

    q_filenames = []
    for q in os.listdir(QUESTIONNAIRE_DIR):
        if questionnaire in q:
            q_filenames.append(q)

    if not q_filenames:
        raise FileNotFoundError(
            f"No exported CSVs matching questionnaire '{questionnaire}' found in {QUESTIONNAIRE_DIR}. "
            f"Run `python -m comp_psych.questionnaires.export.fb_export_questionnaires --questionnaire {questionnaire}` first."
        )

    qd = []
    for q in q_filenames:
        session_group = q.split('_')
        session = session_group[2]
        group = session_group[3].split('.')[0]

        q_data = pd.read_csv(os.path.join(QUESTIONNAIRE_DIR, q))

        q_data = (
            q_data
            .loc[q_data['item'] == 'subscale_sum',
                ['prolificId', 'type', 'value']]
            .rename(columns={'prolificId': 'participant_id'})
            .pivot(index='participant_id', columns='type', values='value')
            .reset_index()
        )

        q_data['session'] = session[1:]
        q_data['group'] = group[-1]

        qd.append(q_data)

    qd = pd.concat(qd, ignore_index=True)

    # Convert only numeric-compatible columns to int
    for col in qd.columns:
        tmp = pd.to_numeric(qd[col], errors="coerce")       # Try converting to numeric (floats allowed first)
        if not tmp.isna().any():                            # Only convert if all values could be coerced to numbers (no NaNs introduced)
            qd[col] = tmp.astype(int)                       # Convert to int

    if subselect is not None:
        qd = subselect_data(qd, subselect, defaults=False)

    return qd


def load_questions(questionnaire, subselect=None):
    """Load per-participant item-level responses for a questionnaire, wide-format.

    Parameters
    ----------
    questionnaire : str
        Questionnaire ID (e.g. 'dass21'); matched as a substring against
        exported CSV filenames in `QUESTIONNAIRE_DIR`.
    subselect : dict, optional
        Filter criteria passed to `comp_psych.core.selection.subselect_data`
        (with `defaults=False`, so no trial-level defaults are applied).

    Returns
    -------
    pandas.DataFrame
        One row per (participant, session), with `participant_id`, one
        column per prompt (ordered by leading prompt number, demographic
        "Please select..." columns excluded), `session`, and `group`.

    Raises
    ------
    FileNotFoundError
        If no exported CSVs match `questionnaire` in `QUESTIONNAIRE_DIR`.
    """
    import re

    def extract_leading_number(col):
        # Extract prompt numbers for ordering 
        match = re.match(r"(\d+)", str(col))
        return int(match.group(1)) if match else float('inf')

    q_filenames = []
    for q in os.listdir(QUESTIONNAIRE_DIR):
        if questionnaire in q:
            q_filenames.append(q)

    if not q_filenames:
        raise FileNotFoundError(
            f"No exported CSVs matching questionnaire '{questionnaire}' found in {QUESTIONNAIRE_DIR}. "
            f"Run `python -m comp_psych.questionnaires.export.fb_export_questionnaires --questionnaire {questionnaire}` first."
        )

    qd = []
    for q in q_filenames:
        session_group = q.split('_')
        session = session_group[2]
        group = session_group[3].split('.')[0]

        q_data = pd.read_csv(os.path.join(QUESTIONNAIRE_DIR, q))

        q_data = (
            q_data
            .loc[q_data['prompt'].notna(),
                ['prolificId', 'value', 'prompt']]
            .rename(columns={'prolificId': 'participant_id'})
            .pivot(index='participant_id', columns='prompt', values='value')
            .reset_index()
        )

        # Reorder columns by prompt number
        question_cols = [c for c in q_data.columns if c != 'participant_id']
        sorted_cols = sorted(question_cols, key=extract_leading_number)
        q_data = q_data[['participant_id'] + sorted_cols]

        q_data['session'] = session[1:]
        q_data['group'] = group[-1]

        qd.append(q_data)

    qd = pd.concat(qd, ignore_index=True)
    qd = qd.loc[:, ~qd.columns.str.contains("Please select", case=False)]

    # Convert only numeric-compatible columns to int
    for col in qd.columns:
        tmp = pd.to_numeric(qd[col], errors="coerce")       # Try converting to numeric (floats allowed first)
        if not tmp.isna().any():                            # Only convert if all values could be coerced to numbers (no NaNs introduced)
            qd[col] = tmp.astype(int)                       # Convert to int

    if subselect is not None:
        qd = subselect_data(qd, subselect, defaults=False)

    return qd

def aggregate_sessions(qd):
    """Collapse a per-(participant, session) DataFrame to one row per participant.

    Every column except `participant_id` (the grouping key) and `group`
    (kept via 'first', assumed constant per participant) is aggregated into
    a list, in the original row order for that participant.

    Parameters
    ----------
    qd : pandas.DataFrame
        Long-format data with a `participant_id` column, as returned by
        `load_scores`, `load_subscales`, or `load_questions`.

    Returns
    -------
    pandas.DataFrame
        One row per participant; all non-`participant_id`/`group` columns
        hold per-session lists.
    """

    # Define dictionary for columns to be aggregated
    agg_dict = {}
    for col in qd.columns:
        if col in ['participant_id']:
            continue
        elif col == 'group':
            agg_dict[col] = 'first'
        else:
            agg_dict[col] = list

    # Aggregate data
    qd = (
        qd
        .groupby('participant_id', as_index=False)
        .agg(agg_dict)
    )

    return qd