"""
created 25.12.2

utilities for gain_loss data

@author: cgrossman
"""

import pandas as pd
import numpy as np
import os
from compPsych.core.env import QUESTIONNAIRE_DIR
from compPsych.core.selection import subselect_data

def load_scores(questionnaire, subselect=None):

    q_filenames = []
    for q in os.listdir(QUESTIONNAIRE_DIR):
        if questionnaire in q:
            q_filenames.append(q)

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

    q_filenames = []
    for q in os.listdir(QUESTIONNAIRE_DIR):
        if questionnaire in q:
            q_filenames.append(q)

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
    import re

    def extract_leading_number(col):
        # Extract prompt numbers for ordering 
        match = re.match(r"(\d+)", str(col))
        return int(match.group(1)) if match else float('inf')

    q_filenames = []
    for q in os.listdir(QUESTIONNAIRE_DIR):
        if questionnaire in q:
            q_filenames.append(q)

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