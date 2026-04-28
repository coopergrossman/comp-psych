"""
created 26.2.11

filtering utilities for all data

@author: cgrossman
"""

import pandas as pd
import numpy as np

def subselect_data(df, subselect=None, defaults=True):

    if subselect is None:
        subselect = {}
    if defaults:
        if 'remove_dropped' not in subselect:
            subselect['remove_dropped'] = True
        if 'remove_practice' not in subselect:
            subselect['remove_practice'] = True

    # Remove dropped trials
    if 'remove_dropped' in subselect and subselect['remove_dropped']:
        if df['rt'].isna().any:
            df = df.dropna(subset=['rt'])
    # Remove practice trials
    if 'remove_practice' in subselect and subselect['remove_practice']:
        df = df[df['practice'] == 0]

    # Filter subjects by number of sessions
    if 'num_sessions' in subselect:
        session_counts = df.groupby('participant_id')['session'].nunique()
        valid_subjs = session_counts[session_counts == subselect['num_sessions']].index
        df = df[df['participant_id'].isin(valid_subjs)]

    # Filter by participant ID
    if 'participant_id' in subselect:
        df = df[df['participant_id'].isin(subselect['participant_id'])]

    # Reset index after filtering
    df = df.reset_index(drop=True)

    return df

