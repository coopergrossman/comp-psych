"""
created 26.2.11

filtering utilities for all data

@author: cgrossman
"""

import pandas as pd
import numpy as np

def subselect_data(df, subselect=None, defaults=True):
    """Filter a trial-level DataFrame by trial- and subject-level criteria.

    Single chokepoint for trial/subject filtering shared across task domains;
    extend here rather than reimplementing per-task filters.

    Parameters
    ----------
    df : pandas.DataFrame
        Trial-level data. Must contain ``rt``, ``practice``, ``participant_id``,
        and ``session_id`` columns for the corresponding filters to apply, and
        ``session`` for the ``num_sessions`` filter.
    subselect : dict, optional
        Filter criteria. Recognized keys:

        - ``remove_dropped`` (bool): drop trials with a missing ``rt``.
        - ``remove_practice`` (bool): drop practice trials (kept rows have
          ``practice`` == 0; practice trials are coded non-zero).
        - ``num_sessions`` (int): keep only participants with exactly this
          many distinct completed sessions.
        - ``group_id`` (list of str): keep only rows whose group letter
          (last character of ``session_id``, e.g. 's1_groupA' -> 'A') is in
          this list.
        - ``participant_id`` (list of str): keep only these participants.
    defaults : bool, default True
        If True and not already set in `subselect`, default
        ``remove_dropped`` and ``remove_practice`` to True.

    Returns
    -------
    pandas.DataFrame
        The filtered data, with a reset integer index.
    """
    if subselect is None:
        subselect = {}
    if defaults:
        if 'remove_dropped' not in subselect:
            subselect['remove_dropped'] = True
        if 'remove_practice' not in subselect:
            subselect['remove_practice'] = True

    # Remove dropped trials
    if 'remove_dropped' in subselect and subselect['remove_dropped']:
        if df['rt'].isna().any():
            df = df.dropna(subset=['rt'])
    # Remove practice trials
    if 'remove_practice' in subselect and subselect['remove_practice']:
        df = df[df['practice'] == 0]

    # Filter subjects by number of sessions
    if 'num_sessions' in subselect:
        session_counts = df.groupby('participant_id')['session'].nunique()
        valid_subjs = session_counts[session_counts == subselect['num_sessions']].index
        df = df[df['participant_id'].isin(valid_subjs)]

    # Filter by group ID (last character of session_id, e.g. 's1_groupA' -> 'A')
    if 'group_id' in subselect:
        df = df[df['session_id'].str[-1].isin(subselect['group_id'])]

    # Filter by participant ID
    if 'participant_id' in subselect:
        df = df[df['participant_id'].isin(subselect['participant_id'])]

    # Reset index after filtering
    df = df.reset_index(drop=True)

    return df

