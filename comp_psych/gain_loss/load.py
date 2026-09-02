"""
created 25.12.2

utilities for gain_loss data

@author: cgrossman
"""

import pandas as pd
import numpy as np
from comp_psych.gain_loss.config import DATA_DIR
from comp_psych.core.selection import subselect_data

def load_gain_loss_data(subselect=None, subselect_defaults=True):
    """Load exported gain_loss trial data from parquet as a tidy DataFrame.

    Parameters
    ----------
    subselect : dict, optional
        Filter criteria passed to `comp_psych.core.selection.subselect_data`.
    subselect_defaults : bool, default True
        If True, apply `subselect_data`'s default filters (drop dropped and
        practice trials) even when `subselect` is None.

    Returns
    -------
    pandas.DataFrame
        Trial-level data with a numeric `session` column parsed from
        `session_id` (e.g. 's3_groupA' -> 3).

    Raises
    ------
    FileNotFoundError
        If `all_data.parquet` hasn't been exported yet for this task.
    """
    parquet_path = DATA_DIR / 'all_data.parquet'
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No exported data found at {parquet_path}. "
            f"Run `python -m comp_psych.gain_loss.export.fb_export` first."
        )
    df = pd.read_parquet(parquet_path)
    df['session'] = df['session_id'].str[1].astype(int)

    if subselect is not None or subselect_defaults:
        df = subselect_data(df, subselect, defaults=subselect_defaults)

    return df