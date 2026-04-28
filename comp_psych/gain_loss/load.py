"""
created 25.12.2

utilities for gain_loss data

@author: cgrossman
"""

import pandas as pd
import numpy as np
from compPsych.gain_loss.config import DATA_DIR
from compPsych.core.selection import subselect_data

def load_gain_loss_data(subselect=None, subselect_defaults=True):
    df = pd.read_parquet(f'{DATA_DIR}\\all_data.parquet')
    df['session'] = df['session_id'].str[1].astype(int)

    if subselect is not None or subselect_defaults:
        df = subselect_data(df, subselect, defaults=subselect_defaults)

    return df