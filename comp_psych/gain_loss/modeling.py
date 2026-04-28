"""
created 26.1.9

utilities for modeling gain/loss data

@author: cgrossman
"""

import numpy as np
import pandas as pd
import os
from scipy.stats import gaussian_kde
from comp_psych.gain_loss.config import MODEL_SAVE_DIR

def get_param_names(model_name):
    if model_name == 'q':
        return ['a', 'beta']
    elif model_name == 'q_a_win_lose':
        return ['a_win', 'a_lose', 'beta']
    elif model_name == 'q_a_win_lose_loss_gain':
        return ['a_win_gain', 'a_lose_gain', 'a_win_loss', 'a_lose_loss', 'beta']
    elif model_name == 'q_a_win_lose_loss_gain_forget':
        return ['a_win_gain', 'a_lose_gain', 'a_win_loss', 'a_lose_loss', 'forget', 'beta']
    