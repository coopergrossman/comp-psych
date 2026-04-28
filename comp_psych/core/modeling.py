"""
created 26.1.9

utilities for modeling gain/loss data

@author: cgrossman
"""

import numpy as np
import pandas as pd
import os
from scipy.stats import gaussian_kde

def compute_map_estimates(samples, param_names, type='subject', num_sessions=None):

    def kde_mode(samples_param):
        # KDE-based mode
        kde = gaussian_kde(samples_param)
        x = np.linspace(samples_param.min(), samples_param.max(), 1000)
        return x[np.argmax(kde(x))]

    if type == 'subject':
        map_estimates = np.zeros(len(param_names))
        for p_ind, param in enumerate(param_names):
             samples_param = samples[f'mu_{param}'].to_numpy()
             map_estimates[p_ind] = kde_mode(samples_param)

    elif type == 'session':
        map_estimates = np.zeros((num_sessions, len(param_names)))
        for p_ind, param in enumerate(param_names):
            for s in range(num_sessions):
                samples_param = samples[f'{param}[{s+1}]'].to_numpy()
                map_estimates[s, p_ind] = kde_mode(samples_param)

    return map_estimates    

def load_model_parameters(model_name=None, model_save_dir=None, param_names=None):
    model_dir = os.path.join(model_save_dir, model_name)
    participants = [
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
    ]

    parameters = pd.DataFrame(columns=['participant_id'] + param_names)

    for participant in participants:
        model_data = np.load(os.path.join(model_dir, participant, 'param_estimates.npz'), allow_pickle=True)
        param_estimates = model_data['param_estimates']

        # create new row index
        row_idx = len(parameters)
        parameters.loc[row_idx, 'participant_id'] = participant

        for p_ind, parameter in enumerate(param_names):
            parameters.loc[row_idx, parameter] = param_estimates[:, p_ind]

    return parameters, param_names