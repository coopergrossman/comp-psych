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
    """Compute per-parameter MAP (posterior mode) estimates from MCMC samples.

    The mode is estimated via a Gaussian KDE fit to the samples, rather than
    a raw histogram argmax, to reduce sensitivity to bin/sample noise.

    Parameters
    ----------
    samples : pandas.DataFrame or mapping
        Posterior draws, keyed by parameter name. For ``type='subject'``,
        must contain a ``mu_{param}`` column per name in `param_names`. For
        ``type='session'``, must contain a ``{param}[{session}]`` column
        (1-indexed) per name in `param_names` and session in
        ``range(num_sessions)``.
    param_names : list of str
        Parameter names to estimate.
    type : {'subject', 'session'}, default 'subject'
        Whether estimates are computed once per parameter ('subject') or
        once per parameter per session ('session').
    num_sessions : int, optional
        Number of sessions; required when `type` is 'session'.

    Returns
    -------
    numpy.ndarray
        1D array of shape ``(len(param_names),)`` for ``type='subject'``, or
        2D array of shape ``(num_sessions, len(param_names))`` for
        ``type='session'``.
    """

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
    """Load saved per-participant MAP parameter estimates for a fitted model.

    Reads ``param_estimates.npz`` (written by
    ``gain_loss.analyses.modeling.fit_stan_model``) from each participant's
    subdirectory under ``model_save_dir/model_name``.

    Parameters
    ----------
    model_name : str
        Name of the fitted model (subdirectory of `model_save_dir`).
    model_save_dir : str or pathlib.Path
        Directory containing per-model subdirectories of fit outputs.
    param_names : list of str
        Parameter names, in the same order as the columns of each
        participant's saved ``param_estimates`` array.

    Returns
    -------
    parameters : pandas.DataFrame
        One row per participant, with a ``participant_id`` column and one
        column per name in `param_names`; each parameter cell holds that
        participant's estimate array (e.g. one value per session).
    param_names : list of str
        The `param_names` passed in, returned unchanged for convenience.
    """
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