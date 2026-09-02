"""
created 25.12.8 

compare gain_loss win-stay lose-shift behavior to questionnaire parameters

@author: cgrossman
"""

import pandas as pd
import numpy as np
from comp_psych.questionnaires.load import load_subscales, aggregate_sessions
from comp_psych.core.modeling import load_model_parameters
from comp_psych.gain_loss.config import MODEL_SAVE_DIR
from comp_psych.gain_loss.modeling import get_param_names
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

def compare_parameters_to_questionnaires(
        subselect=None, model_name = 'q',
        questionnaire='dass21',
        plot_flag=True
):
    """Regress fitted Stan model parameters against questionnaire subscales.

    Loads per-session MAP parameter estimates for `model_name` and
    per-session subscale scores for `questionnaire`, aligns them by
    (participant, session), and fits one OLS model per parameter
    (`param ~ A * S * D`).

    Parameters
    ----------
    subselect : dict
        Must include `num_sessions`, the minimum number of completed
        questionnaires required to keep a participant.
    model_name : str, default 'q'
        Fitted Stan model to load (see `comp_psych.gain_loss.modeling.get_param_names`).
    questionnaire : str, default 'dass21'
        Questionnaire to load subscales for.
    plot_flag : bool, default True
        If True, show scatter/regression-coefficient plots via
        `plot_compare_parameters_to_questionnaires`.
    """

    # Load parameters and questionnaire data
    param_names = get_param_names(model_name)
    parameters, param_names = load_model_parameters(
        model_name=model_name, 
        model_save_dir=MODEL_SAVE_DIR, 
        param_names=param_names
    )
    qd = load_subscales(questionnaire)
    qd = aggregate_sessions(qd)

    # Find and remove subjects with only 2 completed questionnaires
    qd['num_questionnaires'] = qd['A'].apply(len)
    qd = qd[qd['num_questionnaires'] >= subselect['num_sessions']].reset_index(drop=True)

    questionnaire_cols = ['A', 'S', 'D']

    # Explode parameters and questionnaire scores separately, each against its
    # own per-row session index, then join on (participant_id, session). A
    # participant's number of completed questionnaires can differ from the
    # number of sessions the model was fit on, so a single positional explode
    # across both arrays breaks whenever those lengths don't match.
    parameters = (
        parameters
        .assign(session=lambda x: x[param_names[0]].apply(lambda v: range(1, len(v) + 1)))
        .explode(param_names + ['session'], ignore_index=True)
    )
    parameters['session'] = parameters['session'].astype(int)
    parameters[param_names] = parameters[param_names].apply(pd.to_numeric, errors='raise')

    qd = (
        qd
        .assign(session=lambda x: x[questionnaire_cols[0]].apply(lambda v: range(1, len(v) + 1)))
        .explode(questionnaire_cols + ['session'], ignore_index=True)
    )
    qd['session'] = qd['session'].astype(int)
    qd[questionnaire_cols] = qd[questionnaire_cols].apply(pd.to_numeric, errors='raise')

    parameters = pd.merge(
        parameters,
        qd[['participant_id', 'session'] + questionnaire_cols],
        on=['participant_id', 'session']
    )

    array_cols = questionnaire_cols + param_names  # columns that contain arrays

    # Run regression analyses
    mdls = {}
    for p_ind, param in enumerate(param_names):
        mdls[p_ind] = smf.ols(f'{param} ~ A * S * D', data=parameters).fit()

    if plot_flag:
        plot_compare_parameters_to_questionnaires(parameters, array_cols, mdls, param_names)



def plot_compare_parameters_to_questionnaires(parameters, array_cols, mdls, param_names):
    """Plot parameter-vs-subscale scatter fits and regression coefficients.

    Parameters
    ----------
    parameters : pandas.DataFrame
        Merged per-(participant, session) parameters and subscale scores, as
        built in `compare_parameters_to_questionnaires`.
    array_cols : list of str
        Questionnaire subscale columns (first 3) followed by model
        parameter columns.
    mdls : dict of int -> statsmodels regression result
        One fitted OLS model per parameter, keyed by index into `param_names`.
    param_names : list of str
        Model parameter names, in the same order as `mdls`' keys.
    """
    questionnaire_vars = array_cols[:3]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=len(param_names),
        figsize=(18, 9),
        sharex='col',
        sharey=False
    )

    for row, q_var in enumerate(questionnaire_vars):
        for col, behav in enumerate(param_names):
            ax = axes[row, col]

            # scatter
            ax.scatter(
                parameters[q_var],
                parameters[behav],
                alpha=0.5
            )

            # fit simple line for plotting (same as partial effect here)
            x = parameters[q_var]
            y = parameters[behav]
            slope, intercept = np.polyfit(x, y, 1)

            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = intercept + slope * x_line

            ax.plot(x_line, y_line)

            if row == 0:
                ax.set_title(behav.replace('_', ' ').title())
            if col == 0:
                ax.set_ylabel(q_var)

            ax.spines[['right', 'top']].set_visible(False)

    plt.suptitle(
        'Parameters vs Questionnaire Subscales\nScatter + Linear Fit',
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



    # Compute global x-axis limits (beta ± SE across all models)
    all_betas = []
    all_ses = []

    for mdl in mdls.values():  # iterate over actual regression results
        all_betas.append(mdl.params[1:])  # skip intercept
        all_ses.append(mdl.bse[1:])

    all_betas = pd.concat(all_betas)
    all_ses = pd.concat(all_ses)

    x_min = min(all_betas - all_ses)
    x_max = max(all_betas + all_ses)
    x_limit = max(abs(x_min), abs(x_max))  # symmetric around zero

    # Create 3x2 figure for six models
    fig, axes = plt.subplots(3, 2, figsize=(5, 8))
    axes = axes.flatten()

    for ax, (p_ind, mdl) in zip(axes, mdls.items()):
        coefs = mdl.params[1:]
        ses = mdl.bse[1:]
        pvals = mdl.pvalues[1:]
        y_pos = np.arange(len(coefs))

        # Gray points
        ax.errorbar(
            coefs, y_pos, xerr=ses,
            fmt='o', color='gray', ecolor='gray',
            capsize=0, linewidth=2
        )

        # Overlay significant points in blue
        for i, (coef, se, p) in enumerate(zip(coefs, ses, pvals)):
            if p < 0.05:
                ax.errorbar(
                    coef, i, xerr=se,
                    fmt='o', color='blue', ecolor='blue',
                    capsize=0, linewidth=2
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(coefs.index)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Beta coefficient')
        ax.set_title(param_names[p_ind].replace('_', ' ').title())
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlim(-x_limit, x_limit)

    plt.suptitle('Regression Betas ± SE (Significant in Blue)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    subselect = {'num_sessions': 3}
    compare_parameters_to_questionnaires(subselect=subselect, model_name = 'q_a_win_lose_loss_gain_forget', questionnaire='dass21', plot_flag=True)