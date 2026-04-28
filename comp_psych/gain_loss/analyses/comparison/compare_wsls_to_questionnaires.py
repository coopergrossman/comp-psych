"""
created 25.12.8 

compare gain_loss win-stay lose-shift behavior to questionnaire wsls

@author: cgrossman
"""

import pandas as pd
import numpy as np
from comp_psych.gain_loss.analyses.behavior.analyze_wsls import analyze_wsls
from comp_psych.questionnaires.load import load_subscales, aggregate_sessions, load_questions
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


def compare_wsls_to_questionnaires(subselect=None, questionnaire='dass21', plot_flag=True):

    # Load wsls and questionnaire wsls
    _, wsls = analyze_wsls(subselect=subselect, plot_flag=False)
    qd = load_subscales(questionnaire)
    qd = aggregate_sessions(qd)

    wsls = pd.merge(wsls, qd, on='participant_id')

    # Find and remove subjects with only 2 completed questionnaires
    wsls['num_questionnaires'] = wsls['A'].apply(len)
    wsls = wsls[wsls['num_questionnaires'] >= subselect['num_sessions']].reset_index(drop=True)

    array_cols = ['A', 'S', 'D', 'win_stay', 'lose_shift', 'win_stay_gain', 'lose_shift_gain', 'win_stay_loss', 'lose_shift_loss']  # columns that contain arrays
    wsls = (
        wsls
        .assign(session=lambda x: x[array_cols[0]].apply(lambda v: range(1, len(v) + 1)))
        .explode(array_cols + ['session'], ignore_index=True)
    )
    wsls[array_cols] = wsls[array_cols].apply(pd.to_numeric, errors='raise')

    # Run regression analyses
    mdl_win_stay = smf.ols('win_stay ~ A + S + D', data=wsls).fit()
    print(mdl_win_stay.summary())
    mdl_lose_shift = smf.ols('lose_shift ~ A + S + D', data=wsls).fit()
    print(mdl_lose_shift.summary())
    
    mdl_win_stay_gain = smf.ols('win_stay_gain ~ A + S + D', data=wsls).fit()
    print(mdl_win_stay_gain.summary())
    mdl_lose_shift_gain = smf.ols('lose_shift_gain ~ A + S + D', data=wsls).fit()
    print(mdl_lose_shift_gain.summary())
    mdl_win_stay_loss = smf.ols('win_stay_loss ~ A + S + D', data=wsls).fit()
    print(mdl_win_stay_loss.summary())
    mdl_lose_shift_loss = smf.ols('lose_shift_loss ~ A + S + D', data=wsls).fit()
    print(mdl_lose_shift_loss.summary())


    if plot_flag:
        plot_compare_wsls_to_questionnaires(wsls, array_cols, mdl_win_stay, mdl_lose_shift, mdl_win_stay_gain, mdl_lose_shift_gain, mdl_win_stay_loss, mdl_lose_shift_loss)



def plot_compare_wsls_to_questionnaires(wsls, array_cols, mdl_win_stay, mdl_lose_shift, mdl_win_stay_gain, mdl_lose_shift_gain, mdl_win_stay_loss, mdl_lose_shift_loss):

    behavior_vars = array_cols[3:]
    questionnaire_vars = array_cols[:3]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=6,
        figsize=(18, 9),
        sharex='col',
        sharey=False
    )

    for row, q_var in enumerate(questionnaire_vars):
        for col, behav in enumerate(behavior_vars):
            ax = axes[row, col]

            # scatter
            ax.scatter(
                wsls[q_var],
                wsls[behav],
                alpha=0.5
            )

            # fit simple line for plotting (same as partial effect here)
            x = wsls[q_var]
            y = wsls[behav]
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
        'WSLS Measures vs Questionnaire Subscales\nScatter + Linear Fit',
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



    models = [
        (mdl_win_stay, 'Win-Stay'),
        (mdl_lose_shift, 'Lose-Shift'),
        (mdl_win_stay_gain, 'Win-Stay Gain'),
        (mdl_lose_shift_gain, 'Lose-Shift Gain'),
        (mdl_win_stay_loss, 'Win-Stay Loss'),
        (mdl_lose_shift_loss, 'Lose-Shift Loss')
    ]

    # Compute global x-axis limits (beta ± SE across all models)
    all_betas = []
    all_ses = []

    for mdl, _ in models:
        all_betas.append(mdl.params[1:])  # skip intercept
        all_ses.append(mdl.bse[1:])

    all_betas = pd.concat(all_betas)
    all_ses = pd.concat(all_ses)

    x_min = min(all_betas - all_ses)
    x_max = max(all_betas + all_ses)
    x_limit = max(abs(x_min), abs(x_max))  # symmetric around zero

    fig, axes = plt.subplots(3, 2, figsize=(5, 8))
    axes = axes.flatten()

    for ax, (mdl, label) in zip(axes, models):
        coefs = mdl.params[1:]
        ses = mdl.bse[1:]
        pvals = mdl.pvalues[1:]
        y_pos = np.arange(len(coefs))

        # Gray points for all
        ax.errorbar(
            coefs, y_pos, xerr=ses,
            fmt='o', color='gray', ecolor='gray',
            capsize=0, linewidth = 2  # removes vertical caps
        )

        # Overlay significant points in blue
        for i, (coef, se, p) in enumerate(zip(coefs, ses, pvals)):
            if p < 0.05:
                ax.errorbar(
                    coef, i, xerr=se,
                    fmt='o', color='blue', ecolor='blue',
                    capsize=0, linewidth = 2
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(coefs.index)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Beta coefficient')
        ax.set_title(label)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlim(-x_limit, x_limit)  # same limits for all subplots

    plt.suptitle('Regression Betas ± SE (Significant in Blue)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    subselect = {'num_sessions': 3}
    compare_wsls_to_questionnaires(subselect=subselect, questionnaire='dass21', plot_flag=True)