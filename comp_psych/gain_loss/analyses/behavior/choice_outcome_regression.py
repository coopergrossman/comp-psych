"""
created 25.12.10

choice outcome regression analysis for gain_loss task

@author: cgrossman
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from comp_psych.gain_loss.load import load_gain_loss_data


def choice_outcome_regression(subselect=None, plot_flag=True, num_trials=5):
    # Load and subselect data, remove dropped and practice trials by default
    data = load_gain_loss_data(subselect=subselect, subselect_defaults=True)

    subjs = data['participant_id'].unique()
    num_subjs = len(subjs)

    win_coefs = np.full((num_subjs, num_trials), np.nan)
    lose_coefs = np.full((num_subjs, num_trials), np.nan)
    win_coefs_gain = np.full((num_subjs, num_trials), np.nan)
    lose_coefs_gain = np.full((num_subjs, num_trials), np.nan)
    win_coefs_loss = np.full((num_subjs, num_trials), np.nan)
    lose_coefs_loss = np.full((num_subjs, num_trials), np.nan)

    for s_ind, subj in enumerate(subjs):
        subj_data = data[data['participant_id'] == subj].copy()
        subj_data['choice_val'] = subj_data['choice'].replace(0,-1)
        subj_data['win_choice'] = pd.to_numeric(subj_data['is_win'] * subj_data['choice_val'])
        subj_data['lose_choice'] = pd.to_numeric((1 - subj_data['is_win']) * subj_data['choice_val'])

        for t_ind in range(num_trials):
            subj_data[f'win_{t_ind+1}'] = subj_data['win_choice'].shift(t_ind + 1)
            subj_data[f'lose_{t_ind+1}'] = subj_data['lose_choice'].shift(t_ind + 1)

        for b_ind in np.where(subj_data['block_change'] == True)[0]:
            for t_ind in range(num_trials):
                subj_data.loc[b_ind:b_ind+t_ind, f'win_{t_ind+1}'] = np.nan
                subj_data.loc[b_ind:b_ind+t_ind, f'lose_{t_ind+1}'] = np.nan

        # Regression on all data
        formula = 'choice ~ ' + ' + '.join(
            [f'win_{i+1}' for i in range(num_trials)]
            + [f'lose_{i+1}' for i in range(num_trials)]
        )
        mdl_choice_outcome = smf.logit(formula, data=subj_data).fit()
        win_coefs[s_ind, :] = mdl_choice_outcome.params[[f'win_{i+1}' for i in range(num_trials)]].values
        lose_coefs[s_ind, :] = mdl_choice_outcome.params[[f'lose_{i+1}' for i in range(num_trials)]].values

        # Regression on gain block trials
        subj_data_gain = subj_data[subj_data['trialType'] == 'gain']
        mdl_choice_outcome_gain = smf.logit(formula, data=subj_data_gain).fit()
        win_coefs_gain[s_ind, :] = mdl_choice_outcome_gain.params[[f'win_{i+1}' for i in range(num_trials)]].values
        lose_coefs_gain[s_ind, :] = mdl_choice_outcome_gain.params[[f'lose_{i+1}' for i in range(num_trials)]].values

        # Regression on loss block trials
        subj_data_loss = subj_data[subj_data['trialType'] == 'loss']
        mdl_choice_outcome_loss = smf.logit(formula, data=subj_data_loss).fit()
        win_coefs_loss[s_ind, :] = mdl_choice_outcome_loss.params[[f'win_{i+1}' for i in range(num_trials)]].values
        lose_coefs_loss[s_ind, :] = mdl_choice_outcome_loss.params[[f'lose_{i+1}' for i in range(num_trials)]].values

    if plot_flag:
        plot_choice_outcome_regression(win_coefs, lose_coefs, win_coefs_gain, lose_coefs_gain, win_coefs_loss, lose_coefs_loss, num_trials)

    return pd.DataFrame({
        'subject_id': list(subjs),
        'win_coefs': list(win_coefs),
        'lose_coefs': list(lose_coefs),
        'win_coefs_gain': list(win_coefs_gain),
        'lose_coefs_gain': list(lose_coefs_gain),
        'win_coefs_loss': list(win_coefs_loss),
        'lose_coefs_loss': list(lose_coefs_loss)
    })


def plot_choice_outcome_regression(win_coefs, lose_coefs, win_coefs_gain, lose_coefs_gain, win_coefs_loss, lose_coefs_loss, num_trials):
    # Plot results
    trials = np.arange(1, num_trials + 1)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 3, 1)
    # Compute SEM of coefficients
    sem = np.nanstd(win_coefs, axis=0) / np.sqrt(win_coefs.shape[0])
    plt.errorbar(trials, np.nanmean(win_coefs, axis=0), yerr=sem, label='Wins', marker='o', color='blue')
    sem = np.nanstd(lose_coefs, axis=0) / np.sqrt(lose_coefs.shape[0])
    plt.errorbar(trials, np.nanmean(lose_coefs, axis=0), yerr=sem, label='Loses', marker='o', color='red')
    plt.plot([0] + list(trials) + [trials[-1]+1], np.zeros(num_trials+2), 'k--')
    plt.title('Both Block Types')
    plt.xlabel('Outcome in the past (trial lag)')
    plt.ylabel('Mean Coefficient')
    plt.xticks(trials)
    plt.xlim(0.5, num_trials+0.5)
    plt.ylim(-0.4, 1.4)
    plt.legend()
    plt.subplot(1, 3, 2)
    sem = np.nanstd(win_coefs_gain, axis=0) / np.sqrt(win_coefs_gain.shape[0])
    plt.errorbar(trials, np.nanmean(win_coefs_gain, axis=0), yerr=sem, label='Reward', marker='o', color='blue')
    sem = np.nanstd(lose_coefs_gain, axis=0) / np.sqrt(lose_coefs_gain.shape[0])
    plt.errorbar(trials, np.nanmean(lose_coefs_gain, axis=0), yerr=sem, label='No Reward', marker='o', color='red')
    plt.plot([0] + list(trials) + [trials[-1]+1], np.zeros(num_trials+2), 'k--')
    plt.title('Gain Block Trials')
    plt.xlabel('Outcome in the past (trial lag)')
    plt.ylabel('Mean Coefficient')
    plt.xticks(trials)
    plt.xlim(0.5, num_trials+0.5)
    plt.ylim(-0.4, 1.4)
    plt.legend()
    plt.subplot(1, 3, 3)
    sem = np.nanstd(win_coefs_loss, axis=0) / np.sqrt(win_coefs_loss.shape[0])
    plt.errorbar(trials, np.nanmean(win_coefs_loss, axis=0), yerr=sem, label='No Loss', marker='o', color='blue')
    sem = np.nanstd(lose_coefs_loss, axis=0) / np.sqrt(lose_coefs_loss.shape[0])
    plt.errorbar(trials, np.nanmean(lose_coefs_loss, axis=0), yerr=sem, label='Loss', marker='o', color='red')
    plt.plot([0] + list(trials) + [trials[-1]+1], np.zeros(num_trials+2), 'k--')
    plt.title('Loss Block Trials')
    plt.xlabel('Outcome in the past (trial lag)')
    plt.ylabel('Mean Coefficient')
    plt.xticks(trials)
    plt.xlim(0.5, num_trials+0.5)
    plt.ylim(-0.4, 1.4)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    subselect = {'num_sessions': 3}
    choice_outcome_regression(subselect=subselect, plot_flag=True, num_trials=5)