"""
created 25.12.8

gain-loss behavior performance analysis

@author: cgrossman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from comp_psych.gain_loss.load import load_gain_loss_data


def analyze_performance(subselect=None, plot_flag=True):
    # Load data subselect data, remove dropped and practice trials by default
    data = load_gain_loss_data(subselect=subselect, subselect_defaults=True)

    subjs = data['participant_id'].unique()
    num_subjs = len(subjs)
    num_sessions = data['session_id'].str[1].nunique()

    win_prob = np.full((num_subjs, num_sessions), np.nan)
    win_prob_gain = np.full((num_subjs, num_sessions), np.nan)
    win_prob_loss = np.full((num_subjs, num_sessions), np.nan)

    corr_prob = np.full((num_subjs, num_sessions), np.nan)
    corr_prob_gain = np.full((num_subjs, num_sessions), np.nan)
    corr_prob_loss = np.full((num_subjs, num_sessions), np.nan)

    for s_ind, subj in enumerate(subjs):
        subj_data = data[data['participant_id'] == subj]
        subj_data = subj_data[subj_data['practice'] == 0]

        for session in range(num_sessions):
            session_data = subj_data[subj_data['spellId'].str.contains(f's{session+1}_')]
            if not session_data.empty:
                win_prob[s_ind, session] = np.sum(session_data['is_win']) / len(session_data['is_win'])
                corr_prob[s_ind, session] = np.sum(session_data['is_correct']) / len(session_data['is_correct'])

                # split by gain/loss trials
                win_prob_gain[s_ind, session] = np.sum((session_data['is_win'] == 1) & (session_data['trialType'] == 'gain')) / np.sum(session_data['trialType'] == 'gain')
                win_prob_loss[s_ind, session] = np.sum((session_data['is_win'] == 1) & (session_data['trialType'] == 'loss')) / np.sum(session_data['trialType'] == 'loss')

                corr_prob_gain[s_ind, session] = np.sum((session_data['is_correct'] == 1) & (session_data['trialType'] == 'gain')) / np.sum(session_data['trialType'] == 'gain')
                corr_prob_loss[s_ind, session] = np.sum((session_data['is_correct'] == 1) & (session_data['trialType'] == 'loss')) / np.sum(session_data['trialType'] == 'loss')

    if plot_flag:
        plot_performance(win_prob, win_prob_gain, win_prob_loss, corr_prob, corr_prob_gain, corr_prob_loss, num_sessions)

    return data, pd.DataFrame({
        'subject_id': list(subjs),
        'win_prob': list(win_prob),
        'win_prob_gain': list(win_prob_gain),
        'win_prob_loss': list(win_prob_loss),
        'corr_prob': list(corr_prob),
        'corr_prob_gain': list(corr_prob_gain),
        'corr_prob_loss': list(corr_prob_loss)
    })


def plot_performance(win_prob, win_prob_gain, win_prob_loss, corr_prob, corr_prob_gain, corr_prob_loss, num_sessions):
    # Plot results
    sessions = np.arange(1, num_sessions + 1)
    plt.figure(figsize=(6, 8))
    plt.subplot(2, 1, 1)
    plt.errorbar(sessions, np.nanmean(win_prob, axis=0), yerr=np.nanstd(win_prob, axis=0), label='Win Probability', marker='o', color='blue')
    plt.errorbar(sessions, np.nanmean(corr_prob, axis=0), yerr=np.nanstd(corr_prob, axis=0), label='Correct Choice Probability', marker='o', color='green')
    plt.xticks(np.arange(1,num_sessions+1))
    plt.title('Both Block Types')
    plt.xlabel('Session')
    plt.ylabel('Probability')
    plt.xlim(0, num_sessions + 1)
    plt.ylim(0, 1)
    plt.legend(loc='lower left')
    plt.subplot(2, 1, 2)
    plt.errorbar(sessions, np.nanmean(win_prob_gain, axis=0), yerr=np.nanstd(win_prob_gain, axis=0), label='Win Probability Gain', marker='o', color=(0.1, 0.1, 0.5))
    plt.errorbar(sessions, np.nanmean(win_prob_loss, axis=0), yerr=np.nanstd(win_prob_loss, axis=0), label='Win Probability Loss', marker='o', color=(0.5, 0.5, 1))
    plt.errorbar(sessions, np.nanmean(corr_prob_gain, axis=0), yerr=np.nanstd(corr_prob_gain, axis=0), label='Correct Choice Probability Gain', marker='o', color=(0.1, 0.5, 0.1))
    plt.errorbar(sessions, np.nanmean(corr_prob_loss, axis=0), yerr=np.nanstd(corr_prob_loss, axis=0), label='Correct Choice Probability Loss', marker='o', color=(0.5, 1, 0.5)) 
    plt.xticks(np.arange(1,num_sessions+1))
    plt.title('Within Block Types')
    plt.xlabel('Session')
    plt.ylabel('Probability')
    plt.xlim(0, num_sessions + 1)
    plt.ylim(0, 1)
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.show(block=True)

    plt.figure(figsize=(12, 4))
    for s_ind in range(num_sessions):
        plt.subplot(1, num_sessions, s_ind + 1)
        plt.hist(win_prob[:, s_ind], bins=20, range=(0, 1), density=False, alpha=0.7, label='Overall', color='purple')
        plt.hist(win_prob_gain[:, s_ind], bins=20, range=(0, 1), alpha=0.7, label='Gain', color='blue')
        plt.hist(win_prob_loss[:, s_ind], bins=20, range=(0, 1),  alpha=0.7, label='Loss', color='red')
        plt.title(f'Session {s_ind + 1}')
        plt.xlabel('Win Probability')
        plt.ylabel('Subject Probability')
        if s_ind == 0:
            plt.legend()
    plt.show()



if __name__ == "__main__":
    subselect = {'num_sessions': 3}
    analyze_performance(subselect=subselect, plot_flag=True)