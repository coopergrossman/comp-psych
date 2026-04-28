"""
created 25.12.2

gain-loss win-stay behavior analysis

@author: cgrossman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from compPsych.gain_loss.load import load_gain_loss_data


def analyze_wsls(subselect=None, plot_flag=True):    
    # Load and subselect data, remove dropped and practice trials by default
    data = load_gain_loss_data(subselect=subselect, subselect_defaults=True)

    # Find subjects
    subjs = data['participant_id'].unique()
    num_subjs = len(subjs)
    num_sessions = data['session_id'].str[1].nunique()

    win_stay = np.full((num_subjs, num_sessions), np.nan)
    lose_shift = np.full((num_subjs, num_sessions), np.nan)
    win_stay_gain = np.full((num_subjs, num_sessions), np.nan)
    lose_shift_gain = np.full((num_subjs, num_sessions), np.nan)
    win_stay_loss = np.full((num_subjs, num_sessions), np.nan)
    lose_shift_loss = np.full((num_subjs, num_sessions), np.nan)

    for s_ind, subj in enumerate(subjs):
        subj_data = data[data['participant_id'] == subj].copy()

        subj_data['is_stay'] = subj_data['chosenStimID'] == subj_data['chosenStimID'].shift(1)
        subj_data['prev_win'] = subj_data['is_win'].shift(1)

        # Do not count trials after stimuli change
        subj_data.loc[subj_data['block_change'] == True, 'is_stay'] = np.nan
        subj_data.loc[subj_data['block_change'] == True, 'prev_win'] = np.nan

        for session in range(num_sessions):
            session_data = subj_data[subj_data['spellId'].str.contains(f's{session+1}_')].copy()
            if not session_data.empty:
                win_stay[s_ind, session] = np.sum((session_data['prev_win'] == 1) & (session_data['is_stay'] == 1)) / np.sum(session_data['prev_win'] == 1)
                lose_shift[s_ind, session] = np.sum((session_data['prev_win'] == 0) & (session_data['is_stay'] == 0)) / np.sum(session_data['prev_win'] == 0)

                # split by gain/loss trials
                win_stay_gain[s_ind, session] = np.sum((session_data['prev_win'] == 1) & (session_data['is_stay'] == 1) & (session_data['trialType'] == 'gain')) / np.sum((session_data['prev_win'] == 1) & (session_data['trialType'] == 'gain'))
                lose_shift_gain[s_ind, session] = np.sum((session_data['prev_win'] == 0) & (session_data['is_stay'] == 0) & (session_data['trialType'] == 'gain')) / np.sum((session_data['prev_win'] == 0) & (session_data['trialType'] == 'gain'))
                win_stay_loss[s_ind, session] = np.sum((session_data['prev_win'] == 1) & (session_data['is_stay'] == 1) & (session_data['trialType'] == 'loss')) / np.sum((session_data['prev_win'] == 1) & (session_data['trialType'] == 'loss'))
                lose_shift_loss[s_ind, session] = np.sum((session_data['prev_win'] == 0) & (session_data['is_stay'] == 0) & (session_data['trialType'] == 'loss')) / np.sum((session_data['prev_win'] == 0) & (session_data['trialType'] == 'loss'))

    if plot_flag:
        plot_wsls(win_stay, lose_shift, win_stay_gain, lose_shift_gain, win_stay_loss, lose_shift_loss, num_sessions)

    return data, pd.DataFrame({
        'participant_id': list(subjs),
        'win_stay': list(win_stay),
        'lose_shift': list(lose_shift),
        'win_stay_gain': list(win_stay_gain),
        'lose_shift_gain': list(lose_shift_gain),
        'win_stay_loss': list(win_stay_loss),
        'lose_shift_loss': list(lose_shift_loss)
    })


def plot_wsls(win_stay, lose_shift, win_stay_gain, lose_shift_gain, win_stay_loss, lose_shift_loss, num_sessions):
    # Plot results
    sessions = np.arange(1, num_sessions + 1)
    plt.figure(figsize=(6, 8))
    plt.subplot(2, 1, 1)
    plt.errorbar(sessions, np.nanmean(win_stay, axis=0), yerr=np.nanstd(win_stay, axis=0), label='Win-Stay', marker='o', color='blue')
    plt.errorbar(sessions, np.nanmean(lose_shift, axis=0), yerr=np.nanstd(lose_shift, axis=0), label='Lose-Shift', marker='o', color='red')
    plt.xticks(sessions)
    plt.title('Win-Stay and Lose-Shift Behavior Across Sessions')
    plt.xlabel('Session')
    plt.ylabel('Probability')
    plt.xlim(0, num_sessions + 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.errorbar(sessions, np.nanmean(win_stay_gain, axis=0), yerr=np.nanstd(win_stay_gain, axis=0), label='Win-Stay Gain', marker='o', color=(0.1, 0.1, 0.5))
    plt.errorbar(sessions, np.nanmean(lose_shift_gain, axis=0), yerr=np.nanstd(lose_shift_gain, axis=0), label='Lose-Shift Gain', marker='o', color=(0.5, 0.1, 0.1))
    plt.errorbar(sessions, np.nanmean(win_stay_loss, axis=0), yerr=np.nanstd(win_stay_loss, axis=0), label='Win-Stay Loss', marker='o', color=(0.5, 0.5, 1))
    plt.errorbar(sessions, np.nanmean(lose_shift_loss, axis=0), yerr=np.nanstd(lose_shift_loss, axis=0), label='Lose-Shift Loss', marker='o', color=(1, 0.5, 0.5))
    plt.xticks(sessions)
    plt.title('Win-Stay and Lose-Shift Behavior by Block Type Across Sessions')
    plt.xlabel('Session')
    plt.ylabel('Probability')
    plt.xlim(0, num_sessions + 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    subselect = {'num_sessions': 3}
    analyze_wsls(subselect=subselect, plot_flag=True)