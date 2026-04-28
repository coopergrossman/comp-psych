"""
created 26.1.12

gain-loss transition behavior analysis

@author: cgrossman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from comp_psych.gain_loss.load import load_gain_loss_data


def format_data(data):
    data.loc[
        (data['prob_reversal'] == 1) & (data['block_change'] == 1),
        'prob_reversal'
    ] = 0
    data['is_correct_bin'] = data['is_correct'].astype(float)
    data['block_loss'] = (data['trialType'] == 'loss').astype(int)

    return data


def analyze_transition_behavior(subselect=None, plot_flag=True):
    # Load data and subselect data, remove dropped and practice trials by default
    data = load_gain_loss_data(subselect=subselect, subselect_defaults=True)
    num_sessions = data['session_id'].str[1].nunique()

    # Format data
    data = format_data(data)

    # Extract performance around probability reversals
    trials_before = 5
    trials_after = 10
    reversal_inds = data.index[data['prob_reversal'] == 1]
    transition = []
    block_loss = []
    session_num = []

    for i in reversal_inds:
        window = data.loc[
            i - trials_before : i + trials_after,
            'is_correct_bin'
        ].to_numpy()
        
        transition.append(window)
        block_loss.append(data.loc[i, 'block_loss'])
        session_num.append(data.loc[i, 'session'])


    transition = np.array(transition)
    block_loss = np.array(block_loss)
    session_num = np.array(session_num)

    # Separate by gain/loss blocks, session, or both
    transition_gain = transition[block_loss == 0]
    transition_loss = transition[block_loss == 1]

    transition_session = []
    transition_session_gain = []
    transition_session_loss = []
    for sess in range(1, num_sessions + 1):
        sess_inds = session_num == sess
        transition_session.append(transition[sess_inds])
        transition_session_gain.append(transition[(block_loss == 0) & sess_inds])
        transition_session_loss.append(transition[(block_loss == 1) & sess_inds])

    if plot_flag:
        plot_transition_behavior(transition, transition_gain, transition_loss, transition_session, transition_session_gain, transition_session_loss, num_sessions, trials_before, trials_after)

    return {
        'transition': list(transition),
        'transition_gain': list(transition_gain),
        'transition_loss': list(transition_loss),
        'transition_session': list(transition_session),
        'transition_session_gain': list(transition_session_gain),
        'transition_session_loss': list(transition_session_loss)
    }


def plot_transition_behavior(transition, transition_gain, transition_loss, transition_session, transition_session_gain, transition_session_loss, num_sessions, trials_before, trials_after):

    mean_transition = np.nanmean(transition, axis=0)
    sem_transition = np.nanstd(transition, axis=0) / np.sqrt(transition.shape[0])
    mean_transition_gain = np.nanmean(transition_gain, axis=0)
    sem_transition_gain = np.nanstd(transition_gain, axis=0) / np.sqrt(transition_gain.shape[0])
    mean_transition_loss = np.nanmean(transition_loss, axis=0)
    sem_transition_loss = np.nanstd(transition_loss, axis=0) / np.sqrt(transition_loss.shape[0])

    x = np.arange(-trials_before, trials_after+1)
    plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(x, mean_transition, color='k')
    plt.fill_between(x, mean_transition - sem_transition, mean_transition + sem_transition, color='k', alpha=0.3)
    plt.axvline(0, linestyle='--', color='k')
    plt.xlabel('Trials relative to probability reversal')
    plt.ylabel('P(correct)')
    plt.title('All transitions')
    plt.ylim(0, 1)
    ax1.spines[['right', 'top']].set_visible(False)

    ax2 = plt.subplot(1, 2, 2)
    plt.plot(x, mean_transition_gain, color='blue', label='Gain')
    plt.fill_between(x, mean_transition_gain - sem_transition_gain, mean_transition_gain + sem_transition_gain, color='blue', alpha=0.3)
    plt.plot(x, mean_transition_loss, color='red', label='Loss')
    plt.fill_between(x, mean_transition_loss - sem_transition_loss, mean_transition_loss + sem_transition_loss, color='red', alpha=0.3)
    plt.axvline(0, linestyle='--', color='k')
    plt.xlabel('Trials relative to probability reversal')
    plt.ylabel('P(correct)')
    plt.title('Gain vs Loss block transitions')
    plt.ylim(0, 1)
    ax2.spines[['right', 'top']].set_visible(False)
    
    plt.legend()
    plt.show()

    plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    colors = plt.cm.Greys(np.linspace(1.0, 0.4, num_sessions))
    for sess in range(num_sessions):
        mean_sess = np.nanmean(transition_session[sess], axis=0)
        sem_sess = np.nanstd(transition_session[sess], axis=0) / np.sqrt(transition_session[sess].shape[0])
        plt.plot(x, mean_sess, label=f'Session {sess+1}', color=colors[sess])
        #plt.fill_between(x, mean_sess - sem_sess, mean_sess + sem_sess, color=colors[sess], alpha=0.3)
    plt.axvline(0, linestyle='--', color='k')
    plt.xlabel('Trials relative to probability reversal')
    plt.ylabel('P(correct)')
    plt.title('All transitions')
    plt.ylim(0, 1)
    plt.legend()
    ax1.spines[['right', 'top']].set_visible(False)

    ax2 = plt.subplot(1, 3, 2)
    colors = plt.cm.Blues(np.linspace(1.0, 0.4, num_sessions))
    for sess in range(num_sessions):
        mean_sess = np.nanmean(transition_session_gain[sess], axis=0)
        sem_sess = np.nanstd(transition_session_gain[sess], axis=0) / np.sqrt(transition_session_gain[sess].shape[0])
        plt.plot(x, mean_sess, label=f'Session {sess+1}', color=colors[sess])
        #plt.fill_between(x, mean_sess - sem_sess, mean_sess + sem_sess, color=colors[sess], alpha=0.3)
    plt.axvline(0, linestyle='--', color='k')
    plt.xlabel('Trials relative to probability reversal')
    plt.ylabel('P(correct)')
    plt.title('Gain block transitions')
    plt.ylim(0, 1)
    plt.legend()
    ax2.spines[['right', 'top']].set_visible(False)

    ax3 = plt.subplot(1, 3, 3)
    colors = plt.cm.Reds(np.linspace(1.0, 0.4, num_sessions))
    for sess in range(num_sessions):
        mean_sess = np.nanmean(transition_session_loss[sess], axis=0)
        sem_sess = np.nanstd(transition_session_loss[sess], axis=0) / np.sqrt(transition_session_loss[sess].shape[0])
        plt.plot(x, mean_sess, label=f'Session {sess+1}', color=colors[sess])
        #plt.fill_between(x, mean_sess - sem_sess, mean_sess + sem_sess, color=colors[sess], alpha=0.3)
    plt.axvline(0, linestyle='--', color='k')
    plt.xlabel('Trials relative to probability reversal')
    plt.ylabel('P(correct)')
    plt.title('Loss block transitions')
    plt.ylim(0, 1)
    plt.legend()
    ax3.spines[['right', 'top']].set_visible(False)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    subselect = {'num_sessions': 3}
    analyze_transition_behavior(subselect=subselect, plot_flag=True)