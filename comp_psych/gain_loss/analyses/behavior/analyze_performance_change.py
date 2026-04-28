"""
created 26.1.12

gain-loss behavior performance change over sessions analysis

@author: cgrossman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from compPsych.gain_loss.analyses.behavior.analyze_performance import analyze_performance

def analyze_performance_change(subselect=None, plot_flag=True):
    
    # Load data
    data, bp = analyze_performance(subselect=subselect, plot_flag=False)

    # Calculate change from first to last session
    bp['win_prob_diff'] = bp['win_prob'].apply(lambda arr: arr[-1] - arr[0])
    bp['corr_prob_diff'] = bp['corr_prob'].apply(lambda arr: arr[-1] - arr[0])
    bp['win_prob_gain_diff'] = bp['win_prob_gain'].apply(lambda arr: arr[-1] - arr[0])
    bp['win_prob_loss_diff'] = bp['win_prob_loss'].apply(lambda arr: arr[-1] - arr[0])
    bp['corr_prob_gain_diff'] = bp['corr_prob_gain'].apply(lambda arr: arr[-1] - arr[0])
    bp['corr_prob_loss_diff'] = bp['corr_prob_loss'].apply(lambda arr: arr[-1] - arr[0])
    # Plot results
    if plot_flag:
        plot_performance_change(bp)

    return data, bp

def plot_performance_change(bp):
    from scipy.stats import ttest_1samp
    
    # Configuration for each subplot: (column, color, xlabel, title)
    plots = [
        ('win_prob_diff', 'blue', 'Win Probability Change', 'All'),
        ('win_prob_gain_diff', 'blue', 'Win Probability Change', 'Gain Block'),
        ('win_prob_loss_diff', 'blue', 'Win Probability Change', 'Loss Block'),
        ('corr_prob_diff', 'green', 'Correct Choice Probability Change', 'All'),
        ('corr_prob_gain_diff', 'green', 'Correct Choice Probability Change', 'Gain Block'),
        ('corr_prob_loss_diff', 'green', 'Correct Choice Probability Change', 'Loss Block'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()
    plt.suptitle('Behavior performance change: last session - first session', fontsize=16)

    bins = np.linspace(-1, 1, 21)

    for ax, (col, color, xlabel, title) in zip(axes, plots):
        data = bp[col]
        ax.hist(data, bins=bins, color=color, alpha=0.7)
        t_stat, p_val = ttest_1samp(data, 0)
        if p_val < 0.05:
            ax.text(
                0.95, 0.95,
                f"p = {p_val:.3f}",
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Number of Subjects')
        ax.set_title(title)
        ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
    plt.show()


if __name__ == "__main__":
    subselect = {'num_sessions': 3}
    analyze_performance_change(subselect=subselect, plot_flag=True)