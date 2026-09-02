"""
created 26.1.12

gain-loss win-stay behavior change over sessions analysis

@author: cgrossman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from comp_psych.gain_loss.analyses.behavior.analyze_wsls import analyze_wsls

def analyze_wsls_change(subselect=None, plot_flag=True):
    """Compute each subject's last-minus-first-session change in WSLS measures.

    Parameters
    ----------
    subselect : dict, optional
        Filter criteria passed through to `analyze_wsls`.
    plot_flag : bool, default True
        If True, show histograms of each change measure with one-sample
        t-tests against zero, via `plot_wsls_change`.

    Returns
    -------
    data : pandas.DataFrame
        The loaded, subselected trial-level data (from `analyze_wsls`).
    summary : pandas.DataFrame
        `analyze_wsls`'s per-subject summary, with `*_diff` columns added
        (last session value minus first session value) for each of
        win_stay, lose_shift, and their gain/loss splits.
    """

    # Load data
    data, wsls = analyze_wsls(subselect=subselect, plot_flag=False)

    # Calculate change from first to last session
    wsls['win_stay_diff'] = wsls['win_stay'].apply(lambda arr: arr[-1] - arr[0])
    wsls['lose_shift_diff'] = wsls['lose_shift'].apply(lambda arr: arr[-1] - arr[0])
    wsls['win_stay_gain_diff'] = wsls['win_stay_gain'].apply(lambda arr: arr[-1] - arr[0])
    wsls['lose_shift_gain_diff'] = wsls['lose_shift_gain'].apply(lambda arr: arr[-1] - arr[0])
    wsls['win_stay_loss_diff'] = wsls['win_stay_loss'].apply(lambda arr: arr[-1] - arr[0])
    wsls['lose_shift_loss_diff'] = wsls['lose_shift_loss'].apply(lambda arr: arr[-1] - arr[0])

    # Plot results
    if plot_flag:
        plot_wsls_change(wsls)

    return data, wsls

def plot_wsls_change(wsls):
    """Plot histograms of each WSLS-change measure with one-sample t-test p-values.

    Parameters
    ----------
    wsls : pandas.DataFrame
        Per-subject summary with `*_diff` columns, as returned by
        `analyze_wsls_change`.
    """
    from scipy.stats import ttest_1samp
    
    # Configuration for each subplot: (column, color, xlabel, title)
    plots = [
        ('win_stay_diff', 'blue', 'Win-Stay Change', 'All'),
        ('win_stay_gain_diff', 'blue', 'Win-Stay Change', 'Gain Block'),
        ('win_stay_loss_diff', 'blue', 'Win-Stay Change', 'Loss Block'),
        ('lose_shift_diff', 'red', 'Lose-Shift Change', 'All'),
        ('lose_shift_gain_diff', 'red', 'Lose-Shift Change', 'Gain Block'),
        ('lose_shift_loss_diff', 'red', 'Lose-Shift Change', 'Loss Block'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()
    plt.suptitle('Win-Stay/Lose-Shift Change: last session - first session', fontsize=16)

    bins = np.linspace(-1, 1, 21)

    for ax, (col, color, xlabel, title) in zip(axes, plots):
        data = wsls[col]
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
    analyze_wsls_change(subselect=subselect, plot_flag=True)