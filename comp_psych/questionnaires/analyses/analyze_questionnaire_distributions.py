"""
created 26.2.4 

analyze questionnaire response distributions

@author: cgrossman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from comp_psych.questionnaires.load import load_scores, load_subscales, load_questions


def plot_response_distributions(df, df_name, questionnaire):
    """
    Plot histograms for each numeric column in a dataframe with:
    - best-fit normal curve
    - mean and median vertical lines
    - KS test for normality
    - skewness
    """

    # Keep only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    n_cols = df_numeric.shape[1]
    if n_cols == 0:
        print(f"No numeric columns in {df_name}")
        return

    # Determine subplot grid
    n_plot_cols = int(np.ceil(np.sqrt(n_cols)))
    n_plot_rows = int(np.ceil(n_cols / n_plot_cols))

    fig, axes = plt.subplots(
        n_plot_rows,
        n_plot_cols,
        figsize=(5 * n_plot_cols, 4 * n_plot_rows),
        constrained_layout=True
    )

    fig.suptitle(f"{questionnaire} {df_name}", fontsize=16)

    axes = np.array(axes).reshape(-1)  # flatten safely

    for ax, col in zip(axes, df_numeric.columns):

        data = df_numeric[col].dropna()

        if len(data) == 0:
            ax.set_visible(False)
            continue

        # Histogram
        bins = np.arange(data.min() - 0.5, data.max() + 1.5, 1)
        ax.hist(data, bins=bins, density=True, alpha=0.6, color='blue')

        # ----- Fit normal distribution -----
        mu, sigma = stats.norm.fit(data)

        x = np.linspace(data.min(), data.max(), 200)
        pdf = stats.norm.pdf(x, mu, sigma)

        ax.plot(x, pdf, color='red')

        # ----- Mean and Median -----
        mean_val = data.mean()
        median_val = data.median()

        ax.axvline(mean_val, linestyle="--")
        ax.axvline(median_val, linestyle=":")

        # ----- KS Test against fitted normal -----
        ks_stat, ks_p = stats.kstest(data, "norm", args=(mu, sigma))

        # ----- Skewness -----
        skew_val = stats.skew(data)

        # ----- Annotation -----
        text = (
            f"KS p = {ks_p:.3g}\n"
            f"Skew = {skew_val:.3g}\n"
            f"μ = {mean_val:.2f}\n"
            f"Median = {median_val:.2f}"
        )

        ax.text(
            0.98,
            0.98,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", alpha=0.2)
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(col, fontsize=8)

    # Turn off unused axes
    for ax in axes[n_cols:]:
        ax.set_visible(False)
    
    plt.show(block=True)

def analyze_questionnaire_distributions(questionnaire, subselect=None):

    # Load questionnaire data
    qd_scores = load_scores(questionnaire, subselect=subselect)
    qd_subscales = load_subscales(questionnaire, subselect=subselect)
    qd_questions = load_questions(questionnaire, subselect=subselect)

    qds = {
        "Scores": qd_scores,
        "Subscales": qd_subscales,
        "Questions": qd_questions
    }

    # Plot distributions for each metric separately
    for name, qd in qds.items():
        # Remove sessions column
        if 'session' in qd.columns:
            qd = qd.drop(columns=['session'])
        plot_response_distributions(qd, name, questionnaire)


if __name__ == "__main__":
    subselect = {'num_sessions': 3}
    questionnaire = 'ocir'
    analyze_questionnaire_distributions(questionnaire, subselect=subselect)

