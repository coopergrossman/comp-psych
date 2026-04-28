"""
created 26.2.11

analyze questionnaire consistency

@author: cgrossman
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from comp_psych.questionnaires.load import load_scores, load_subscales, load_questions, aggregate_sessions


def _prepare_df(df, questionnaire):
    df = df.copy()

    # Aggregate data across sessions for each participant
    df = aggregate_sessions(df)

    # Set participant_id as index and drop group/session columns if they exist
    if 'participant_id' in df.columns:
        df = df.set_index('participant_id')

    if 'group' in df.columns:
        df = df.drop(columns=['group'])

    if 'session' in df.columns:
        df = df.drop(columns=['session'])

    # Prefix columns with questionnaire name
    df = df.add_prefix(f"{questionnaire}_")

    return df

def _estimate_std (scores_df, subscales_df, questions_df):
    # Find standard deviation of each array in each row
    scores_std = scores_df.copy()
    cols = [c for c in scores_std.columns if c != "participant_id"]
    scores_std[cols] = scores_std[cols].applymap(lambda x: np.std(x) if isinstance(x, (list, np.ndarray, pd.Series)) else np.nan)

    subscales_std = subscales_df.copy()
    cols = [c for c in subscales_std.columns if c != "participant_id"]
    subscales_std[cols] = subscales_std[cols].applymap(lambda x: np.std(x) if isinstance(x, (list, np.ndarray, pd.Series)) else np.nan)

    questions_std = questions_df.copy()
    cols = [c for c in questions_std.columns if c != "participant_id"]
    questions_std[cols] = questions_std[cols].applymap(lambda x: np.std(x) if isinstance(x, (list, np.ndarray, pd.Series)) else np.nan)

    return scores_std, subscales_std, questions_std


def _estimate_cv (scores_df, subscales_df, questions_df):
    # Find coefficient of variation (sd / mean) of each array in each row
    scores_cv = scores_df.copy()
    cols = [c for c in scores_cv.columns if c != "participant_id"]
    scores_cv[cols] = scores_cv[cols].applymap(lambda x: np.std(x) / np.mean(x) if isinstance(x, (list, np.ndarray, pd.Series)) and np.mean(x) != 0 else np.nan)

    subscales_cv = subscales_df.copy()
    cols = [c for c in subscales_cv.columns if c != "participant_id"]
    subscales_cv[cols] = subscales_cv[cols].applymap(lambda x: np.std(x) / np.mean(x) if isinstance(x, (list, np.ndarray, pd.Series)) and np.mean(x) != 0 else np.nan)

    questions_cv = questions_df.copy()
    cols = [c for c in questions_cv.columns if c != "participant_id"]
    questions_cv[cols] = questions_cv[cols].applymap(lambda x: np.std(x) / np.mean(x) if isinstance(x, (list, np.ndarray, pd.Series)) and np.mean(x) != 0 else np.nan)

    return scores_cv, subscales_cv, questions_cv


def _estimate_mssd(scores_df, subscales_df, questions_df):

    def _mssd(x):
        if not isinstance(x, (list, np.ndarray, pd.Series)):
            return np.nan

        x = np.asarray(x, dtype=float)

        if len(x) < 2:
            return np.nan

        diffs = np.diff(x)
        return np.mean(diffs ** 2)

    scores_mssd = scores_df.copy()
    cols = scores_mssd.columns
    scores_mssd[cols] = scores_mssd[cols].applymap(_mssd)

    subscales_mssd = subscales_df.copy()
    cols = subscales_mssd.columns
    subscales_mssd[cols] = subscales_mssd[cols].applymap(_mssd)

    questions_mssd = questions_df.copy()
    cols = questions_mssd.columns
    questions_mssd[cols] = questions_mssd[cols].applymap(_mssd)

    return scores_mssd, subscales_mssd, questions_mssd
    

def plot_variance_distributions(scores_var, subscales_var, questions_var, questionnaire, variance):
    # ---------- Figure 1: Scores + Subscales ----------
    score_cols = list(scores_var.columns)
    subscale_cols = list(subscales_var.columns) if subscales_var is not None and len(subscales_var.columns) > 0 else []

    all_cols = score_cols + subscale_cols
    n_cols = len(all_cols)

    if n_cols > 0:

        n_plot_cols = int(np.ceil(np.sqrt(n_cols)))
        n_plot_rows = int(np.ceil(n_cols / n_plot_cols))

        fig, axes = plt.subplots(
            n_plot_rows,
            n_plot_cols,
            figsize=(4 * n_plot_cols, 3.5 * n_plot_rows),
            constrained_layout=True
        )

        axes = np.array(axes).reshape(-1)

        plot_idx = 0

        # Plot score variance
        for col in score_cols:
            data = scores_var[col].dropna()
            axes[plot_idx].hist(data, bins=10, density=True, color='blue')
            axes[plot_idx].set_title(col, fontsize=10)
            axes[plot_idx].set_xlabel(variance)

            axes[plot_idx].spines["top"].set_visible(False)
            axes[plot_idx].spines["right"].set_visible(False)

            plot_idx += 1

        # Plot subscale variance
        for col in subscale_cols:
            data = subscales_var[col].dropna()
            axes[plot_idx].hist(data, bins=10, density=True, color='blue')
            axes[plot_idx].set_title(col, fontsize=10)
            axes[plot_idx].set_xlabel(variance)

            axes[plot_idx].spines["top"].set_visible(False)
            axes[plot_idx].spines["right"].set_visible(False)

            plot_idx += 1

        # Hide unused axes
        for ax in axes[plot_idx:]:
            ax.set_visible(False)

        fig.suptitle(f"{questionnaire} Score/Subscale Variance", fontsize=16, y=1.02)
        plt.show(block=True)


    # ---------- Figure 2: Question Variance ----------
    question_cols = list(questions_var.columns)
    n_cols = len(question_cols)

    if n_cols > 0:

        n_plot_cols = int(np.ceil(np.sqrt(n_cols)))
        n_plot_rows = int(np.ceil(n_cols / n_plot_cols))

        fig, axes = plt.subplots(
            n_plot_rows,
            n_plot_cols,
            figsize=(4 * n_plot_cols, 3.5 * n_plot_rows),
            constrained_layout=True
        )

        axes = np.array(axes).reshape(-1)

        for i, col in enumerate(question_cols):
            data = questions_var[col].dropna()

            axes[i].hist(data, bins=10, density=True, color='blue')
            axes[i].set_title(col, fontsize=9)
            axes[i].set_xlabel(variance)

            axes[i].spines["top"].set_visible(False)
            axes[i].spines["right"].set_visible(False)

        # Hide unused axes
        for ax in axes[len(question_cols):]:
            ax.set_visible(False)

        fig.suptitle(f"{questionnaire} Question Variance", fontsize=16, y=1.02)
        plt.show(block=True)


def analyze_questionnaire_consistency(questionnaires, subselect=None, variance='cv'):
    """
    Analyze consistency of questionnaire responses across sessions by:
    - calculating variance of scores, subscales, and questions for each participant across sessions
    - plotting distributions of these variances for each questionnaire
    """

    for questionnaire in questionnaires:

        qd_scores = _prepare_df(load_scores(questionnaire, subselect=subselect), questionnaire)
        qd_subscales = _prepare_df(load_subscales(questionnaire, subselect=subselect), questionnaire)
        qd_questions = _prepare_df(load_questions(questionnaire, subselect=subselect), questionnaire)

        if variance == 'std':        # standard deviation
            scores_var, subscales_var, questions_var = _estimate_std(qd_scores, qd_subscales, qd_questions)
        elif variance == 'cv':        # coefficient of variation (sd / mean)
            scores_var, subscales_var, questions_var = _estimate_cv(qd_scores, qd_subscales, qd_questions)
        elif variance == 'mssd':    # mean squared successive difference (MSSD)
            scores_var, subscales_var, questions_var = _estimate_mssd(qd_scores, qd_subscales, qd_questions)
        
        plot_variance_distributions(scores_var, subscales_var, questions_var, questionnaire, variance)


if __name__ == "__main__":
    subselect = {'num_sessions': 3}
    questionnaires = ['dass21', 'ocir', 'spq']
    analyze_questionnaire_consistency(questionnaires, subselect=subselect, variance='mssd')