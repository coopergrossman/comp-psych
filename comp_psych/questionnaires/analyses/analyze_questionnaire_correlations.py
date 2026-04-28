"""
created 26.2.10 

analyze questionnaire correlations

@author: cgrossman
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from comp_psych.questionnaires.load import load_scores, load_subscales, load_questions


def _prepare_df(df, questionnaire):
    df = df.copy()

    # Create unique identifier from participant_id and session
    if 'participant_id' in df.columns and 'session' in df.columns:
        df['participant_session'] = (
            df['participant_id'].astype(str) + "_" + df['session'].astype(str)
        )
        df = df.drop(columns=['participant_id', 'session'])
        df = df.set_index('participant_session')

    elif 'participant_id' in df.columns:
        df = df.set_index('participant_id')

    if 'group' in df.columns:
        df = df.drop(columns=['group'])

    # Prefix columns with questionnaire name
    df = df.add_prefix(f"{questionnaire}_")

    return df


def _plot_corr_matrix(ax, df, title):
    corr = df.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Apply mask by setting upper triangle to NaN
    corr_masked = corr.mask(mask)

    im = ax.imshow(corr, aspect="auto", cmap='cool')

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if mask[i, j]:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color="white"))

    ax.set_title(title, fontsize=12)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))

    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.columns, fontsize=6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return im


def plot_correlations(scores_df, subscales_df, questions_df):

    scores_combined = pd.concat(scores_df, axis=1, join="outer")
    scores_subscales_combined = pd.concat(scores_df + subscales_df, axis=1, join="outer")
    all_combined = pd.concat(scores_df + subscales_df + questions_df, axis=1, join="outer")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    im1 = _plot_corr_matrix(axes[0], scores_combined, "Scores")
    im2 = _plot_corr_matrix(axes[1], scores_subscales_combined, "Scores + Subscales")
    im3 = _plot_corr_matrix(axes[2], all_combined, "Scores + Subscales + Questions")

    fig.suptitle("Questionnaire Correlations", fontsize=16)

    fig.colorbar(im3, ax=axes.ravel().tolist(), shrink=0.8)

    plt.show(block=True)


def analyze_questionnaire_correlations(questionnaires, subselect=None):

    scores_list = []
    subscales_list = []
    questions_list = []

    for questionnaire in questionnaires:

        qd_scores = _prepare_df(load_scores(questionnaire, subselect=subselect), questionnaire)
        qd_subscales = _prepare_df(load_subscales(questionnaire, subselect=subselect), questionnaire)
        qd_questions = _prepare_df(load_questions(questionnaire, subselect=subselect), questionnaire)

        scores_list.append(qd_scores)
        subscales_list.append(qd_subscales)
        questions_list.append(qd_questions)

    plot_correlations(scores_list, subscales_list, questions_list)


if __name__ == "__main__":
    subselect = {'num_sessions': 3}
    questionnaires = ['dass21', 'ocir', 'spq']
    analyze_questionnaire_correlations(questionnaires, subselect=subselect)