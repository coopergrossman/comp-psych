"""
created 26.1.8

fit stan models to behavior

@author: cgrossman
"""

import numpy as np
import pandas as pd
import os
from comp_psych.gain_loss.load import load_gain_loss_data
from comp_psych.core.modeling import compute_map_estimates
from comp_psych.gain_loss.modeling import get_param_names
from comp_psych.gain_loss.config import MODEL_DIR, MODEL_SAVE_DIR
from cmdstanpy import CmdStanModel


def format_data(data, subjects):
    data['action'] = (data['chosenStimID'] == data['rightStimID']).astype(int)
    data['block_loss'] = (data['trialType'] == 'loss').astype(int)
    return data

def save_stan_outputs(save_path, samples, summary, param_estimates, subj_param_estimates):
    # Save numeric arrays (NumPy-stable)
    np.savez_compressed(
        os.path.join(save_path, 'param_estimates.npz'),
        param_estimates=param_estimates,
        subj_param_estimates=subj_param_estimates,
    )
    # Save samples as parquet to preserve column types
    samples.to_parquet(os.path.join(save_path, 'samples.parquet'))

    # Save summary table
    summary.to_csv(os.path.join(save_path, "summary.csv"), index=False)

def fit_stan_model(
        subselect=None, 
        iter=2000, 
        warmup=1000, 
        num_chains=4, 
        delta=0.9, 
        model_name='q', 
        save_flag=True, 
        force_rerun=False
):
    
    # Load all data
    data = load_gain_loss_data(subselect=subselect, subselect_defaults=True)
    subjects = data['participant_id'].unique()
    data = format_data(data, subjects)                  # Format variables for Stan model

    # Compile Stan model
    stan_model_path = f"{MODEL_DIR}/{model_name}.stan"
    stan_model = CmdStanModel(stan_file=stan_model_path)
    param_names = get_param_names(model_name)

    # Run for each subject separately
    for s_ind, subject in enumerate(subjects):
        print(f"Fitting subject {s_ind+1}/{len(subjects)}...")

        data_subj = data[data['participant_id'] == subject]
        num_sessions = data_subj['session'].nunique()

        save_path = f"{MODEL_SAVE_DIR}/{model_name}/{subject}"

        # Skip subjects with only one session or if the subject has already been run and not forcing rerun
        if num_sessions <= 2 or (os.path.exists(save_path) and not force_rerun):
            continue

        # Format data for Stan input            
        N = num_sessions
        Tsesh =  data_subj['session'].value_counts().sort_index().to_numpy()
        T = Tsesh.max()

        # Stan variables must have fixed dimensions, so we pad with zeros
        choice = np.zeros((N, T), dtype=int)               # 0 for stimulus 1, 1 for stimulus 2
        action = np.zeros((N, T), dtype=int)               # 0 for left, 1 for right
        outcome = np.zeros((N, T), dtype=int)              # 0 for no win, 1 for win
        block_change = np.zeros((N, T), dtype=int)         # 0 for no change, 1 for change
        block_loss = np.zeros((N, T), dtype=int)           # 0 for gain, 1 for loss

        # Loop through sessions to fill in data
        for session in range(N):
            session_data = data_subj[data_subj['session'] == session+1]
            n_trials = len(session_data)

            choice[session, :n_trials] = session_data['choice'].values
            action[session, :n_trials] = session_data['action'].values
            outcome[session, :n_trials] = session_data['is_win'].values
            block_change[session, :n_trials] = session_data['block_change'].values
            block_loss[session, :n_trials] = session_data['block_loss'].values
            
        # Define Stan input data
        input_data = {
            'N': N,
            'T': T,
            'Tsesh': Tsesh,
            'choice': choice,
            'action': action,
            'outcome': outcome,
            'block_change': block_change,
            'block_loss': block_loss,
        }

        # Fit Stan model
        fit = stan_model.sample(data=input_data, 
                                iter_sampling=iter, 
                                iter_warmup=warmup, 
                                chains=num_chains, 
                                adapt_delta=delta, 
                                show_progress=True,
                                output_dir=save_path
                                )
        
        samples = fit.draws_pd()
        summary = fit.summary()

        param_estimates = compute_map_estimates(samples, param_names, type='session', num_sessions=num_sessions)
        subj_param_estimates = compute_map_estimates(samples, param_names, type='subject')

        if save_flag:
            save_stan_outputs(save_path, samples, summary, param_estimates, subj_param_estimates)


if __name__ == "__main__":
    subselect = {'participant_id': ['5a09ebdf087f2e0001eae39f']}
    fit_stan_model(
        subselect=subselect, 
        iter=100, 
        warmup=100, 
        model_name='q_a_win_lose_loss_gain_forget', 
        force_rerun=True
    )