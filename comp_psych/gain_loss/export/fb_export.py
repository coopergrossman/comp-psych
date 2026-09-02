"""
created 25.11.19

export and parse data from firebase

@author: cgrossman
"""

import pandas as pd
import numpy as np
import json
from datetime import date, datetime
import firebase_admin
from firebase_admin import credentials, firestore
import os.path
from os import listdir
import csv
from pathlib import Path
from typing import Set
from comp_psych.gain_loss.config import DATA_DIR, DESIGNS_DIR
from comp_psych.core.env import FB_CREDENTIALS_FILE, DEMOGRAPHICS_DIR


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code.

    Parameters
    ----------
    obj : Any
        Object encountered by `json.dump`'s `default` hook.

    Returns
    -------
    str
        ISO-format string, for `datetime.date`/`datetime.datetime` objects.

    Raises
    ------
    TypeError
        For any other non-serializable type.
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def initialize_firebase(credentials_file):
    """Initialize the Firebase app (if not already initialized) and return a client.

    Parameters
    ----------
    credentials_file : str or pathlib.Path
        Path to a Firebase service-account JSON key file.

    Returns
    -------
    google.cloud.firestore.Client
        Firestore client for the initialized app.
    """
    if not firebase_admin._apps:
        cred = credentials.Certificate(credentials_file)
        firebase_admin.initialize_app(cred)
    return firestore.client()


def export_raw_data(cs_data, task_name, output_dir, completed_participants=None):
    """Export raw subject task data from Firestore to per-participant JSON files.

    Structure read: participants/<participant_id>/spells/<session_id>/modules/<task_name>/task-data.
    Incremental: skips sessions already present in an existing output file, so
    re-running only fetches newly-added sessions.

    Parameters
    ----------
    cs_data : google.cloud.firestore.CollectionReference
        Reference to the `participants` collection.
    task_name : str
        Module document ID under each session's `modules` subcollection
        (e.g. 'gain_loss').
    output_dir : str or pathlib.Path
        Directory under which `raw_data/<participant_id>.json` files are written.
    completed_participants : set of str, optional
        If given, only participants whose ID is in this set are exported.
    """
    raw_data_dir = f'{output_dir}/raw_data'
    Path(raw_data_dir).mkdir(parents=True, exist_ok=True)

    for participant_ref in cs_data.list_documents():
        participant_id = participant_ref.id
        if completed_participants is not None and participant_id not in completed_participants:
            continue

        # Load existing export, if any
        output_file = f'{raw_data_dir}/{participant_id}.json'
        if os.path.exists(output_file):
            with open(output_file) as f:
                participant_data = json.load(f)
            existing_sessions = set(participant_data.get('sessions', {}).keys())
        else:
            participant_data = {'participant_id': participant_id, 'sessions': {}}
            existing_sessions = set()

        updated = False
        for session_ref in participant_ref.collection('spells').list_documents():
            session_id = session_ref.id
            if session_id in existing_sessions:
                continue  # skip the expensive part: modules + task-data reads

            task_ref = session_ref.collection('modules').document(task_name)
            task_data_docs = list(task_ref.collection('task-data').stream())
            if not task_data_docs:
                continue

            task_data_list = []
            for doc in task_data_docs:
                d = doc.to_dict()
                d['_doc_id'] = doc.id
                task_data_list.append(d)

            participant_data['sessions'][session_id] = task_data_list
            updated = True

        if updated:
            with open(output_file, 'w') as f:
                json.dump(participant_data, f, indent=2, default=json_serial)

def load_json_to_dataframe(json_file_path):
    """Load a single participant's raw-export JSON into a flat DataFrame.

    Parameters
    ----------
    json_file_path : str or pathlib.Path
        Path to a `raw_data/<participant_id>.json` file written by
        `export_raw_data`.

    Returns
    -------
    pandas.DataFrame
        One row per trial document, with `participant_id` and `session_id`
        columns prepended to that document's own fields.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    participant_id = data['participant_id']
    sessions = data['sessions']
    
    # Flatten the nested structure
    all_rows = []
    for session_id, task_data_list in sessions.items():
        for task_doc in task_data_list:
            row = {
                'participant_id': participant_id,
                'session_id': session_id,
                **task_doc  # Unpack all fields from the task document
            }
            all_rows.append(row)
    
    return pd.DataFrame(all_rows)

def add_fields_gain_loss(df):
    """Derive gain_loss task-design fields by joining trial data to design CSVs.

    Adds `is_win`, `pWin_1`, `pWin_2`, `stimIndex_1`, `stimIndex_2`,
    `block_change`, `prob_reversal`, `is_correct`, and `choice`, looked up
    per trial from the design files under `gain_loss.config.DESIGNS_DIR`
    matching each session's `designNo`.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw trial data for one participant, as returned by
        `load_json_to_dataframe`.

    Returns
    -------
    pandas.DataFrame
        `df` with the derived columns added, or an empty DataFrame if `df`
        has no `designNo` column or its trial count doesn't match the
        concatenated design files (a signal of corrupted/incomplete data).
    """

    # Only assess subjects with any sessions with design number
    if 'designNo' in df.columns:
        # Remove sessions with missing design number
        if df['designNo'].isna().any():
            df = df.dropna(subset=['designNo'])

        # Add field to inicate win/loss
        df['is_win'] = ((df['trialType'] == 'gain') & (df['amount'] == 25)) | ((df['trialType'] == 'loss') & (df['amount'] == 0))
        df['is_win'] = df['is_win'].astype('boolean')
        df.loc[df['amount'].isna(), 'is_win'] = pd.NA

        # Add field to indicate correct/incorrect choice, reward probabilities, stimulus positions, and block changes
        design_numbers = df['designNo'][df['practice'].diff().fillna(0) == -1].to_numpy()
        correct_choice = np.array([])
        pWin_1 = np.array([])
        pWin_2 = np.array([])
        stimIndex_1 = np.array([])
        stimIndex_2 = np.array([])
        block_change = np.array([])
        prob_reversal = np.array([])
        choice_A = np.array([])
        for design_number in design_numbers:
            practice_trials = pd.read_csv(f'{DESIGNS_DIR}/practice{int(design_number)}.csv')
            main_trials = pd.read_csv(f'{DESIGNS_DIR}/set{int(design_number)}.csv')
            all_trials = pd.concat([practice_trials, main_trials], ignore_index=True)

            pWin_1 = np.concatenate((pWin_1, all_trials['pWin_1'].to_numpy()), axis=0)
            pWin_2 = np.concatenate((pWin_2, all_trials['pWin_2'].to_numpy()), axis=0)
            stimIndex_1 = np.concatenate((stimIndex_1, all_trials['stimIndex_1'].to_numpy()), axis=0)
            stimIndex_2 = np.concatenate((stimIndex_2, all_trials['stimIndex_2'].to_numpy()), axis=0)

            block_change = np.concatenate((block_change, all_trials['blockID'].diff() != 0), axis=0)
            prob_reversal = np.concatenate((prob_reversal, (all_trials['pWin_1'].diff() != 0) | (all_trials['blockID'].diff() != 0)), axis=0)

            stimIDs = all_trials[['stimID_1', 'stimID_2']].to_numpy()
            stim_idx = all_trials[['stimIndex_1', 'stimIndex_2']].to_numpy().astype(int) - 1
            stimIDs = stimIDs[np.arange(len(stimIDs))[:, None], stim_idx]
            high_prob = all_trials['pWin_1'] < all_trials['pWin_2']
            correct_choice = np.concatenate((correct_choice, np.where(high_prob, stimIDs[:, 1], stimIDs[:, 0])), axis=0)
            choice_A = np.concatenate((choice_A, stimIDs[:, 0]), axis=0)
        
        if len(correct_choice) != len(df):
            df = pd.DataFrame()
        else:
            df['pWin_1'] = pWin_1
            df['pWin_2'] = pWin_2
            df['stimIndex_1'] = stimIndex_1
            df['stimIndex_2'] = stimIndex_2
            df['block_change'] = block_change
            df['prob_reversal'] = prob_reversal
            df['is_correct'] = df['chosenStimID'].to_numpy() == correct_choice
            df['is_correct'] = df['is_correct'].astype('boolean')
            df.loc[df['amount'].isna(), 'is_correct'] = pd.NA
            df['choice'] = (df['chosenStimID'] == choice_A).astype(int)


    else:
        df = pd.DataFrame()

    return df


def create_dataframe(raw_data_dir, completed_participants=None):
    """Load and concatenate all participants' raw JSON exports into one DataFrame.

    Parameters
    ----------
    raw_data_dir : str or pathlib.Path
        Directory of `<participant_id>.json` files (as written by `export_raw_data`).
    completed_participants : set of str, optional
        If given, only participants whose ID (JSON filename stem) is in this
        set are included.

    Returns
    -------
    pandas.DataFrame
        Combined trial-level data across participants, with design fields
        added via `add_fields_gain_loss`. Empty if no participant yielded data.
    """
    all_dfs = []
    
    json_files = [f for f in listdir(raw_data_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        if completed_participants is None or json_file.split('.')[0] in completed_participants:
            file_path = os.path.join(raw_data_dir, json_file)
            df = load_json_to_dataframe(file_path)
            df = add_fields_gain_loss(df)
            if not df.empty:
                all_dfs.append(df)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Total participants: {combined_df['participant_id'].nunique()}")
        return combined_df
    else:
        print("No data loaded")
        return pd.DataFrame()


def load_all_completed_participants():
    """Load approved participant IDs from every inclusion list in DEMOGRAPHICS_DIR.

    Unlike the questionnaire export's `load_completed_participants` (which
    reads a single group's list), this reads every file matching
    `inclusion_list*.csv` in `DEMOGRAPHICS_DIR`, since this export isn't
    scoped to one spell/group.

    Returns
    -------
    set of str
        Approved participant IDs, unioned across all inclusion list files.
        Empty if no inclusion list files are found (no filtering applied).
    """

    # Load master subject inclusion lists (any file that includes 'inclusion_list')
    csv_filenames = [f for f in os.listdir(DEMOGRAPHICS_DIR) if 'inclusion_list' in f and f.endswith('.csv')]
    
    if not csv_filenames:
        print(f"[Export] Warning: No subject inclusion list files found in {DEMOGRAPHICS_DIR}")
        print(f"[Export] Proceeding without filtering by completion status")
        return set()
    
    completed_participants: Set[str] = set()
    
    for csv_filename in csv_filenames:
        csv_path = os.path.join(DEMOGRAPHICS_DIR, csv_filename)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prolific_id = row.get("participant_id", "").strip()
                
                # If in the master inclusion list, add to set
                if prolific_id:
                    completed_participants.add(prolific_id)
    
    print(f"[Export] Loaded {len(completed_participants)} participant(s) with all modules complete and status approved from {len(csv_filenames)} file(s)")
    return completed_participants


def fb_export_gain_loss(force_refresh=False):
    """Export gain_loss data from Firestore (if needed) and rebuild all_data.parquet.

    Parameters
    ----------
    force_refresh : bool, default False
        If True, re-download raw data from Firestore even if local raw JSON
        files already exist. If False and raw files already exist, skips
        the Firestore fetch and only re-parses the existing JSON.
    """
    task_name = 'gain_loss'
    output_dir = DATA_DIR
    credentials_file = FB_CREDENTIALS_FILE
    raw_data_dir = f'{output_dir}/raw_data'

    # Check if directory exists and has files
    has_existing_files = (
        os.path.exists(raw_data_dir) and 
        len([f for f in os.listdir(raw_data_dir) if os.path.isfile(os.path.join(raw_data_dir, f))]) > 0
    )

    # Find included subjects
    completed_participants = load_all_completed_participants()
    
    if not has_existing_files or force_refresh:
        # Initialize Firebase
        client = initialize_firebase(credentials_file)
        cs_data = client.collection('participants')

        # Export raw data
        print("Exporting raw data...")
        export_raw_data(cs_data, task_name, output_dir, completed_participants)
    
    # Parse subject data
    print("Creating dataframe...")
    combined_df = create_dataframe(raw_data_dir, completed_participants)
    combined_df.to_parquet(f'{output_dir}/all_data.parquet', index=False)


if __name__ == "__main__":
    force_refresh = True
    fb_export_gain_loss(force_refresh = force_refresh)