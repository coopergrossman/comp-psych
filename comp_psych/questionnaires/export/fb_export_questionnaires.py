#!/usr/bin/env python3
"""
Export questionnaire module data from Firestore.

This script downloads questionnaire module documents from Firestore and
exports them to CSV files in a long (item-level) format.

Firestore path pattern:
    participants/{prolificId}/spells/{sessionDocId}/modules/{moduleNumber}_{questionnaire}

By default, only participants who are approved and complete (as defined
by the subject inclusion list) are included. This filtering can be bypassed
using the --all flag.

Outputs:
    - Individual CSV files (optional):
        q_{questionnaire}_{prolificId}_{spell}.csv
    - Aggregated CSV file:
        q_{questionnaire}_{spell}.csv

CSV columns:
    sessionId, groupId, prolificId,
    item, prompt, type, options, value,
    changes, distinct_values, duration_ms, first_interaction_ms,
    hover_ms, indecision_per_min, shown_at_ms, time_on_final_ms, toggles,
    moduleName, moduleId, uid

Subject inclusion list:
    To enable filtering by completion/approval status, place the inclusion
    list CSV at:

        DEMOGRAPHICS_DIR/inclusion_list_{groupId}.csv

    Only participants listed in this file will be included unless --all
    is specified. If the file is missing, the script proceeds without
    filtering.

Basic usage:
    # Export DASS-21 questionnaire data (aggregated output only)
    python -m questionnaires.export.fb_export_questionnaire \
        --spell s1_groupA \
        --questionnaire dass21

Examples:
    # Specify output directory
    python -m questionnaires.export.fb_export_questionnaire \
        --spell s1_groupA \
        --questionnaire dass21 \
        --out q_dass21/

    # Export for a specific participant
    python -m questionnaires.export.fb_export_questionnaire \
        --spell s1_groupA \
        --questionnaire dass21 \
        --prolific-id PARTICIPANT_ID

    # Bypass subject inclusion filtering
    python -m questionnaires.export.fb_export_questionnaire \
        --spell s1_groupA \
        --questionnaire dass21 \
        --all

    # Output modes
    python -m questionnaires.export.fb_export_questionnaire \
        --spell s1_groupA \
        --questionnaire dass21 \
        --output-mode individual

    python -m questionnaires.export.fb_export_questionnaire \
        --spell s1_groupA \
        --questionnaire dass21 \
        --output-mode both

    # Specify Firebase credentials
    python -m questionnaires.export.fb_export_questionnaire \
        --spell s1_groupA \
        --questionnaire dass21 \
        --credentials path/to/credentials.json

Arguments:
    --spell (required)
        Session document ID (e.g., 's1_groupA', '0_groupZ').
        The 's' prefix is optional.

    --questionnaire (required)
        Questionnaire ID (e.g., 'dass21', 'ocir', 'spq', 'demography').

    --out (optional)
        Output directory for CSV files (default: QUESTIONNAIRE_DIR).

    --prolific-id (optional)
        Restrict export to a single participant.

    --all (optional)
        Disable filtering by subject inclusion list.

    --output-mode (optional)
        One of:
            'aggregated'  – aggregated CSV only (default)
            'individual'  – individual participant CSVs only
            'both'        – both individual and aggregated CSVs

    --credentials (optional)
        Path to Firebase service account JSON key file.
        If omitted, the script will attempt to use:
            1. GOOGLE_APPLICATION_CREDENTIALS (recommended), or
            2. Auto-detection of Firebase credential files in the current directory.
"""

import os, csv, json, argparse, re
import warnings
import sys
import glob
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
from comp_psych.core.env import FB_CREDENTIALS_FILE, DEMOGRAPHICS_DIR, QUESTIONNAIRE_DIR


@contextmanager
def suppress_stderr():
    """Temporarily suppress stderr output at file descriptor level."""
    try:
        # Save original stderr file descriptor
        old_stderr_fd = sys.stderr.fileno()
        old_stderr = sys.stderr
        
        # Open /dev/null and redirect stderr to it
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, old_stderr_fd)
        os.close(devnull_fd)
        
        # Also redirect Python's sys.stderr
        sys.stderr = open(os.devnull, "w")
        
        yield
        
    finally:
        # Restore original stderr
        sys.stderr.close()
        sys.stderr = old_stderr
        os.dup2(old_stderr.fileno(), old_stderr_fd)


# Suppress Google Cloud library warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress stderr during Firestore imports (warnings are emitted at import time)
with suppress_stderr():
    from google.cloud import firestore
    from google.oauth2 import service_account


def to_serializable(x: Any) -> Any:
    """Convert Firestore types to JSON-serializable types."""
    if isinstance(x, datetime):
        if x.tzinfo is None:
            x = x.replace(tzinfo=timezone.utc)
        return x.isoformat()
    try:
        from google.cloud.firestore import DocumentReference, GeoPoint
        if isinstance(x, DocumentReference):
            return x.path
        if isinstance(x, GeoPoint):
            return {"_geopoint": {"latitude": x.latitude, "longitude": x.longitude}}
    except Exception:
        pass
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, list):
        return [to_serializable(v) for v in x]
    if isinstance(x, dict):
        return {k: to_serializable(v) for k, v in x.items()}
    return x


def transform_to_long_format(
        module_data: Dict[str, Any],
        questionnaire_config: Dict[str, Any]
        ) -> List[Dict[str, Any]]:
    """
    Transform module data into long format: one row per item in perItem array.
    
    Returns a list of rows with columns:
    sessionId, groupId, prolificId, item, prompt, type, options, value,
    changes, distinct_values, duration_ms, first_interaction_ms, hover_ms,
    indecision_per_min, shown_at_ms, time_on_final_ms, toggles,
    moduleName, moduleId, uid
    """
    rows = []
    
    # Extract metadata (same for all rows from this participant)
    session_id = module_data.get("sessionId", "")
    group_id = module_data.get("groupId", "")
    prolific_id = module_data.get("prolificId", "")
    
    # Extract payload data
    payload = module_data.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}
    
    # Extract module info
    module_name = payload.get("id", module_data.get("id", ""))
    if module_name is None:
        module_name = ""
    
    module_id = payload.get("index", module_data.get("index", ""))
    if module_id is None:
        module_id = ""
    
    uid = payload.get("ownerUid", module_data.get("ownerUid", ""))
    if uid is None:
        uid = ""
    
    # Extract telemetry data
    telemetry = payload.get("telemetry", {})
    if not isinstance(telemetry, dict):
        telemetry = {}
    telemetry_items = telemetry.get("items", {})
    if not isinstance(telemetry_items, dict):
        telemetry_items = {}
    
    # Extract perItem array
    per_item = payload.get("perItem", [])
    if not isinstance(per_item, list):
        per_item = []
    
    # Create one row per item in perItem array
    for item_data in per_item:
        if not isinstance(item_data, dict):
            continue
        
        item_id = item_data.get("id", "")
        if item_id is None:
            item_id = ""
        
        prompt = item_data.get("prompt", "")
        if prompt is None:
            prompt = ""
        
        # type comes from 'sad' field
        item_type = item_data.get(questionnaire_config["type_field"], "")
        if item_type is None:
            item_type = ""
        
        # options field - serialize if it's a list or dict
        options = item_data.get("options", "")
        if options is None:
            options = ""
        elif isinstance(options, (list, dict)):
            # Convert to JSON string for CSV
            options = json.dumps(to_serializable(options))
        else:
            options = str(options)
        
        value = item_data.get("value", "")
        if value is None:
            value = ""
        
        # Extract telemetry data for this item
        item_telemetry = telemetry_items.get(item_id, {})
        if not isinstance(item_telemetry, dict):
            item_telemetry = {}
        
        changes = item_telemetry.get("changes", "")
        if changes is None:
            changes = ""
        elif isinstance(changes, (list, dict)):
            changes = json.dumps(to_serializable(changes))
        else:
            changes = str(changes)
        
        distinct_values = item_telemetry.get("distinct_values", "")
        if distinct_values is None:
            distinct_values = ""
        elif isinstance(distinct_values, (list, dict)):
            distinct_values = json.dumps(to_serializable(distinct_values))
        else:
            distinct_values = str(distinct_values)
        
        duration_ms_item = item_telemetry.get("duration_ms", "")
        if duration_ms_item is None:
            duration_ms_item = ""
        
        first_interaction_ms = item_telemetry.get("first_interaction_ms", "")
        if first_interaction_ms is None:
            first_interaction_ms = ""
        
        hover_ms = item_telemetry.get("hover_ms", "")
        if hover_ms is None:
            hover_ms = ""
        
        indecision_per_min = item_telemetry.get("indecision_per_min", "")
        if indecision_per_min is None:
            indecision_per_min = ""
        
        shown_at_ms = item_telemetry.get("shown_at_ms", "")
        if shown_at_ms is None:
            shown_at_ms = ""
        
        time_on_final_ms = item_telemetry.get("time_on_final_ms", "")
        if time_on_final_ms is None:
            time_on_final_ms = ""
        
        toggles = item_telemetry.get("toggles", "")
        if toggles is None:
            toggles = ""
        elif isinstance(toggles, (list, dict)):
            toggles = json.dumps(to_serializable(toggles))
        else:
            toggles = str(toggles)
        
        rows.append({
            "sessionId": session_id,
            "groupId": group_id,
            "prolificId": prolific_id,
            "item": item_id,
            "prompt": prompt,
            "type": item_type,
            "options": options,
            "value": value,
            "changes": changes,
            "distinct_values": distinct_values,
            "duration_ms": duration_ms_item,
            "first_interaction_ms": first_interaction_ms,
            "hover_ms": hover_ms,
            "indecision_per_min": indecision_per_min,
            "shown_at_ms": shown_at_ms,
            "time_on_final_ms": time_on_final_ms,
            "toggles": toggles,
            "moduleName": module_name,
            "moduleId": module_id,
            "uid": uid
        })
    
    # Add summary rows for module-level information
    # Extract summary values
    duration_ms_module = payload.get("duration_ms", "")
    if duration_ms_module is None:
        duration_ms_module = ""
    
    scores = payload.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}
    
    count = scores.get("count", "")
    if count is None:
        count = ""
    
    total = scores.get("total", "")
    if total is None:
        total = ""
    
    started_at = payload.get("startedAt", "")
    if isinstance(started_at, datetime):
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        started_at = started_at.isoformat()
    elif started_at is None:
        started_at = ""
    
    submitted_at = payload.get("submittedAt", "")
    if isinstance(submitted_at, datetime):
        if submitted_at.tzinfo is None:
            submitted_at = submitted_at.replace(tzinfo=timezone.utc)
        submitted_at = submitted_at.isoformat()
    elif submitted_at is None:
        submitted_at = ""
    
    # Add summary rows (prompt, type, options empty)
    summary_items = [
        ("duration_ms", duration_ms_module),
        ("count", count),
        ("total", total),
        ("startedAt", started_at),
        ("submittedAt", submitted_at)
    ]
    
    for item_name, item_value in summary_items:
        rows.append({
            "sessionId": session_id,
            "groupId": group_id,
            "prolificId": prolific_id,
            "item": item_name,
            "prompt": "",
            "type": "",
            "options": "",
            "value": str(item_value) if item_value is not None else "",
            "changes": "",
            "distinct_values": "",
            "duration_ms": "",
            "first_interaction_ms": "",
            "hover_ms": "",
            "indecision_per_min": "",
            "shown_at_ms": "",
            "time_on_final_ms": "",
            "toggles": "",
            "moduleName": module_name,
            "moduleId": module_id,
            "uid": uid
        })
    
    if not questionnaire_config.get("subscales"):
        return rows
    
    # Add subscale rows (prompt, options empty, but type has A/D/S)
    subscales = scores.get("subscales", {})
    if not isinstance(subscales, dict):
        subscales = {}
    
    for subscale_type in questionnaire_config["subscales"]:
        subscale_data = subscales.get(subscale_type, {})
        if not isinstance(subscale_data, dict):
            subscale_data = {}
        
        subscale_n = subscale_data.get("n", "")
        if subscale_n is None:
            subscale_n = ""
        
        subscale_sum = subscale_data.get("sum", "")
        if subscale_sum is None:
            subscale_sum = ""
        
        # Add subscale_n row
        rows.append({
            "sessionId": session_id,
            "groupId": group_id,
            "prolificId": prolific_id,
            "item": "subscale_n",
            "prompt": "",
            "type": subscale_type,
            "options": "",
            "value": str(subscale_n) if subscale_n is not None else "",
            "changes": "",
            "distinct_values": "",
            "duration_ms": "",
            "first_interaction_ms": "",
            "hover_ms": "",
            "indecision_per_min": "",
            "shown_at_ms": "",
            "time_on_final_ms": "",
            "toggles": "",
            "moduleName": module_name,
            "moduleId": module_id,
            "uid": uid
        })
        
        # Add subscale_sum row
        rows.append({
            "sessionId": session_id,
            "groupId": group_id,
            "prolificId": prolific_id,
            "item": "subscale_sum",
            "prompt": "",
            "type": subscale_type,
            "options": "",
            "value": str(subscale_sum) if subscale_sum is not None else "",
            "changes": "",
            "distinct_values": "",
            "duration_ms": "",
            "first_interaction_ms": "",
            "hover_ms": "",
            "indecision_per_min": "",
            "shown_at_ms": "",
            "time_on_final_ms": "",
            "toggles": "",
            "moduleName": module_name,
            "moduleId": module_id,
            "uid": uid
        })
    
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]):
    """Write rows to CSV with specified columns only."""
    # Define the exact column order
    header = [
        "sessionId", "groupId", "prolificId",
        "item", "prompt", "type", "options", "value",
        "changes", "distinct_values", "duration_ms", "first_interaction_ms",
        "hover_ms", "indecision_per_min", "shown_at_ms", "time_on_final_ms", "toggles",
        "moduleName", "moduleId", "uid"
    ]
    
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
        return
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            # Convert None to empty string for CSV (R will read as NA)
            clean_row = {k: (row.get(k, "") if row.get(k) is not None else "") for k in header}
            w.writerow(clean_row)


def aggregate_csv_files(
        out_dir: str, 
        normalized_spell: str, 
        prolific_ids: List[str],
        questionnaire: str
) -> str:
    """
    Aggregate all individual participant CSV files into one combined CSV file.
    
    Args:
        out_dir: Output directory containing the CSV files
        normalized_spell: Normalized spell name (e.g., 's1_groupA')
        prolific_ids: List of prolific IDs to aggregate
        questionnaire: Name of the questionnaire
    
    Returns:
        Path to the aggregated CSV file
    """
    # Define the exact column order
    header = [
        "sessionId", "groupId", "prolificId",
        "item", "prompt", "type", "options", "value",
        "changes", "distinct_values", "duration_ms", "first_interaction_ms",
        "hover_ms", "indecision_per_min", "shown_at_ms", "time_on_final_ms", "toggles",
        "moduleName", "moduleId", "uid"
    ]
    
    # Output filename: {spell}.csv (e.g., s1_groupA.csv)
    aggregated_filename = f"q_{questionnaire}_{normalized_spell}.csv"
    aggregated_path = os.path.join(out_dir, aggregated_filename)
    
    all_rows = []
    
    # Read all individual participant CSV files
    for prolific_id in prolific_ids:
        individual_filename = f"q_{questionnaire}_{prolific_id}_{normalized_spell}.csv"
        individual_path = os.path.join(out_dir, individual_filename)
        
        if not os.path.exists(individual_path):
            print(f"[Export] Warning: Individual file not found: {individual_filename}")
            continue
        
        with open(individual_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
    
    # Write aggregated CSV
    with open(aggregated_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in all_rows:
            # Convert None to empty string for CSV (R will read as NA)
            clean_row = {k: (row.get(k, "") if row.get(k) is not None else "") for k in header}
            w.writerow(clean_row)
    
    return aggregated_path


def normalize_spell(spell: str) -> str:
    """
    Normalize spell parameter to always have 's' prefix.
    Examples:
      '0_groupZ' -> 's0_groupZ'
      's0_groupZ' -> 's0_groupZ'
      's1_groupA' -> 's1_groupA'
    """
    if spell.startswith("s"):
        return spell
    # Add 's' prefix if it doesn't start with 's'
    return f"s{spell}"

def get_questionnaire_config(questionnaire: str) -> Dict[str, Any]:
    if questionnaire == "dass21":
        return {
            "name": "dass21",
            "type_field": "sad",
            "subscales": ["A", "D", "S"]
        }
    elif questionnaire == "ocir":
        return {
            "name": "ocir",
            "type_field": "sub",
            "subscales": ["C", "H", "N", "O", "OR", "W"]
        }
    elif questionnaire == "spq":
        return {
            "name": "spq",
            "type_field": "sub",
            "subscales": None
        }
    elif questionnaire == "demography":
        return {
            "name": "demography",
            "type_field": "type",
            "subscales": None
        }
    else:
        raise ValueError(f"Unsupported questionnaire: {questionnaire}")

def find_firebase_credentials(directory: str = ".") -> Optional[str]:
    """
    Search for Firebase credential JSON files in the specified directory.
    
    Looks for files matching common Firebase credential patterns:
    - Files ending with '-firebase-adminsdk-*.json'
    - Files containing 'firebase' in the name and ending with '.json'
    
    Returns the path to the first matching credential file found, or None if none found.
    """
    # Pattern 1: Firebase admin SDK files (most common pattern)
    # Format: *-firebase-adminsdk-*.json
    pattern1 = os.path.join(directory, "*-firebase-adminsdk-*.json")
    matches = glob.glob(pattern1)
    if matches:
        return matches[0]
    
    # Pattern 2: Any JSON file with 'firebase' in the name
    pattern2 = os.path.join(directory, "*firebase*.json")
    matches = glob.glob(pattern2)
    if matches:
        return matches[0]
    
    return None


def parse_path(path: str) -> Dict[str, Optional[str]]:
    """
    Extract components from Firestore path:
    participants/{prolificId}/spells/{sessionDocId}/modules/{moduleDocId}
    """
    parts = path.split("/")
    out = {
        "prolificId": None,
        "sessionDocId": None,
        "moduleDocId": None
    }
    
    for i, p in enumerate(parts[:-1]):
        if p == "participants" and i + 1 < len(parts):
            out["prolificId"] = parts[i + 1]
        elif p == "spells" and i + 1 < len(parts):
            out["sessionDocId"] = parts[i + 1]
        elif p == "modules" and i + 1 < len(parts):
            out["moduleDocId"] = parts[i + 1]
    
    return out


def load_completed_participants(session_doc_id: str) -> Set[str]:
    """
    Load prolificIds of participants who have been approved
    from the inclusion_list CSV file.
    
    Returns a set of prolificIds who have been approved.
    
    Subject List File Location:
    Master subject inclusion list is expected at:
        DEMOGRAPHICS_DIR/inclusion_list_{group_id}.csv
    """
    
    # Load master subject inclusion list
    group_id = session_doc_id.split("_", 1)[-1]
    csv_filename = f"inclusion_list_{group_id}.csv"
    csv_path = os.path.join(DEMOGRAPHICS_DIR, csv_filename)
    
    if not os.path.exists(csv_path):
        print(f"[Export] Warning: Subject inclusion list file not found: {csv_path}")
        print(f"[Export] Proceeding without filtering by completion status")
        return set()
    
    completed_participants: Set[str] = set()
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prolific_id = row.get("participant_id", "").strip()
            
            # If in the master inclusion list, add to set
            if prolific_id:
                completed_participants.add(prolific_id)
    
    print(f"[Export] Loaded {len(completed_participants)} participant(s) with all modules complete and status approved from {csv_filename}")
    return completed_participants


def export_questionnaire_modules(
        db: firestore.Client,
        session_doc_id: str,
        questionnaire: str,
        prolific_id_filter: Optional[str] = None,
        allowed_prolific_ids: Optional[Set[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Export questionnaire module documents from Firestore, one per participant.
    Path: participants/{prolificId}/spells/{sessionDocId}/modules/{moduleNumber}_questionnaire
    
    Uses modules collection group to discover module documents.
    Returns a dictionary mapping prolificId -> module document (one per participant).
    """
    questionnaire_by_subject: Dict[str, Dict[str, Any]] = {}
    seen_paths = set()  # Track unique module paths to avoid duplicates
    subjects_found = set()
    
    print(f"[Export] Scanning questionnaire module documents for spell='{session_doc_id}'...")
    
    # Use modules collection group to find questionnaire modules
    # Modules are at: participants/{prolificId}/spells/{sessionDocId}/modules/{moduleNumber}_questionnaire
    for module_doc in db.collection_group("modules").stream():
        path_info = parse_path(module_doc.reference.path)
        
        prolific_id = path_info["prolificId"]
        session_doc_id_from_path = path_info["sessionDocId"]
        module_doc_id = path_info["moduleDocId"]
        
        # Filter for specified session
        if session_doc_id_from_path != session_doc_id:
            continue
        
        # Filter for modules ending with questionnaire
        if not module_doc_id or not module_doc_id.endswith(f"_{questionnaire}"):
            continue
        
        # Filter by prolificId if specified
        if prolific_id_filter and prolific_id != prolific_id_filter:
            continue
        
        # Filter by allowed prolificIds (from subject_list CSV)
        if allowed_prolific_ids is not None and prolific_id not in allowed_prolific_ids:
            continue
        
        # Build module path
        module_path = f"participants/{prolific_id}/spells/{session_doc_id_from_path}/modules/{module_doc_id}"
        
        # Skip if we've already processed this module (each participant should have only one)
        if module_path in seen_paths or prolific_id in questionnaire_by_subject:
            continue
        seen_paths.add(module_path)
        
        # Get module data
        module_data = module_doc.to_dict() or {}
        
        # Extract sessionId and groupId from sessionDocId
        # sessionDocId format: "s0_groupZ" -> sessionId: "0", groupId: "Z"
        session_id = None
        group_id = None
        if session_doc_id_from_path:
            match = re.match(r"s(\d+)_group(.+)", session_doc_id_from_path)
            if match:
                session_id = match.group(1)
                group_id = match.group(2)
        
        # Build record with path information and module data
        record = {
            "sessionDocId": session_doc_id_from_path,
            "sessionId": session_id or "",
            "groupId": group_id or "",
            "prolificId": prolific_id,
            "moduleDocId": module_doc_id,
            **module_data  # Include all other fields from module document
        }
        
        questionnaire_by_subject[prolific_id] = record
        subjects_found.add(prolific_id)
    
    print(f"[Export] Found {len(subjects_found)} subject(s) with {questionnaire} module documents")
    return questionnaire_by_subject

def main():
    ap = argparse.ArgumentParser(
        description="Download dass21 module documents from Firestore for comp-psych-longitudinal-2025 project"
    )
    ap.add_argument(
        "--out",
        default=QUESTIONNAIRE_DIR,
        help="Output directory for CSV files (default: QUESTIONNAIRE_DIR)"
    )
    ap.add_argument(
        "--spell",
        required=True,
        help="Session document ID (spell) (e.g., 's1_groupA')"
    )
    ap.add_argument(
        "--questionnaire",
        required=False,
        default=None,
        help="Questionnaire ID (questionnaire) (e.g., 'dass21')"
    )
    ap.add_argument(
        "--prolific-id",
        default=None,
        help="Filter by specific prolificId (optional, for testing with one subject)"
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Download all files in the session regardless of subject_list (bypasses filtering by completion status)"
    )
    ap.add_argument(
        "--credentials",
        default=FB_CREDENTIALS_FILE,
        help="Path to service account JSON key file. If not specified, uses GOOGLE_APPLICATION_CREDENTIALS env var, or looks for default filename in current directory."
    )
    ap.add_argument(
        "--output-mode",
        choices=["individual", "aggregated", "both"],
        default="aggregated",
        help="Output mode: 'individual' (only individual participant files), 'aggregated' (only summary file), or 'both' (default: aggregated)"
    )
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    # Initialize Firestore client with credentials
    credentials_path = args.credentials
    if not credentials_path:
        # Priority 1: Check GOOGLE_APPLICATION_CREDENTIALS environment variable (recommended for shared usage)
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            print(f"[Export] Using credentials from GOOGLE_APPLICATION_CREDENTIALS: {credentials_path}")
        else:
            # Priority 2: Search for Firebase credential files in current directory
            found_credentials = find_firebase_credentials(".")
            if found_credentials:
                credentials_path = found_credentials
                print(f"[Export] Found and using credentials file: {credentials_path}")
            else:
                print("[Export] Error: No credentials file found.")
                print("[Export] Please use one of the following options:")
                print("[Export]   1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable:")
                print("[Export]      export GOOGLE_APPLICATION_CREDENTIALS=\"/path/to/your/credentials.json\"")
                print("[Export]   2. Use --credentials flag:")
                print("[Export]      python fb_export_dass21.py --spell s1_groupA --credentials /path/to/your/credentials.json")
                print("[Export]   3. Place a Firebase credentials JSON file in the current directory")
                print("[Export]      (files matching *-firebase-adminsdk-*.json or *firebase*.json will be auto-detected)")
                return
    
    if not os.path.exists(credentials_path):
        print(f"[Export] Error: Credentials file not found: {credentials_path}")
        print("[Export] Please verify the path is correct and the file exists.")
        return
    
    # Load credentials and initialize client (suppress warnings during initialization)
    with suppress_stderr():
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        db = firestore.Client(credentials=credentials, project=credentials.project_id)
    
    # Normalize spell only for filenames (add 's' prefix if missing)
    # Use original spell for querying Firestore
    normalized_spell = normalize_spell(args.spell)

    if not args.questionnaire:
        questionnaires = ["dass21", "ocir", "spq", "demography"]
    else:
        questionnaires = [args.questionnaire]

    for questionnaire in questionnaires:
        # Get questionnaire configuration
        questionnaire_config = get_questionnaire_config(questionnaire)
        
        print(f"[Export] Starting export...")
        print(f"[Export] Project: {db.project}")
        print(f"[Export] Spell (query): {args.spell}")
        print(f"[Export] Spell (filename): {normalized_spell}")
        print(f"[Export] Questionnaire: {questionnaire_config['name']}")
        
        # Load completed participants from subject_list CSV (unless --all is specified)
        if args.all:
            print(f"[Export] --all flag specified: downloading all files regardless of subject_list")
            completed_participants = None
        else:
            completed_participants = load_completed_participants(args.spell)
        
        if args.prolific_id:
            print(f"[Export] Filtering by prolificId: {args.prolific_id}")
            # If specific prolific_id is provided and not using --all, check if it's in completed list
            if not args.all and completed_participants and args.prolific_id not in completed_participants:
                print(f"[Export] Warning: {args.prolific_id} is not in the completed participants list")
        else:
            if not args.all and completed_participants:
                print(f"[Export] Filtering to {len(completed_participants)} participant(s) with all modules complete and status approved")
            elif args.all:
                print(f"[Export] Downloading all participants in session (subject_list filtering disabled)")
        
        # Export questionnaire module documents
        # If --all is specified, allow all participants (allowed_ids = None)
        # If specific prolific_id is provided, only use filter if it's in completed list
        # Otherwise, use completed_participants set to filter all
        allowed_ids = None
        if args.all:
            # --all flag: download all participants regardless of subject_list
            allowed_ids = None
        elif args.prolific_id:
            # If specific ID provided, only filter if it's not in completed list (will be filtered out)
            if completed_participants and args.prolific_id not in completed_participants:
                allowed_ids = set()  # Empty set means no participants will match
        else:
            allowed_ids = completed_participants
        
        questionnaire_by_subject = export_questionnaire_modules(
            db,
            args.spell,
            prolific_id_filter=args.prolific_id,
            allowed_prolific_ids=allowed_ids,
            questionnaire=questionnaire,
        )
        
        if not questionnaire_by_subject:
            print("[Export] No questionnaire module documents found matching criteria")
            return
        
        # Check if any output files already exist before writing (based on output mode)
        existing_files = []
        
        if args.output_mode in ["individual", "both"]:
            # Check individual files
            for prolific_id in questionnaire_by_subject.keys():
                filename = f"q_{questionnaire}_{prolific_id}_{normalized_spell}.csv"
                filepath = os.path.join(args.out, filename)
                if os.path.exists(filepath):
                    existing_files.append(filename)
        
        if args.output_mode in ["aggregated", "both"]:
            # Check aggregated file
            aggregated_filename = f"q_{questionnaire}_{normalized_spell}.csv"
            aggregated_filepath = os.path.join(args.out, aggregated_filename)
            if os.path.exists(aggregated_filepath):
                existing_files.append(aggregated_filename)
        
        if existing_files:
            print("[Export] Error: The following file(s) already exist and would be overwritten:")
            for filename in existing_files:
                print(f"[Export]   - {filename}")
            print("[Export] Please remove or rename existing files before running the export.")
            return
        
        # Transform data for all subjects (needed for both individual and aggregated modes)
        all_subject_data = {}
        prolific_ids_list = []
        for prolific_id, module_data in questionnaire_by_subject.items():
            rows = transform_to_long_format(module_data, questionnaire_config)
            all_subject_data[prolific_id] = rows
            prolific_ids_list.append(prolific_id)
        
        # Write individual CSV files if requested
        if args.output_mode in ["individual", "both"]:
            for prolific_id in prolific_ids_list:
                filename = f"q_{questionnaire}_{prolific_id}_{normalized_spell}.csv"
                filepath = os.path.join(args.out, filename)
                rows = all_subject_data[prolific_id]
                
                print(f"[Export] Writing {len(rows)} row(s) to {filename}...")
                write_csv(filepath, rows)
                print(f"[Export] ✓ Saved {filename}")
        
        # Create aggregated CSV if requested
        if args.output_mode in ["aggregated", "both"]:
            if args.output_mode == "aggregated":
                # For aggregated-only mode, write temporary individual files first, then aggregate
                temp_files = []
                for prolific_id in prolific_ids_list:
                    filename = f"q_{questionnaire}_{prolific_id}_{normalized_spell}.csv"
                    filepath = os.path.join(args.out, filename)
                    rows = all_subject_data[prolific_id]
                    write_csv(filepath, rows)
                    temp_files.append(prolific_id)
                
                print(f"[Export] Aggregating {len(temp_files)} subject file(s) into combined CSV...")
                aggregated_path = aggregate_csv_files(args.out, normalized_spell, temp_files, questionnaire=questionnaire)
                
                # Remove temporary individual files
                for prolific_id in temp_files:
                    filename = f"q_{questionnaire}_{prolific_id}_{normalized_spell}.csv"
                    filepath = os.path.join(args.out, filename)
                    os.remove(filepath)
            else:
                # For "both" mode, aggregate from existing individual files
                print(f"[Export] Aggregating {len(prolific_ids_list)} subject file(s) into combined CSV...")
                aggregated_path = aggregate_csv_files(args.out, normalized_spell, prolific_ids_list, questionnaire=questionnaire)
            
            # Count total rows in aggregated file
            with open(aggregated_path, "r", encoding="utf-8") as f:
                row_count = sum(1 for _ in csv.DictReader(f))
            print(f"[Export] ✓ Saved aggregated file: {os.path.basename(aggregated_path)} ({row_count} row(s))")
        
        # Print summary
        if args.output_mode == "individual":
            print(f"[Export] Done! Exported {len(questionnaire_by_subject)} individual subject file(s) to {args.out}")
        elif args.output_mode == "aggregated":
            print(f"[Export] Done! Exported 1 aggregated file to {args.out}")
        else:  # both
            print(f"[Export] Done! Exported {len(questionnaire_by_subject)} subject file(s) and 1 aggregated file to {args.out}")

if __name__ == "__main__":
    main()

