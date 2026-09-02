"""
created 2026.9.1

run all Firebase exports (questionnaires, explore_exploit, gain_loss) for a spell

@author: cgrossman
"""

import argparse
import subprocess
import sys
from pathlib import Path

from comp_psych.core.env import DEMOGRAPHICS_DIR
from comp_psych.explore_exploit.export.fb_export import fb_export_explore_exploit
from comp_psych.gain_loss.export.fb_export import fb_export_gain_loss

# Repo root (parent of the comp_psych package), needed to run the questionnaire
# export script as `python -m comp_psych...` regardless of the caller's cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]


def check_inclusion_list(spell: str) -> Path:
    """Verify that an inclusion list exists for the group parsed from `spell`.

    Mirrors the group-parsing convention used by the individual export
    scripts (e.g. ``load_completed_participants``): the group id is
    everything after the first underscore, e.g. 's1_groupA' -> 'groupA'.

    Parameters
    ----------
    spell : str
        Session document ID (e.g. 's1_groupA').

    Returns
    -------
    pathlib.Path
        Path to the matching inclusion list CSV.

    Raises
    ------
    FileNotFoundError
        If no inclusion list matching the parsed group is found.
    """
    group_id = spell.split("_", 1)[-1]
    inclusion_list_path = DEMOGRAPHICS_DIR / f"inclusion_list_{group_id}.csv"

    if not inclusion_list_path.exists():
        raise FileNotFoundError(
            f"No inclusion list found at {inclusion_list_path}. "
            f"An inclusion list must be provided before running exports."
        )

    return inclusion_list_path


def fb_export_all(spell: str, force_refresh: bool = False) -> None:
    """Run the questionnaire, explore_exploit, and gain_loss exports for a spell.

    Parameters
    ----------
    spell : str
        Session document ID (e.g. 's1_groupA').
    force_refresh : bool, default False
        Passed through to the explore_exploit and gain_loss exports; if
        True, re-downloads raw task data from Firestore even if local files
        already exist. The questionnaire export always re-queries Firestore
        regardless of this flag.

    Raises
    ------
    FileNotFoundError
        If no inclusion list exists for `spell`'s group (see
        `check_inclusion_list`).
    """
    inclusion_list_path = check_inclusion_list(spell)
    print(f"[Export All] Using inclusion list: {inclusion_list_path}")

    print(f"[Export All] Exporting questionnaires for spell={spell}...")
    subprocess.run(
        [
            sys.executable, "-m",
            "comp_psych.questionnaires.export.fb_export_questionnaires",
            "--spell", spell,
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    print("[Export All] Exporting explore_exploit task data...")
    fb_export_explore_exploit(force_refresh=force_refresh)

    print("[Export All] Exporting gain_loss task data...")
    fb_export_gain_loss(force_refresh=force_refresh)

    print("[Export All] Done!")


def main():
    ap = argparse.ArgumentParser(
        description="Run all Firebase exports (questionnaires, explore_exploit, gain_loss) for a spell."
    )
    ap.add_argument(
        "--spell",
        required=True,
        help="Session document ID (spell) (e.g., 's1_groupA')"
    )
    ap.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download of explore_exploit/gain_loss raw data even if local files already exist"
    )
    args = ap.parse_args()

    fb_export_all(args.spell, force_refresh=args.force_refresh)


if __name__ == "__main__":
    main()
