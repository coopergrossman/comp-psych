"""
created 2026.2.2

global constants and configurations

@author: cgrossman

"""

from pathlib import Path

# Root of comp_psych repo
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# User-specific data locations
DATA_ROOT = Path("C:/Users/cooper/Documents/githubRepositories/comp-psych/data")
TASK_ROOT = Path("C:/Users/cooper/Documents/githubRepositories/Comp_Psych_Longitudinal_tasks")
DEMOGRAPHICS_DIR = DATA_ROOT / "demo_data"
QUESTIONNAIRE_DIR = DATA_ROOT / "questionnaire_data"

# Credentials
FB_CREDENTIALS_FILE = Path(
    "C:/Users/cooper/Documents/githubRepositories/comp-psych/comp_psych/core/fb_credentials.json"
)

NUM_SESSIONS = 12