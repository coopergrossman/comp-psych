"""
created 2026.2.2

global constants and configurations

@author: cgrossman

"""

from pathlib import Path

# Root of compPsych repo
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# User-specific data locations
DATA_ROOT = Path("C:/Users/cooper/Desktop/oLab/compPsych")
TASK_ROOT = Path("C:/Users/cooper/Documents/githubRepositories/Comp_Psych_Longitudinal_tasks")
DEMOGRAPHICS_DIR = DATA_ROOT / "demo_data"
QUESTIONNAIRE_DIR = DATA_ROOT / "questionnaire_data"

# Credentials
FB_CREDENTIALS_FILE = Path(
    "C:/Users/cooper/Documents/githubRepositories/oLabAnalysis/compPsych/config/fb_credentials.json"
)

NUM_SESSIONS = 12