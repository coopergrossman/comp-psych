"""
created 25.12.2

task constants and configurations for gain_loss task

@author: cgrossman

"""

from compPsych.core.env import DATA_ROOT, TASK_ROOT
from pathlib import Path

TASK_NAME = "gain_loss"

DATA_DIR = DATA_ROOT / TASK_NAME / "data"
MODEL_DIR = (
    Path(__file__).resolve().parent / "analyses" / "modeling" / "stan_models"
)
MODEL_SAVE_DIR = DATA_ROOT / TASK_NAME / "stan_fits"
DESIGNS_DIR = TASK_ROOT / TASK_NAME / "task" / "static" / "designs" / "csv"