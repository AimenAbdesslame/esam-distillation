"""Paths, data parameters and training parameters of the distillation study.

Every value can be overridden with an environment variable, so the experiment
scripts never need command line parsing.  The experiments themselves live in
`experiments/<name>/settings.json`.
"""
import os
from pathlib import Path

# ============================================================
# 1. LOCATIONS
# ============================================================
WORK = Path(os.environ.get("ESAM_WORK", "/kaggle/working"))
ESAM_ROOT = Path(os.environ.get("ESAM_ROOT", WORK / "ESAM"))
DATA_ROOT = Path(os.environ.get("ESAM_DATA", WORK / "data/scenenn_stream/322"))
RAW_ROOT = Path(os.environ.get("ESAM_RAW", WORK / "tmp/scenenn"))
FRAMES = Path(os.environ.get("ESAM_FRAMES", WORK / "distill/frames"))
TEACHER = Path(os.environ.get("ESAM_TEACHER", WORK / "distill/teacher"))
STUDENT = Path(os.environ.get("ESAM_STUDENT", WORK / "distill/student"))

# ============================================================
# 2. MODEL FILES
# ============================================================
ESAM_CKPT = Path(os.environ.get(
    "ESAM_CKPT", ESAM_ROOT / "work_dirs/ckpt/ESAM-E_CA_online_epoch_128.pth"))
TEACHER_CFG = Path(os.environ.get(
    "ESAM_TEACHER_CFG", ESAM_ROOT / "configs/ESAM-E_CA/ESAM-E_online_stream.py"))
STUDENT_CFG = Path(os.environ.get(
    "ESAM_STUDENT_CFG", WORK / "tools/esampp_tiny_scenenn.py"))

# ============================================================
# 3. DATA PARAMETERS
# ============================================================
NUM_FRAMES = int(os.environ.get("ESAM_NUM_FRAMES", 80))
VAL_START = int(os.environ.get("ESAM_VAL_START", 40))
CLUSTERS = int(os.environ.get("ESAM_CLUSTERS", 100))
SCORE_THR = float(os.environ.get("ESAM_SCORE_THR", 0.3))
MIN_POINTS = int(os.environ.get("ESAM_MIN_POINTS", 100))

# ============================================================
# 4. TRAINING PARAMETERS
# ============================================================
MAX_MINUTES = float(os.environ.get("ESAM_MAX_MINUTES", 110))
LR = float(os.environ.get("ESAM_LR", 1e-3))
DEVICE = "cuda"

# ============================================================
# 5. EXPERIMENT SWITCHES
# ============================================================
FREEZE_SHARED = os.environ.get("ESAM_FREEZE_SHARED", "0") == "1"   # train only the TinyExpert
TARGET = os.environ.get("ESAM_TARGET", "teacher")                  # "teacher" or "gt"
