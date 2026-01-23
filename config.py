import os

# Model size and family selection
LARGE = False   # False => 7-8B, True => 24-32B (LIFT was run only on smaller models)
MISTRAL = False # False => Qwen, True => Mistral

# idx mapping: (not LARGE: mistral->0, qwen->1) ; (LARGE: mistral->2, qwen->3)
idx = ((1, 0), (3, 2))[LARGE][MISTRAL]

# Base paths (adjust BASE_DIR to the parent directory where this repo lives)
BASE_DIR = "/root"
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "hf_checkpoints")

OUT_DIR = os.path.join(BASE_DIR, "lift-out")           # must exist before running
DATA_DIR = os.path.join(BASE_DIR, "lift", "data-files") # e.g. /path/to/lift/data-files

# per-model subdirectories
_model_tag = "mistral" if MISTRAL else "qwen"
_size_tag = "-large" if LARGE else ""
data_subdir = os.path.join(DATA_DIR, _model_tag + _size_tag)
out_subdir = os.path.join(OUT_DIR, _model_tag + _size_tag)

# LIFT configuration (values preserved)
GEN_EPOCHS = 1
TARGET_EXAMPLES = 1000
PRELIM_TEMP = [0.1, 0.5, 0.0, 0.0][idx]
VARIOUS_TEMP = [0.2, 1.6, 0.0, 0.0][idx]

# Weights and thresholds per dataset (order for those with 3 values: gsm8k, strategyqa/commonsenseqa, svamp/asdiv)
ANSWER_WEIGHT = [
    [1.8, 0.4, 0.8],
    [1.0, 2.5, 3.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
][idx]
SCORE_ANSWER_THRESHOLD = [0.4, 0.7, 0.0, 0.0][idx]
SAMPLE_WEIGHT = [
    [1.0, 1.0, 1.0],
    [1.5, 1.0, 3.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
][idx]
SCORE_THRESHOLD = [
    (0.5, 12),
    (0.5, 12),
    (0, 0),
    (0, 0),
][idx]

BATCH_SIZE = [4, 2, 0, 0][idx]
EPOCHS = [1, 1, 0, 0][idx]
LR = [7e-5, 4e-4, 0, 0][idx]
GRAD_ACCUM_STEPS = 2
MAX_LENGTH = 2048
KL_LAMBDA = [0.8, 2.0, 0.0, 0.0][idx]
