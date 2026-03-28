import os

# Model size and family selection
LARGE = True   # False => 4B, True => 7-8B
MISTRAL = True # False => Qwen, True => Mistral

# idx mapping: (large: mistral->0, qwen->1) ; (small: mistral->2, qwen->3)
idx = ((3, 2), (1, 0))[LARGE][MISTRAL]

# Base paths (adjust BASE_DIR to the parent directory where this repo lives)
BASE_DIR = "/home/ubuntu"
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "hf_checkpoints")

OUT_DIR = os.path.join(BASE_DIR, "lift-out")           # must exist before running
DATA_DIR = os.path.join(BASE_DIR, "lift", "data-files") # e.g. /path/to/lift/data-files

GPUS = 1
PER_GPU = 2

# per-model subdirectories
_model_tag = "mistral" if MISTRAL else "qwen"
_size_tag = "" if LARGE else "-small"
data_subdir = os.path.join(DATA_DIR, _model_tag + _size_tag)
out_subdir = os.path.join(OUT_DIR, _model_tag + _size_tag)

# LIFT configuration
GEN_EPOCHS = 1
TARGET_EXAMPLES = 1000
PRELIM_TEMP = [0.1, 0.5, 0.0, 0.5][idx]
VARIOUS_TEMP = [0.2, 1.6, 0.0, 1.6][idx]

LAMBDA = 1.0 # Used for ablations
# Weights and thresholds per dataset (order for those with 3 values: gsm8k, strategyqa/commonsenseqa, svamp/asdiv)
ANSWER_WEIGHT = [x * LAMBDA for x in [
    [1.8, 0.4, 0.8],
    [1.0, 2.5, 3.0],
    [0.0, 0.0, 0.0],
    [1.0, 2.5, 3.0],
][idx]]
SCORE_ANSWER_THRESHOLD = [0.4, 0.7, 0.0, 0.7][idx]
SAMPLE_WEIGHT = [
    [1.0, 2.0, 1.0],
    [1.5, 1.0, 3.0],
    [0.0, 0.0, 0.0],
    [1.5, 1.0, 3.0],
][idx]
SCORE_THRESHOLD = [
    (0.5, 12),
    (0.5, 12),
    (0, 0),
    (0.5, 12),
][idx]

BATCH_SIZE = [4, 2, 0, 2][idx]
EPOCHS = [1, 1, 0, 1][idx]
LR = [7e-5, 3e-4, 0, 3e-4][idx]
GRAD_ACCUM_STEPS = 2
MAX_LENGTH = 1024
KL_LAMBDA = [0.8, 2.0, 0.0, 2.0][idx]
