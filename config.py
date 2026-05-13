"""
Standalone inference configuration.

Paths are relative to the workspace root.
"""

# ── Data ──────────────────────────────────────────────────────────────────

DATA_DIR = "./data/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth"
PATCH_SIZE = (128, 160, 112)
BATCH_SIZE = 1
SEED = 42
TRAIN_RATIO = 0.8

# ── Model architecture ────────────────────────────────────────────────────

MODEL_SIZE = "B"           # S, B, M, or L
KERNEL_SIZE = 3
DEEP_SUPERVISION = True

# ── Inference settings ────────────────────────────────────────────────────

USE_AMP = True             # Automatic mixed precision (CUDA only)
POSTPROCESS = True         # Morphological postprocessing
NUM_FOLDS = 7              # Total folds for ensemble (if using fold-based ensemble)

# ── Ablation loss configurations ──────────────────────────────────────────

ABLATION_LOSSES = {
    "dscpp": {
        "label": "DSC++",
        "loss_type": "dscpp",
    },
    "dscpp_ce": {
        "label": "DSC++ + CE",
        "loss_type": "dscpp_ce",
    },
    "dscpp_ce_ace": {
        "label": "DSC++ + CE + ACE",
        "loss_type": "dscpp_ce_ace",
    },
}

# ── Checkpoint paths ──────────────────────────────────────────────────────
# Adjust these to point to your trained checkpoints

BASELINE_CKPT = "./checkpoints-ensemble-v2/best_model.pth"

ABLATION_CKPT_TEMPLATE = "./checkpoints-cv-{loss_type}/fold_0/best_model.pth"

ENSEMBLE_CKPT_DIR = "./checkpoints-ensemble-v2"  # Will discover fold_*/best_model.pth
ENSEMBLE_FOLD_CKPT_TEMPLATE = "./checkpoints-cv-{loss_type}/fold_{fold}/best_model.pth"

GLIGAN_ROOT = "./weights/gligan"
GLIGAN_MEDNEXT_CKPT = "./checkpoints-mednext-diff-ssa/best_model.pth"
