ece 880/780 Course project
==================================================

This directory provides a self-contained toolkit to: (1) train MedNeXt models (single-fold, cross-validation, and ablation variants), and (2) run inference (single model, multi-fold ensemble, and GliGAN-augmented ensemble).

This document explains where to place data and weights, how to run each workflow, expected outputs, and operational recommendations for reproducible experiments.

Contents
--------
- Training utilities: `run_training.py` and single-purpose wrappers (`train_ce.py`, `train_dscpp.py`, `train_dscpp_ce.py`, `train_dscpp_ce_ace.py`).
- Inference utilities: `run_inference.py` and wrappers (`inf_ensemble.py`, `inf_no_ensemble.py`, `inf_gligan_ensemble.py`).
- Config: `config.py` (adjust paths and defaults here).

Prerequisites
-------------
- Python 3.8+ (use the project's virtual environment). Activate with:

```powershell
& .\.venv\Scripts\Activate.ps1
```

- Install required packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install the standalone package requirements for this folder:

```bash
pip install -r standalone_inference/requirements.txt
```

- GPU recommended for training and inference (CUDA + PyTorch installed). If CUDA is unavailable the scripts will fall back to CPU.

Data placement
--------------
Place your BraTS-style dataset (images and ground-truth label maps) under the `data/` folder. The canonical default used by the scripts is:

- `data/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/`

If your dataset is in a different location, update `standalone_inference/config.py` `DATA_DIR` or pass `--data-dir /path/to/data` to the CLI.

Expected layout (example):

```
data/
└── MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/
    ├── patient_000/
    │   ├── t1.nii.gz
    │   ├── t1ce.nii.gz
    │   ├── t2.nii.gz
    │   ├── flair.nii.gz
    │   └── label.nii.gz
    └── ...
```

GLIGAN weights placement
------------------------
If you intend to use the GliGAN-augmented ensemble, place the GliGAN model checkpoints and related assets under one of these paths (both are supported; `config.py` controls the default):

- Preferred (project-wide): `weights/gligan/`
- Standalone (inside this package): `standalone_inference/weights/gligan/`

What to place inside the `gligan` folder:
- The trained GliGAN generator/checkpoints required by your augmentation code (the exact filenames depend on how the GliGAN component was exported). Example:

```
weights/gligan/
├── generator.pth
├── config.json
└── checkpoints/
    └── gan_checkpoint_000.pth
```

If you downloaded weights with `scripts/download_weights.py`, the script places them under `weights/` by default; you can move or symlink them to `weights/gligan/` as needed.

Check `standalone_inference/config.py` and set `GLIGAN_ROOT` to the exact folder containing the GliGAN assets.

Checkpoint layout for MedNeXt models
-----------------------------------
Training outputs are expected as follows (this is the convention used by the ensemble utilities):

- Single-fold checkpoints: `checkpoints-mednext-diff-ssa/best_model.pth` (or similar)
- Cross-validation layout (ensemble input):

```
checkpoints-cv-<variant>/
├── fold_0/
│   └── best_model.pth
├── fold_1/
│   └── best_model.pth
└── fold_2/
    └── best_model.pth
```

The ensemble CLI discovers `fold_*/best_model.pth` automatically when given `--ckpt-dir`.

MedNeXt location
----------------
`standalone_inference` requires the MedNeXt model sources to be available in one of these locations:

- `src/models/mednext` (recommended)
- `standalone_inference/mednext`

Clone the MedNeXt repository into your workspace root (or into `standalone_inference/mednext`) and ensure `create_mednext_v1` is exposed. Example:

```bash
# clone into workspace root (creates MedNeXt/ or src/models/mednext)
git clone <MEDNEXT_REPO_URL>

# or clone directly into the standalone folder
git clone <MEDNEXT_REPO_URL> standalone_inference/mednext
```

Replace `<MEDNEXT_REPO_URL>` with the repository URL you use for MedNeXt. After cloning, verify that one of the expected import paths exists; the inference utilities will import from `src/models/mednext` or `standalone_inference/mednext` automatically.

How to run — Training
----------------------
All training scripts are located in `standalone_inference/`. Use the wrappers for single-responsibility runs.

1) CE-only training

```powershell
.venv\Scripts\python.exe standalone_inference\train_ce.py --data-dir data --out-dir checkpoints-ce --epochs 50 --batch-size 2
```

2) DSC++-only training

```powershell
.venv\Scripts\python.exe standalone_inference\train_dscpp.py --data-dir data --out-dir checkpoints-dscpp --epochs 50
```

3) DSC++ + CE training

```powershell
.venv\Scripts\python.exe standalone_inference\train_dscpp_ce.py --data-dir data --out-dir checkpoints-dscpp-ce --epochs 50
```

4) DSC++ + CE + ACE training (ACE placeholder: replace with project ACE loss if available)

```powershell
.venv\Scripts\python.exe standalone_inference\train_dscpp_ce_ace.py --data-dir data --out-dir checkpoints-dscpp-ce-ace --epochs 50
```

5) Cross-validation wrapper (example: folds 0..6)

```powershell
.venv\Scripts\python.exe standalone_inference\run_training.py crossval --data-dir data --out-dir checkpoints-cv-dscpp_ce --start-fold 0 --end-fold 6 --epochs 40
```

Notes
- The `Trainer` class in `run_training.py` implements a minimal training loop. Replace the internal loss function with your project's exact loss functions if required.
- Use smaller `--batch-size` on limited GPU memory.

How to run — Inference
----------------------
All inference utilities are in `standalone_inference/run_inference.py`. Use the dedicated wrapper scripts for convenience.

1) Baseline (single checkpoint):

```powershell
.venv\Scripts\python.exe standalone_inference\run_inference.py baseline --ckpt ./checkpoints-mednext-diff-ssa/best_model.pth --data-dir data
```

2) Single-model (no ensemble) wrapper:

```powershell
.venv\Scripts\python.exe standalone_inference\inf_no_ensemble.py --ckpt ./checkpoints-mednext-diff-ssa/best_model.pth --data-dir data
```

3) Multi-fold ensemble (discover `fold_*/best_model.pth`):

```powershell
.venv\Scripts\python.exe standalone_inference\inf_ensemble.py --ckpt-dir ./checkpoints-cv-dscpp_ce --n-folds 7 --data-dir data
```

4) Explicit checkpoint list (ensemble):

```powershell
.venv\Scripts\python.exe standalone_inference\inf_ensemble.py --ckpts ./fold_0/best_model.pth ./fold_1/best_model.pth --data-dir data
```

5) GliGAN ensemble (original + GliGAN-augmented average):

```powershell
.venv\Scripts\python.exe standalone_inference\inf_gligan_ensemble.py --gligan-root weights/gligan --mednext-ckpt ./checkpoints-mednext-diff-ssa/best_model.pth --data-dir data
```

Configuration and defaults
--------------------------
Edit `standalone_inference/config.py` to change defaults for:

- `DATA_DIR`, `BATCH_SIZE`, `MODEL_SIZE`, `PATCH_SIZE`
- `GLIGAN_ROOT` (where GliGAN weights are located)
- Checkpoint templates such as `ABLATION_CKPT_TEMPLATE` and `ENSEMBLE_CKPT_DIR`

Operational tips
----------------
- Reproducibility: set `cfg.SEED` in `config.py` and avoid non-deterministic data augmentations when validating.
- GPU memory: reduce `--batch-size` or use `MODEL_SIZE='S'` for smaller networks.
- Checkpointing: training wrappers save per-epoch files named `epoch_XXX.pth`. For ensembles, copy the desired `best_model.pth` into `fold_*` subfolders.
- Logging: add TensorBoard or CSV logging to `Trainer` if you need experiment tracking.

Outputs
-------
- Console summary: per-region DSC / HD95 and mean DSC.
- Checkpoints: saved under the `--out-dir` you provide (per-epoch).

Common issues & resolutions
---------------------------
- "Checkpoint not found": verify `--ckpt` or `--ckpt-dir` and that `best_model.pth` exists.
- "GliGAN module not available": ensure `GLIGAN_ROOT` is set and that the augmentation module dependencies are installed. The code will fall back to standard inference if GliGAN cannot be initialized.
- OOM errors: lower `--batch-size` or switch to a smaller `MODEL_SIZE`.

Contact / Next steps
--------------------
If you want, I can:
- Wire the exact project loss functions (true DSC++, ACE) into `Trainer`.
- Add AMP, schedulers, and resume-from-checkpoint in `Trainer`.
- Add result export (CSV) and automatic comparison scripts.

Choose one next task and I will implement it.
