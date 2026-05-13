
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_STANDALONE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_STANDALONE_DIR)
sys.path.insert(0, _STANDALONE_DIR)
sys.path.append(_ROOT)
sys.path.append(os.path.join(_ROOT, "src"))

from data.dataset import build_eval_loader, labels_to_regions

from embedded_utils import unpack_mednext_outputs, compute_brats_metrics, flatten_metrics, mean_record, apply_postprocess_3ch

import config as cfg
from inference_utils import (
    load_mednext_model,
    predict_single_model,
    predict_ensemble,
    discover_fold_checkpoints,
    resolve_path,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_baseline_inference(data_dir, patch_size, batch_size, use_amp, postprocess, ckpt_path):
    print("\n" + "="*70)
    print("  BASELINE MEDNEXT INFERENCE")
    print("="*70)

    model, _ = load_mednext_model(ckpt_path, cfg.MODEL_SIZE, cfg.KERNEL_SIZE, device)
    val_loader = build_eval_loader(data_dir, patch_size, batch_size, cfg.SEED, cfg.TRAIN_RATIO)

    records = []
    for images, masks, patient_ids in tqdm(val_loader, desc="  Inferring"):
        images = images.to(device).float()
        targets = labels_to_regions(masks.to(device).float())

        logits = predict_single_model(model, images, use_amp, device)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        if postprocess:
            preds_np = preds.cpu().numpy()
            preds_np = np.stack([apply_postprocess_3ch(preds_np[i]) for i in range(preds_np.shape[0])])
            preds = torch.from_numpy(preds_np).to(device)

        for i in range(images.shape[0]):
            metrics = compute_brats_metrics(preds[i].cpu().numpy(), targets[i].cpu().numpy())
            flat = flatten_metrics(metrics)
            flat["patient_id"] = patient_ids[i] if i < len(patient_ids) else f"patient_{len(records)}"
            records.append(flat)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_results(records, "Baseline MedNeXt (no ensemble)")
    return records


@torch.no_grad()
def run_ablation_inference(data_dir, patch_size, batch_size, use_amp, postprocess, loss_type):
    print("\n" + "="*70)
    print(f"  ABLATION: {loss_type.upper()}")
    print("="*70)
    
    if loss_type not in cfg.ABLATION_LOSSES:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(cfg.ABLATION_LOSSES.keys())}")
    
    ablation_cfg = cfg.ABLATION_LOSSES[loss_type]
    ckpt_path = cfg.ABLATION_CKPT_TEMPLATE.format(loss_type=loss_type)
    
    print(f"  Loss: {ablation_cfg['label']}")
    print(f"  Loading checkpoint...")
    model, _ = load_mednext_model(ckpt_path, cfg.MODEL_SIZE, cfg.KERNEL_SIZE, device, 
                                   label=loss_type)
    
    print(f"  Loading validation data...")
    val_loader = build_eval_loader(data_dir, patch_size, batch_size, cfg.SEED, cfg.TRAIN_RATIO)
    
    records = []
    for images, masks, patient_ids in tqdm(val_loader, desc="  Inferring"):
        images = images.to(device).float()
        targets = labels_to_regions(masks.to(device).float())
        
        logits = predict_single_model(model, images, use_amp, device)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        if postprocess:
            preds_np = preds.cpu().numpy()
            preds_np = np.stack([apply_postprocess_3ch(preds_np[i]) for i in range(preds_np.shape[0])])
            preds = torch.from_numpy(preds_np).to(device)
        
        for i in range(images.shape[0]):
            metrics = compute_brats_metrics(preds[i].cpu().numpy(), targets[i].cpu().numpy())
            flat = flatten_metrics(metrics)
            flat["patient_id"] = patient_ids[i] if i < len(patient_ids) else f"patient_{len(records)}"
            records.append(flat)
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    pp_tag = "with postproc" if postprocess else "no postproc"
    print_results(records, f"Ablation {loss_type} ({pp_tag})")
    return records


@torch.no_grad()
def run_single_model_inference(data_dir, patch_size, batch_size, use_amp, postprocess, ckpt_path):
    """Run inference without ensemble (single model)."""
    print("\n" + "="*70)
    print("  SINGLE MODEL INFERENCE (NO ENSEMBLE)")
    print("="*70)
    
    print(f"  Loading checkpoint...")
    model, _ = load_mednext_model(ckpt_path, cfg.MODEL_SIZE, cfg.KERNEL_SIZE, device)
    
    print(f"  Loading validation data...")
    val_loader = build_eval_loader(data_dir, patch_size, batch_size, cfg.SEED, cfg.TRAIN_RATIO)
    
    records = []
    for images, masks, patient_ids in tqdm(val_loader, desc="  Inferring"):
        images = images.to(device).float()
        targets = labels_to_regions(masks.to(device).float())
        
        logits = predict_single_model(model, images, use_amp, device)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        if postprocess:
            preds_np = preds.cpu().numpy()
            preds_np = np.stack([apply_postprocess_3ch(preds_np[i]) for i in range(preds_np.shape[0])])
            preds = torch.from_numpy(preds_np).to(device)
        
        for i in range(images.shape[0]):
            metrics = compute_brats_metrics(preds[i].cpu().numpy(), targets[i].cpu().numpy())
            flat = flatten_metrics(metrics)
            flat["patient_id"] = patient_ids[i] if i < len(patient_ids) else f"patient_{len(records)}"
            records.append(flat)
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    pp_tag = "with postproc" if postprocess else "no postproc"
    print_results(records, f"Single Model ({pp_tag})")
    return records


@torch.no_grad()
def run_ensemble_inference(data_dir, patch_size, batch_size, use_amp, postprocess, 
                          ckpt_dir_or_list, num_folds):
    """Run ensemble inference (average multiple fold checkpoints)."""
    print("\n" + "="*70)
    print("  ENSEMBLE INFERENCE (MULTI-FOLD)")
    print("="*70)
    
    if isinstance(ckpt_dir_or_list, list):
        ckpt_paths = ckpt_dir_or_list
    else:
        ckpt_paths = discover_fold_checkpoints(ckpt_dir_or_list, num_folds)
    
    if not ckpt_paths:
        raise ValueError(f"No checkpoints found in {ckpt_dir_or_list}")
    
    print(f"  Found {len(ckpt_paths)} fold checkpoints")
    print(f"  Loading models...")
    
    models = []
    for i, ckpt_path in enumerate(ckpt_paths):
        model, _ = load_mednext_model(ckpt_path, cfg.MODEL_SIZE, cfg.KERNEL_SIZE, device,
                                      label=f"Fold {i}")
        models.append(model)
    
    print(f"  Loading validation data...")
    val_loader = build_eval_loader(data_dir, patch_size, batch_size, cfg.SEED, cfg.TRAIN_RATIO)
    
    records = []
    for images, masks, patient_ids in tqdm(val_loader, desc="  Inferring"):
        images = images.to(device).float()
        targets = labels_to_regions(masks.to(device).float())
        
        probs = predict_ensemble(models, images, use_amp, device)
        preds = (probs > 0.5).float()
        
        if postprocess:
            preds_np = preds.cpu().numpy()
            preds_np = np.stack([apply_postprocess_3ch(preds_np[i]) for i in range(preds_np.shape[0])])
            preds = torch.from_numpy(preds_np).to(device)
        
        for i in range(images.shape[0]):
            metrics = compute_brats_metrics(preds[i].cpu().numpy(), targets[i].cpu().numpy())
            flat = flatten_metrics(metrics)
            flat["patient_id"] = patient_ids[i] if i < len(patient_ids) else f"patient_{len(records)}"
            records.append(flat)
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    pp_tag = "with postproc" if postprocess else "no postproc"
    print_results(records, f"Ensemble ({len(models)}-fold, {pp_tag})")
    return records


@torch.no_grad()
def run_gligan_ensemble_inference(data_dir, patch_size, batch_size, use_amp, postprocess,
                                  gligan_root, mednext_ckpt):
    print("\n" + "="*70)
    print("  GLIGAN ENSEMBLE INFERENCE")
    print("="*70)
    
    try:
        from data.gligan_augment import GliGANSynthesisConfig, GliGANOnTheFlyAugmenter
    except ImportError as e:
        print(f"  [ERROR] GliGAN module not available: {e}")
        print(f"  Please ensure gligan_augment module is properly installed.")
        sys.exit(1)
    
    print(f"  Loading MedNeXt checkpoint...")
    model, _ = load_mednext_model(mednext_ckpt, cfg.MODEL_SIZE, cfg.KERNEL_SIZE, device)
    
    print(f"  Loading GliGAN augmenter...")
    abs_gligan_root = resolve_path(gligan_root)
    
    if not os.path.isdir(abs_gligan_root):
        print(f"  [WARNING] GliGAN root not found at {abs_gligan_root}")
        print(f"  Falling back to standard inference without augmentation.")
        return run_single_model_inference(data_dir, patch_size, batch_size, use_amp, 
                                         postprocess, mednext_ckpt)
    
    try:
        cfg_gligan = GliGANSynthesisConfig(
            checkpoint_root=abs_gligan_root,
            preset="brats2024",
            label_source="gan",
            probability=1.0,
            device=device.type,
            verbose=False,
        )
        augmenter = GliGANOnTheFlyAugmenter(cfg_gligan)
        print(f"  GliGAN loaded successfully")
    except Exception as e:
        print(f"  [WARNING] Failed to initialize GliGAN: {e}")
        print(f"  Falling back to standard inference.")
        return run_single_model_inference(data_dir, patch_size, batch_size, use_amp, 
                                         postprocess, mednext_ckpt)
    
    print(f"  Loading validation data...")
    val_loader = build_eval_loader(data_dir, patch_size, batch_size, cfg.SEED, cfg.TRAIN_RATIO)
    
    records = []
    for images, masks, patient_ids in tqdm(val_loader, desc="  Inferring"):
        images = images.to(device).float()
        targets = labels_to_regions(masks.to(device).float())
        
        # Prediction on original
        logits_orig = predict_single_model(model, images, use_amp, device)
        probs_orig = torch.sigmoid(logits_orig)
        
        # Prediction on GliGAN-augmented
        images_aug = augmenter(images, masks)
        logits_aug = predict_single_model(model, images_aug, use_amp, device)
        probs_aug = torch.sigmoid(logits_aug)
        
        # Ensemble (average)
        probs = (probs_orig + probs_aug) / 2.0
        preds = (probs > 0.5).float()
        
        if postprocess:
            preds_np = preds.cpu().numpy()
            preds_np = np.stack([apply_postprocess_3ch(preds_np[i]) for i in range(preds_np.shape[0])])
            preds = torch.from_numpy(preds_np).to(device)
        
        for i in range(images.shape[0]):
            metrics = compute_brats_metrics(preds[i].cpu().numpy(), targets[i].cpu().numpy())
            flat = flatten_metrics(metrics)
            flat["patient_id"] = patient_ids[i] if i < len(patient_ids) else f"patient_{len(records)}"
            records.append(flat)
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    pp_tag = "with postproc" if postprocess else "no postproc"
    print_results(records, f"GliGAN Ensemble (original + augmented, {pp_tag})")
    return records


def print_results(records, label):
    if not records:
        print("  No records to display.")
        return
    
    mean = mean_record(records)
    print(f"\n  {label}:")
    print(f"    ET  DSC: {mean['ET_dsc']:.4f}  HD95: {mean['ET_hd95']:.2f}")
    print(f"    TC  DSC: {mean['TC_dsc']:.4f}  HD95: {mean['TC_hd95']:.2f}")
    print(f"    WT  DSC: {mean['WT_dsc']:.4f}  HD95: {mean['WT_hd95']:.2f}")
    print(f"    Mean DSC: {(mean['ET_dsc'] + mean['TC_dsc'] + mean['WT_dsc']) / 3:.4f}")


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Standalone MedNeXt inference runner")
    
    subparsers = parser.add_subparsers(dest="mode", help="Inference mode")
    
    # Baseline mode
    baseline = subparsers.add_parser("baseline", help="Baseline MedNeXt inference")
    baseline.add_argument("--data-dir", default=cfg.DATA_DIR)
    baseline.add_argument("--ckpt", default=cfg.BASELINE_CKPT)
    baseline.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    baseline.add_argument("--no-postprocess", action="store_true")
    
    # Ablation mode
    ablation = subparsers.add_parser("ablation", help="Ablation study inference")
    ablation.add_argument("--loss", required=True, choices=list(cfg.ABLATION_LOSSES.keys()),
                         help="Loss type to evaluate")
    ablation.add_argument("--data-dir", default=cfg.DATA_DIR)
    ablation.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    ablation.add_argument("--no-postprocess", action="store_true")
    
    # Single model (no ensemble)
    single = subparsers.add_parser("no_ensemble", help="Single model inference (no ensemble)")
    single.add_argument("--ckpt", required=True, help="Path to checkpoint")
    single.add_argument("--data-dir", default=cfg.DATA_DIR)
    single.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    single.add_argument("--no-postprocess", action="store_true")
    
    # Ensemble mode
    ensemble = subparsers.add_parser("ensemble", help="Multi-fold ensemble inference")
    ensemble.add_argument("--ckpt-dir", default=cfg.ENSEMBLE_CKPT_DIR,
                         help="Directory with fold_*/best_model.pth or checkpoint list")
    ensemble.add_argument("--ckpts", nargs="+", help="Explicit checkpoint paths")
    ensemble.add_argument("--data-dir", default=cfg.DATA_DIR)
    ensemble.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    ensemble.add_argument("--n-folds", type=int, default=cfg.NUM_FOLDS)
    ensemble.add_argument("--no-postprocess", action="store_true")
    
    # GliGAN ensemble mode
    gligan = subparsers.add_parser("gligan_ensemble", help="GliGAN ensemble inference")
    gligan.add_argument("--gligan-root", default=cfg.GLIGAN_ROOT)
    gligan.add_argument("--mednext-ckpt", default=cfg.GLIGAN_MEDNEXT_CKPT)
    gligan.add_argument("--data-dir", default=cfg.DATA_DIR)
    gligan.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    gligan.add_argument("--no-postprocess", action="store_true")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not args.mode:
        print("Error: Please specify an inference mode")
        print("Available modes: baseline, ablation, no_ensemble, ensemble, gligan_ensemble")
        sys.exit(1)
    
    print(f"PyTorch {torch.__version__} | Device: {device}")
    print(f"MedNeXt-{cfg.MODEL_SIZE} | Patch size: {cfg.PATCH_SIZE}")
    
    use_amp = cfg.USE_AMP and (device.type == "cuda")
    postprocess = not args.no_postprocess
    
    if args.mode == "baseline":
        run_baseline_inference(
            args.data_dir, cfg.PATCH_SIZE, args.batch_size, use_amp, postprocess,
            args.ckpt
        )
    
    elif args.mode == "ablation":
        run_ablation_inference(
            args.data_dir, cfg.PATCH_SIZE, args.batch_size, use_amp, postprocess,
            args.loss
        )
    
    elif args.mode == "no_ensemble":
        run_single_model_inference(
            args.data_dir, cfg.PATCH_SIZE, args.batch_size, use_amp, postprocess,
            args.ckpt
        )
    
    elif args.mode == "ensemble":
        ckpt_input = args.ckpts if args.ckpts else args.ckpt_dir
        run_ensemble_inference(
            args.data_dir, cfg.PATCH_SIZE, args.batch_size, use_amp, postprocess,
            ckpt_input, args.n_folds
        )
    
    elif args.mode == "gligan_ensemble":
        run_gligan_ensemble_inference(
            args.data_dir, cfg.PATCH_SIZE, args.batch_size, use_amp, postprocess,
            args.gligan_root, args.mednext_ckpt
        )
    
    print("\n  Done.")


if __name__ == "__main__":
    main()
