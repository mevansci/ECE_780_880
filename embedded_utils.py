

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt
from typing import Dict, List
def labels_to_regions(seg: torch.Tensor) -> torch.Tensor:
    """Convert integer label map to region masks (ET, TC, WT)."""
    et = (seg == 3).float()
    tc = ((seg == 1) | (seg == 3)).float()
    wt = ((seg == 1) | (seg == 2) | (seg == 3)).float()
    return torch.cat([et, tc, wt], dim=1)


def unpack_mednext_outputs(outputs):
    """Normalize MedNeXt outputs to (logits, features, raw)."""
    if isinstance(outputs, dict):
        return outputs["logits"], outputs.get("features"), outputs
    if isinstance(outputs, (list, tuple)):
        return outputs[0], None, outputs
    return outputs, None, outputs


# BraTS metric computation
REGIONS = ["ET", "TC", "WT"]
METRICS = ["DSC", "HD95", "Sensitivity", "Specificity", "Precision"]
_MAX_HD = 374.0


def compute_hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """95th-percentile Hausdorff Distance."""
    pred_b = pred.astype(bool)
    targ_b = target.astype(bool)

    if not pred_b.any() and not targ_b.any():
        return 0.0
    if not pred_b.any() or not targ_b.any():
        return _MAX_HD

    pred_surf = pred_b ^ binary_erosion(pred_b)
    targ_surf = targ_b ^ binary_erosion(targ_b)

    dt_targ = distance_transform_edt(~targ_b)
    dt_pred = distance_transform_edt(~pred_b)

    d1 = dt_targ[pred_surf] if pred_surf.any() else np.array([])
    d2 = dt_pred[targ_surf] if targ_surf.any() else np.array([])

    distances = np.concatenate([d1, d2]) if len(d1) > 0 or len(d2) > 0 else np.array([0.0])
    return float(np.percentile(distances, 95) if distances.size > 0 else _MAX_HD)


def compute_region_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute region metrics (DSC, HD95, Sensitivity, Specificity, Precision)."""
    p = pred.astype(bool)
    t = target.astype(bool)
    eps = 1e-7

    tp = float((p & t).sum())
    fp = float((p & ~t).sum())
    fn = float((~p & t).sum())
    tn = float((~p & ~t).sum())

    return {
        "DSC": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "HD95": compute_hd95(p, t),
        "Sensitivity": (tp + eps) / (tp + fn + eps),
        "Specificity": (tn + eps) / (tn + fp + eps),
        "Precision": (tp + eps) / (tp + fp + eps),
    }


def compute_brats_metrics(pred_binary: np.ndarray, target: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute per-region BraTS metrics."""
    return {
        region: compute_region_metrics(pred_binary[i], target[i])
        for i, region in enumerate(REGIONS)
    }


def flatten_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Flatten nested metrics dict to flat keys."""
    return {
        f"{region}_{metric}": metrics[region][metric]
        for region in REGIONS
        for metric in METRICS
    }


def mean_record(records: List[Dict]) -> Dict:
    """Compute mean of numeric fields across records."""
    if not records:
        return {}
    keys = [k for k in records[0] if k != "patient_id"]
    out = {"patient_id": "MEAN"}
    for k in keys:
        vals = [r[k] for r in records if isinstance(r.get(k), (int, float))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def apply_postprocess_3ch(preds: np.ndarray, min_size=50) -> np.ndarray:
    """Morphological postprocessing: remove small components, fill holes."""
    from scipy import ndimage
    
    result = preds.copy()
    for ch in range(preds.shape[0]):
        pred_bin = preds[ch] > 0.5
        
        # Remove small components
        labeled, num_features = ndimage.label(pred_bin)
        for i in range(1, num_features + 1):
            if (labeled == i).sum() < min_size:
                pred_bin[labeled == i] = 0
        
        # Fill small holes
        pred_inv = ~pred_bin
        labeled_inv, num_inv = ndimage.label(pred_inv)
        for i in range(1, num_inv + 1):
            if (labeled_inv == i).sum() < min_size:
                pred_bin[labeled_inv == i] = 1
        
        result[ch] = pred_bin.astype(np.float32)
    
    return result


def compute_dice_per_region(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> dict:
    """Compute mean Dice per region across a batch."""
    dice_scores = {}
    for i, name in enumerate(REGIONS):
        p = preds[:, i].float().reshape(preds.shape[0], -1)
        t = targets[:, i].float().reshape(targets.shape[0], -1)
        intersection = (p * t).sum(dim=1)
        dice = (2.0 * intersection + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)
        dice_scores[name] = dice.mean().item()
    return dice_scores
