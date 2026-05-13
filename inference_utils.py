
import os
import torch
from contextlib import nullcontext
from pathlib import Path

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast


def resolve_path(path):
    """Resolve path relative to workspace root."""
    if os.path.isabs(path):
        return path
    # Assume we're in standalone_inference/; go up one level to workspace root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, path)


def load_mednext_model(ckpt_path, model_size, kernel_size, device, label="MedNeXt"):
    """Load MedNeXt model from checkpoint and return (model, ckpt)."""
    abs_path = resolve_path(ckpt_path)
    
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Checkpoint not found: {abs_path}")
    
    # Import MedNeXt creation function. Prefer local adapter if present.
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
        sys.path.insert(0, os.path.join(root, "src"))
        sys.path.insert(0, os.path.join(root, "MedNeXt"))

    create_mednext_v1 = None
    try:
        # Prefer the adapter bundled with standalone_inference
        from standalone_inference.mednext_adapter import create_mednext_v1
    except Exception:
        try:
            from mednext_adapter import create_mednext_v1
        except Exception:
            try:
                from models.mednext import create_mednext_v1
            except Exception:
                try:
                    from nnunet_mednext.mednext_v1 import create_mednext_v1
                except Exception as e:
                    raise ImportError(
                        "Could not import MedNeXt model.\n"
                        "Add the MedNeXt model code to the workspace or copy it into\n"
                        "`standalone_inference/mednext.py` so that `create_mednext_v1` is available.\n"
                        f"Underlying error: {e}"
                    )
    
    model = create_mednext_v1(
        num_input_channels=4,
        num_classes=3,
        model_id=model_size,
        kernel_size=kernel_size,
        deep_supervision=True,
    ).to(device)
    
    ckpt = torch.load(abs_path, map_location=device, weights_only=False)
    key = "model_state_dict" if "model_state_dict" in ckpt else "mednext_state_dict"
    model.load_state_dict(ckpt[key])
    model.eval()
    
    epoch = ckpt.get("epoch", "?")
    val_dice = ckpt.get("val_dice_avg", "?")
    val_dice_str = f"{val_dice:.4f}" if isinstance(val_dice, float) else str(val_dice)
    print(f"  [{label}] {os.path.relpath(abs_path)} (epoch={epoch}, val_dice={val_dice_str})")
    
    return model, ckpt


def predict_single_model(model, images, use_amp, device):
    """Return logits from a single model."""
    ctx = autocast("cuda") if use_amp else nullcontext()
    
    with torch.no_grad(), ctx:
        output = model(images)
    
    # Handle both direct tensor and tuple/list outputs
    if isinstance(output, (list, tuple)):
        logits = output[0]
    else:
        logits = output
    
    return logits


def predict_ensemble(models, images, use_amp, device, weights=None):
    """Soft-average predictions from multiple models."""
    if weights is None:
        weights = [1.0] * len(models)
    
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    probs = torch.zeros(images.shape[0], 3, *images.shape[2:], device=device)
    
    for model, weight in zip(models, weights):
        logits = predict_single_model(model, images, use_amp, device)
        probs = probs + weight * torch.sigmoid(logits)
    
    return probs


def discover_fold_checkpoints(ckpt_dir_or_template, num_folds):
    """Find fold checkpoints from a dir or template."""
    abs_path = resolve_path(ckpt_dir_or_template)
    
    if "{fold}" in abs_path:
        # Template mode
        ckpts = []
        for fold_idx in range(num_folds):
            ckpt_path = abs_path.format(fold=fold_idx)
            if os.path.isfile(ckpt_path):
                ckpts.append(ckpt_path)
        return ckpts
    else:
        # Directory mode
        from glob import glob
        if os.path.isdir(abs_path):
            paths = sorted(glob(os.path.join(abs_path, "fold_*", "best_model.pth")))
            return paths
        elif os.path.isfile(abs_path):
            return [abs_path]
        else:
            return []
