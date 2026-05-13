
# MedNeXt Training Script for BraTS 2024 SSA Brain Tumor Segmentation
# Trains MedNeXt on the BraTS-SSA dataset with:
# - 4 input modalities (T1, T1ce, T2w, T2-FLAIR)
# - 3 output region channels using region-based approach:
#     - ET  (Enhancing Tumor):  label == 3
#     - TC  (Tumor Core):       label in {1, 3}
#     - WT  (Whole Tumor):      label in {1, 2, 3}
# - Per-class Dice score evaluation on validation set


import os
import sys
import argparse
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

import torchio as tio
from torchio.data import SubjectsLoader, SubjectsDataset

from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric

sys.path.append(os.path.join(os.path.dirname(__file__), "mednext"))
from nnunet_mednext import create_mednext_v1

# Convert integer label map (B,1,H,W,D) to region-based binary masks (B,3,H,W,D).
def labels_to_regions(seg: torch.Tensor) -> torch.Tensor:

    et = (seg == 3).float()
    tc = ((seg == 1) | (seg == 3)).float()
    wt = ((seg == 1) | (seg == 2) | (seg == 3)).float()
    return torch.cat([et, tc, wt], dim=1)


# Dataset
# Loads all 4 MRI modalities + segmentation mask for each subject

class BraTSSSADataset(SubjectsDataset):
    

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        subject_dirs = sorted(
            [d for d in glob(os.path.join(root_dir, "BraTS-SSA-*")) if os.path.isdir(d)]
        )

        subjects = []
        for sdir in subject_dirs:
            sid = os.path.basename(sdir)
            t1n = os.path.join(sdir, f"{sid}-t1n.nii.gz")
            t1c = os.path.join(sdir, f"{sid}-t1c.nii.gz")
            t2w = os.path.join(sdir, f"{sid}-t2w.nii.gz")
            t2f = os.path.join(sdir, f"{sid}-t2f.nii.gz")
            seg = os.path.join(sdir, f"{sid}-seg.nii.gz")

            # Skip incomplete subjects
            if not all(os.path.exists(p) for p in [t1n, t1c, t2w, t2f, seg]):
                print(f"Skipping {sid}: missing files")
                continue

            subjects.append(
                tio.Subject(
                    t1n=tio.ScalarImage(t1n),
                    t1c=tio.ScalarImage(t1c),
                    t2w=tio.ScalarImage(t2w),
                    t2f=tio.ScalarImage(t2f),
                    seg=tio.LabelMap(seg),
                )
            )

        print(f"Found {len(subjects)} subjects in {root_dir}")
        super().__init__(subjects, transform=transform)


# Stack the 4 modalities into a single 4-channel input tensor.

def collate_4ch(batch):

    t1n = torch.stack([s["t1n"][tio.DATA] for s in batch])  # (B,1,H,W,D)
    t1c = torch.stack([s["t1c"][tio.DATA] for s in batch])
    t2w = torch.stack([s["t2w"][tio.DATA] for s in batch])
    t2f = torch.stack([s["t2f"][tio.DATA] for s in batch])
    images = torch.cat([t1n, t1c, t2w, t2f], dim=1)        # (B,4,H,W,D)
    masks = torch.stack([s["seg"][tio.DATA] for s in batch]) # (B,1,H,W,D)
    return images, masks

# Transforms

def get_transforms(patch_size=(128, 160, 112)):
    preprocessing = tio.Compose([
        tio.ToCanonical(),
        tio.ZNormalization(masking_method=lambda x: x > 0),  # Normalize non-zero voxels
        tio.CropOrPad(patch_size),
    ])

    augmentation = tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2), p=0.5),
        tio.RandomAffine(
            scales=(0.9, 1.1),
            degrees=15,
            translation=5,
            p=0.3,
        ),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.3),
        tio.RandomNoise(std=(0, 0.1), p=0.2),
        tio.RandomBlur(std=(0, 1.0), p=0.2),
    ])

    train_transforms = tio.Compose([preprocessing, augmentation])
    val_transforms = preprocessing

    return train_transforms, val_transforms


# Dice computation per region

def compute_dice_per_region(preds: torch.Tensor, targets: torch.Tensor, eps=1e-7):

    region_names = ["ET", "TC", "WT"]
    dice_scores = {}
    for i, name in enumerate(region_names):
        p = preds[:, i].float().reshape(preds.shape[0], -1)
        t = targets[:, i].float().reshape(targets.shape[0], -1)
        intersection = (p * t).sum(dim=1)
        union = p.sum(dim=1) + t.sum(dim=1)
        dice = (2.0 * intersection + eps) / (union + eps)
        dice_scores[name] = dice.mean().item()
    return dice_scores


# Visualization

def save_comparison(images, targets, preds_binary, epoch, save_dir="predictions"):

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(save_dir, exist_ok=True)

    # Use first sample in batch, find slice with most tumor (WT channel)
    wt_mask = targets[0, 2].cpu().numpy()  # WT channel
    pixels_per_slice = wt_mask.sum(axis=(0, 1))
    best_slice = int(pixels_per_slice.argmax())
    if pixels_per_slice[best_slice] == 0:
        best_slice = wt_mask.shape[-1] // 2

    s = best_slice
    region_names = ["ET", "TC", "WT"]
    colors_gt = ["green", "blue", "yellow"]
    colors_pred = ["red", "cyan", "orange"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Top row: input modalities
    modality_names = ["T1", "T1ce", "T2w", "T2-FLAIR"]
    for i in range(4):
        axes[0, i].imshow(images[0, i, :, :, s].cpu().numpy(), cmap="gray")
        axes[0, i].set_title(modality_names[i])
        axes[0, i].axis("off")

    # Bottom row: GT regions, Pred regions, overlay
    for i in range(3):
        axes[1, i].imshow(targets[0, i, :, :, s].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"GT {region_names[i]}")
        axes[1, i].axis("off")

    # Overlay
    inp_slice = images[0, 0, :, :, s].cpu().numpy()
    axes[1, 3].imshow(inp_slice, cmap="gray")
    for i, (cg, cp) in enumerate(zip(colors_gt, colors_pred)):
        gt_s = targets[0, i, :, :, s].cpu().numpy()
        pr_s = preds_binary[0, i, :, :, s].cpu().numpy()
        if gt_s.max() > 0:
            axes[1, 3].contour(gt_s, levels=[0.5], colors=cg, linewidths=1.5)
        if pr_s.max() > 0:
            axes[1, 3].contour(pr_s, levels=[0.5], colors=cp, linewidths=1.5, linestyles="dashed")
    axes[1, 3].set_title("Overlay (solid=GT, dashed=pred)")
    axes[1, 3].axis("off")

    plt.suptitle(f"Epoch {epoch}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"comparison_epoch_{epoch}.png"), bbox_inches="tight", dpi=150)
    plt.close()


# Training

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Transforms
    train_transforms, val_transforms = get_transforms(
        patch_size=tuple(args.patch_size)
    )

    # Dataset
    DATA_DIR = "ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
    full_dataset = BraTSSSADataset(root_dir=DATA_DIR, transform=None)
    all_subjects = full_dataset.subjects

    # Train/val split
    num_subjects = len(all_subjects)
    num_train = int(args.train_ratio * num_subjects)
    num_val = num_subjects - num_train

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(num_subjects, generator=generator).tolist()
    train_subjects = [all_subjects[i] for i in indices[:num_train]]
    val_subjects = [all_subjects[i] for i in indices[num_train:]]

    train_dataset = SubjectsDataset(train_subjects, transform=train_transforms)
    val_dataset = SubjectsDataset(val_subjects, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_4ch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_4ch,
        pin_memory=True,
    )

    print(f"Train: {num_train} subjects, Val: {num_val} subjects")

    # Model: 4 input channels (modalities), 3 output channels (ET, TC, WT)
    model = create_mednext_v1(
        num_input_channels=4,
        num_classes=3,
        model_id=args.model_size,
        kernel_size=args.kernel_size,
        deep_supervision=args.deep_supervision,
    )
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: MedNeXt-{args.model_size} (k={args.kernel_size}) | Params: {num_params:,}")

    # Loss: Dice + Focal
    loss_fn = DiceFocalLoss(
        sigmoid=True,
        focal_weight=1.0,
        lambda_dice=1.0,
        gamma=2.0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Mixed precision
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None

    # Training state
    best_val_dice = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_dice_ET": [], "val_dice_TC": [], "val_dice_WT": [], "val_dice_avg": []}

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("predictions", exist_ok=True)

    if device.type == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()

    for epoch in range(1, args.epochs + 1):
        
        # Training
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for step, (images, masks) in enumerate(pbar):
            images = images.to(device)       # (B, 4, H, W, D)
            targets = labels_to_regions(masks.to(device))  # (B, 3, H, W, D)

            if use_amp:
                with autocast("cuda"):
                    outputs = model(images)
                    if args.deep_supervision and isinstance(outputs, list):
                        # Deep supervision: weighted sum of losses at multiple scales
                        ds_weights = [1.0, 0.5, 0.25, 0.125, 0.0625]
                        loss = 0.0
                        for i, (out, w) in enumerate(zip(outputs, ds_weights)):
                            # Resize target to match output resolution
                            if out.shape != targets.shape:
                                t = F.interpolate(targets, size=out.shape[2:], mode="nearest")
                            else:
                                t = targets
                            loss += w * loss_fn(out, t)
                        loss = loss / sum(ds_weights[:len(outputs)])
                    else:
                        if isinstance(outputs, list):
                            outputs = outputs[0]
                        loss = loss_fn(outputs, targets)
                    loss = loss / args.accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(images)
                if args.deep_supervision and isinstance(outputs, list):
                    ds_weights = [1.0, 0.5, 0.25, 0.125, 0.0625]
                    loss = 0.0
                    for i, (out, w) in enumerate(zip(outputs, ds_weights)):
                        if out.shape != targets.shape:
                            t = F.interpolate(targets, size=out.shape[2:], mode="nearest")
                        else:
                            t = targets
                        loss += w * loss_fn(out, t)
                    loss = loss / sum(ds_weights[:len(outputs)])
                else:
                    if isinstance(outputs, list):
                        outputs = outputs[0]
                    loss = loss_fn(outputs, targets)
                loss = loss / args.accum_steps
                loss.backward()

                if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += loss.item() * args.accum_steps
            pbar.set_postfix(loss=f"{loss.item() * args.accum_steps:.4f}")

            del images, targets, outputs, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

        scheduler.step()
        epoch_loss /= len(train_loader)
        history["train_loss"].append(epoch_loss)

        # Validation
        model.eval()
        val_dice_sums = {"ET": 0.0, "TC": 0.0, "WT": 0.0}
        val_count = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                images = images.to(device)
                targets = labels_to_regions(masks.to(device))

                if use_amp:
                    with autocast("cuda"):
                        outputs = model(images)
                else:
                    outputs = model(images)

                if isinstance(outputs, list):
                    outputs = outputs[0]

                preds_binary = (torch.sigmoid(outputs) > 0.5).float()
                batch_dice = compute_dice_per_region(preds_binary, targets)

                for k in val_dice_sums:
                    val_dice_sums[k] += batch_dice[k] * images.shape[0]
                val_count += images.shape[0]

                # Save comparison on last batch of selected epochs
                if epoch % args.save_every == 0 or epoch == 1:
                    save_comparison(images, targets, preds_binary, epoch)

            del images, targets, outputs, preds_binary
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Average dice scores
        val_dice = {k: v / val_count for k, v in val_dice_sums.items()}
        val_dice_avg = np.mean(list(val_dice.values()))

        history["val_dice_ET"].append(val_dice["ET"])
        history["val_dice_TC"].append(val_dice["TC"])
        history["val_dice_WT"].append(val_dice["WT"])
        history["val_dice_avg"].append(val_dice_avg)

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch} | Loss: {epoch_loss:.4f} | "
            f"Dice ET: {val_dice['ET']:.4f}  TC: {val_dice['TC']:.4f}  WT: {val_dice['WT']:.4f}  "
            f"Avg: {val_dice_avg:.4f} | LR: {lr_now:.6f}"
        )

        # Checkpointing
        if val_dice_avg > best_val_dice:
            best_val_dice = val_dice_avg
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_dice": val_dice,
                    "val_dice_avg": val_dice_avg,
                    "train_loss": epoch_loss,
                    "args": vars(args),
                },
                os.path.join(args.checkpoint_dir, "best_model.pth"),
            )
            print(f"  -> New best model saved! (Avg Dice: {val_dice_avg:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

        # Save latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_dice": val_dice,
                "val_dice_avg": val_dice_avg,
                "train_loss": epoch_loss,
                "args": vars(args),
            },
            os.path.join(args.checkpoint_dir, "latest_model.pth"),
        )

    print(f"\nTraining complete. Best Avg Val Dice: {best_val_dice:.4f}")

    # Plot training curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        epochs_range = range(1, len(history["train_loss"]) + 1)
        ax1.plot(epochs_range, history["train_loss"], "b-", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Loss")
        ax1.set_title("Training Loss (Dice + Focal)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs_range, history["val_dice_ET"], label="ET", linewidth=2)
        ax2.plot(epochs_range, history["val_dice_TC"], label="TC", linewidth=2)
        ax2.plot(epochs_range, history["val_dice_WT"], label="WT", linewidth=2)
        ax2.plot(epochs_range, history["val_dice_avg"], "k--", label="Avg", linewidth=2)
        ax2.axhline(y=best_val_dice, color="r", linestyle=":", alpha=0.5, label=f"Best Avg: {best_val_dice:.4f}")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Dice Score")
        ax2.set_title("Validation Dice Scores per Region")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("predictions/training_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Training curves saved to predictions/training_curves.png")
    except ImportError:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train MedNeXt on BraTS-SSA")
    parser.add_argument("--model_size", type=str, default="B", choices=["S", "B", "M", "L"],
                        help="MedNeXt model size")
    parser.add_argument("--kernel_size", type=int, default=3, help="Convolution kernel size")
    parser.add_argument("--deep_supervision", action="store_true", help="Enable deep supervision")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 160, 112],
                        help="Patch size (H W D)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per step")
    parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps (effective batch = batch_size * accum_steps)")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2.7e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--save_every", type=int, default=5, help="Save comparison images every N epochs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
