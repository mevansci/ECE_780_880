import argparse
import os
import sys
import time
from typing import Optional, Sequence

import torch
import torch.nn as nn

_STANDALONE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_STANDALONE_DIR)
sys.path.insert(0, _STANDALONE_DIR)
sys.path.append(_ROOT)
sys.path.append(os.path.join(_ROOT, "src"))

from inference_utils import load_mednext_model
from data.dataset import build_eval_loader


class Trainer:

    def __init__(self, data_dir: str, out_dir: str, size: str = "S", kernel: int = 3,
                 batch_size: int = 2, lr: float = 1e-4, loss_name: str = "dscpp",
                 device: Optional[torch.device] = None):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.size = size
        self.kernel = kernel
        self.batch_size = batch_size
        self.lr = lr
        self.loss_name = loss_name
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Create model via MedNeXt factory
        try:
            from standalone_inference.mednext_adapter import create_mednext_v1
        except Exception:
            try:
                from models.mednext import create_mednext_v1
            except Exception:
                raise ImportError("create_mednext_v1 not found; clone MedNeXt per README")

        self.model = create_mednext_v1(
            num_input_channels=4,
            num_classes=3,
            model_id=self.size,
            kernel_size=self.kernel,
            deep_supervision=True,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = self._make_loss(self.loss_name)

    def _make_loss(self, name: str):
        bce = nn.BCEWithLogitsLoss()

        def dscpp_loss(logits, targets):
            probs = torch.sigmoid(logits)
            smooth = 1e-6
            probs_flat = probs.view(probs.size(0), -1)
            targ_flat = (targets > 0).float().view(targets.size(0), -1)
            inter = (probs_flat * targ_flat).sum(dim=1)
            denom = probs_flat.sum(dim=1) + targ_flat.sum(dim=1)
            dice = (2 * inter + smooth) / (denom + smooth)
            return 1.0 - dice.mean()

        def ce_loss(logits, targets):
            if targets.ndim == logits.ndim:
                targets = targets.squeeze(1).long()
            return nn.CrossEntropyLoss()(logits, targets.long())

        # Expose plain Cross-Entropy as a named option
        if name == "ce":
            return ce_loss

        def combined(logits, targets):
            return dscpp_loss(logits, targets) + ce_loss(logits, targets)

        if name == "dscpp":
            return dscpp_loss
        if name == "dscpp_ce":
            return combined
        if name == "dscpp_ce_ace":
            return combined
        return bce

    def _build_loader(self, shuffle: bool = True):
        return build_eval_loader(self.data_dir, batch_size=self.batch_size, shuffle=shuffle)

    def _save_checkpoint(self, epoch: int):
        os.makedirs(self.out_dir, exist_ok=True)
        ckpt = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
        }
        path = os.path.join(self.out_dir, f"epoch_{epoch:03d}.pth")
        torch.save(ckpt, path)
        return path

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        count = 0
        for batch in loader:
            imgs = batch["image"].to(self.device)
            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            if labels is None:
                loss = torch.tensor(0.0, device=self.device)
            else:
                loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            count += 1
        return total_loss / max(1, count)

    def run(self, epochs: int = 1):
        loader = self._build_loader(shuffle=True)
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            loss = self.train_epoch(loader)
            ckpt = self._save_checkpoint(epoch)
            print(f"Epoch {epoch}/{epochs} loss={loss:.4f} saved={ckpt} time={time.time()-t0:.1f}s")


class CrossValidator:

    def __init__(self, start_fold: int, end_fold: int, base_out_dir: str, **trainer_kwargs):
        self.start_fold = start_fold
        self.end_fold = end_fold
        self.base_out_dir = base_out_dir
        self.trainer_kwargs = trainer_kwargs

    def run(self, epochs: int = 1):
        for fold in range(self.start_fold, self.end_fold + 1):
            print(f"Running fold {fold}")
            out_dir = os.path.join(self.base_out_dir, f"fold_{fold}")
            trainer = Trainer(out_dir=out_dir, **self.trainer_kwargs)
            trainer.run(epochs=epochs)


class AblationRunner:

    def __init__(self, losses: Sequence[str], out_dir: str, **trainer_kwargs):
        self.losses = list(losses)
        self.out_dir = out_dir
        self.trainer_kwargs = trainer_kwargs

    def run(self, epochs: int = 1):
        for loss in self.losses:
            print(f"Starting ablation run: {loss}")
            out_dir = os.path.join(self.out_dir, f"ablation_{loss}")
            trainer = Trainer(out_dir=out_dir, loss_name=loss, **self.trainer_kwargs)
            trainer.run(epochs=epochs)


def parse_args():
    p = argparse.ArgumentParser(prog="run_training.py")
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default="data")
    common.add_argument("--out-dir", default="checkpoints-standalone")
    common.add_argument("--size", default="S")
    common.add_argument("--kernel", type=int, default=3)
    common.add_argument("--batch-size", type=int, default=2)
    common.add_argument("--epochs", type=int, default=2)
    common.add_argument("--lr", type=float, default=1e-4)

    t = sub.add_parser("train", parents=[common])
    c = sub.add_parser("crossval", parents=[common])
    c.add_argument("--start-fold", type=int, default=0)
    c.add_argument("--end-fold", type=int, default=4)

    a = sub.add_parser("ablation", parents=[common])
    a.add_argument("--losses", nargs="*", help="list of ablation loss names")

    return p.parse_args()


def main():
    args = parse_args()
    trainer_kwargs = dict(
        data_dir=args.data_dir,
        size=args.size,
        kernel=args.kernel,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    if args.mode == "train":
        trainer = Trainer(out_dir=args.out_dir, **trainer_kwargs)
        trainer.run(epochs=args.epochs)
    elif args.mode == "crossval":
        cv = CrossValidator(start_fold=args.start_fold, end_fold=args.end_fold,
                            base_out_dir=args.out_dir, **trainer_kwargs)
        cv.run(epochs=args.epochs)
    elif args.mode == "ablation":
        losses = args.losses or ["dscpp", "dscpp_ce", "dscpp_ce_ace"]
        ar = AblationRunner(losses=losses, out_dir=args.out_dir, **trainer_kwargs)
        ar.run(epochs=args.epochs)


if __name__ == "__main__":
    main()
