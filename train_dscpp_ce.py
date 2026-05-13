"""Train with combined DSC++ + CE.

Wrapper for `Trainer` using combined DSC++ + CE loss.
"""
import argparse
from run_training import Trainer


def parse_args():
    p = argparse.ArgumentParser(prog="train_dscpp_ce.py")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="checkpoints-dscpp-ce")
    p.add_argument("--size", default="S")
    p.add_argument("--kernel", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    return p.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(data_dir=args.data_dir, out_dir=args.out_dir,
                      size=args.size, kernel=args.kernel,
                      batch_size=args.batch_size, lr=args.lr,
                      loss_name="dscpp_ce")
    trainer.run(epochs=args.epochs)


if __name__ == "__main__":
    main()
