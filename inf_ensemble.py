"""Wrapper to run multi-fold ensemble inference (calls `run_ensemble_inference`)."""
import argparse
from run_inference import run_ensemble_inference


def parse_args():
    p = argparse.ArgumentParser(prog="inf_ensemble.py")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--ckpts", nargs="+", help="Explicit checkpoint paths")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-folds", type=int, default=None)
    p.add_argument("--no-postprocess", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    postprocess = not args.no_postprocess
    ckpt_input = args.ckpts if args.ckpts else args.ckpt_dir
    run_ensemble_inference(args.data_dir, None, args.batch_size, False, postprocess, ckpt_input, args.n_folds)


if __name__ == "__main__":
    main()
