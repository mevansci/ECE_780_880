"""Wrapper to run GliGAN ensemble inference (calls `run_gligan_ensemble_inference`)."""
import argparse
from run_inference import run_gligan_ensemble_inference


def parse_args():
    p = argparse.ArgumentParser(prog="inf_gligan_ensemble.py")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--gligan-root", default=None)
    p.add_argument("--mednext-ckpt", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--no-postprocess", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    postprocess = not args.no_postprocess
    run_gligan_ensemble_inference(args.data_dir, None, args.batch_size, False, postprocess, args.gligan_root, args.mednext_ckpt)


if __name__ == "__main__":
    main()
