"""Wrapper to run single-model (no ensemble) inference (calls `run_single_model_inference`)."""
import argparse
from run_inference import run_single_model_inference


def parse_args():
    p = argparse.ArgumentParser(prog="inf_no_ensemble.py")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--no-postprocess", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    postprocess = not args.no_postprocess
    run_single_model_inference(args.data_dir, None, args.batch_size, False, postprocess, args.ckpt)


if __name__ == "__main__":
    main()
