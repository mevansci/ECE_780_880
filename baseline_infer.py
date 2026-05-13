import argparse
from run_inference import run_single_model_inference
import config as cfg


def parse_args():
    p = argparse.ArgumentParser(prog="baseline_infer.py")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default=cfg.DATA_DIR)
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--no-postprocess", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    postprocess = not args.no_postprocess
    run_single_model_inference(args.data_dir, cfg.PATCH_SIZE, args.batch_size, cfg.USE_AMP, postprocess, args.ckpt)


if __name__ == "__main__":
    main()
