from __future__ import annotations

import argparse

from embodied_arm.synthetic_data import generate_synthetic_grasp_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_grasp_log.csv")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    generate_synthetic_grasp_log(args.out, samples=args.samples, seed=args.seed)
    print({"written": args.out, "samples": args.samples})


if __name__ == "__main__":
    main()

