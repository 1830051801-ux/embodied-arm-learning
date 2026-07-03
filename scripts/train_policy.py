from __future__ import annotations

import argparse
from pathlib import Path

from embodied_arm.train_bc import train_behavior_cloning


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/sample_grasp_log.csv")
    parser.add_argument("--out", default="models/bc_policy.pt")
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    metrics = train_behavior_cloning(args.csv, args.out, epochs=args.epochs)
    print(metrics)


if __name__ == "__main__":
    main()

