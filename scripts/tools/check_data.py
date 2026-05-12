#!/usr/bin/env python3
"""Check that ESA-ADB Lite data paths are ready before a long run."""
from __future__ import annotations

import argparse
from pathlib import Path


TRAIN_NAMES = ["3_months", "10_months", "21_months", "42_months", "84_months"]


def gb(size: int) -> float:
    return size / 1024**3


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ESA-ADB Lite data layout")
    parser.add_argument(
        "--data-path",
        default="../data/preprocessed",
        help="Preprocessed data root. Must contain multivariate/ESA-Mission1-semi-supervised",
    )
    parser.add_argument(
        "--mission-root",
        default="../data/esa-adb/mission1/ESA-Mission1",
        help="Raw Mission1 metadata root. Must contain labels.csv, anomaly_types.csv, channels.csv",
    )
    args = parser.parse_args()

    base = Path(args.data_path) / "multivariate" / "ESA-Mission1-semi-supervised"
    mission_root = Path(args.mission_root)

    required = [base / "84_months.test.csv"]
    required.extend(base / f"{name}.train.csv" for name in TRAIN_NAMES)
    required.extend(mission_root / name for name in ["labels.csv", "anomaly_types.csv", "channels.csv"])

    missing = [path for path in required if not path.exists()]
    if missing:
        print("Missing files:")
        for path in missing:
            print(f"  {path}")
        return 1

    print("Data layout OK.")
    print("CSV sizes:")
    for path in required:
        if path.suffix == ".csv" and "preprocessed" in str(path):
            print(f"  {path.name:20s} {gb(path.stat().st_size):7.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

