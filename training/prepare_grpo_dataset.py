#!/usr/bin/env python
"""Export MP-20 CSV to JSONL for GRPO training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from training.dataset_utils import default_csv_path, export_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", default=None)
    p.add_argument("--out", default="data/mp20_grpo_train.jsonl")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    args = p.parse_args()
    csv_path = args.csv_path or default_csv_path()
    n = export_jsonl(csv_path, args.out, start=args.start, end=args.end)
    print(f"Wrote {n} records to {args.out}")


if __name__ == "__main__":
    main()
