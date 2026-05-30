"""
Build GRPO training datasets from MP-20 CSV (same layout as SFT notebook).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from crystext_rewards import build_prompt

try:
    import pandas as pd
    from datasets import Dataset
except ImportError as exc:
    raise ImportError("Install pandas and datasets: pip install pandas datasets") from exc


def row_to_record(row: Any) -> Dict[str, str]:
    formula = str(row["pretty_formula"])
    spacegroup = str(int(row["spacegroup.number"]))
    reference_cif = str(row["cif"])
    return {
        "prompt": build_prompt(formula=formula, spacegroup=spacegroup),
        "formula": formula,
        "spacegroup": spacegroup,
        "reference_cif": reference_cif,
    }


def load_mp20_csv(
    csv_path: str,
    start: int = 0,
    end: Optional[int] = None,
) -> "Dataset":
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"MP-20 CSV not found: {csv_path}\n"
            "Download MP-20 (DiffCSP layout) or pass --dataset_jsonl with a prepared file."
        )
    df = pd.read_csv(path)
    if end is not None:
        df = df.iloc[start:end]
    else:
        df = df.iloc[start:]
    records = [row_to_record(row) for _, row in df.iterrows()]
    return Dataset.from_list(records)


def load_jsonl(jsonl_path: str, limit: Optional[int] = None) -> "Dataset":
    records: List[Dict[str, str]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            obj = json.loads(line)
            if "prompt" not in obj:
                obj = row_to_record_from_dict(obj)
            records.append(obj)
    return Dataset.from_list(records)


def row_to_record_from_dict(obj: Dict[str, Any]) -> Dict[str, str]:
    formula = obj.get("formula") or obj.get("pretty_formula")
    spacegroup = str(obj.get("spacegroup") or obj.get("spacegroup.number"))
    reference_cif = obj.get("reference_cif") or obj.get("cif")
    if formula is None or reference_cif is None:
        raise ValueError("JSONL record needs formula/pretty_formula and cif/reference_cif")
    return {
        "prompt": build_prompt(formula=str(formula), spacegroup=spacegroup),
        "formula": str(formula),
        "spacegroup": spacegroup,
        "reference_cif": str(reference_cif),
    }


def export_jsonl(csv_path: str, out_path: str, start: int = 0, end: Optional[int] = None) -> int:
    ds = load_mp20_csv(csv_path, start=start, end=end)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")
    return len(ds)


def default_csv_path() -> str:
    candidates = [
        "DiffCSP/data/mp_20/train.csv",
        os.path.join("data", "mp_20", "train.csv"),
        os.path.join("..", "DiffCSP", "data", "mp_20", "train.csv"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return candidates[0]
