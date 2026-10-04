"""Shared input and output helpers."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing import validate_dataset


DEFAULT_DATA_PATH = Path("data/creditcard.csv")


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and validate the Kaggle-format transaction CSV."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Download creditcard.csv from Kaggle "
            "and place it at data/creditcard.csv, or pass --data-path."
        )
    return validate_dataset(pd.read_csv(dataset_path))


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")
