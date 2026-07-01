from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


DATASET_ID = 17
TARGET_COLUMN = "Diagnosis"
FEATURES_FILE = Path("data/wdbc_features.csv")
TARGET_FILE = Path("data/wdbc_target.csv")

CLASS_LABELS = {
    "B": "Tumor Benigno",
    "M": "Tumor Maligno",
}


@dataclass(frozen=True)
class DatasetBundle:
    features: pd.DataFrame
    target: pd.Series
    source: str


def load_dataset(cache: bool = True) -> DatasetBundle:
    """Load the WDBC dataset from local cache or UCI ML Repository."""
    if cache and FEATURES_FILE.exists() and TARGET_FILE.exists():
        features = pd.read_csv(FEATURES_FILE)
        target = pd.read_csv(TARGET_FILE)[TARGET_COLUMN]
        return DatasetBundle(features=features, target=target, source="cache")

    dataset = fetch_ucirepo(id=DATASET_ID)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    if TARGET_COLUMN not in targets.columns:
        raise ValueError(f"Target column {TARGET_COLUMN!r} not found in UCI dataset.")

    target = targets[TARGET_COLUMN].copy()

    if cache:
        FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(FEATURES_FILE, index=False)
        target.to_frame(TARGET_COLUMN).to_csv(TARGET_FILE, index=False)

    return DatasetBundle(features=features, target=target, source="uci")


def build_dataframe(bundle: DatasetBundle) -> pd.DataFrame:
    df = bundle.features.copy()
    df[TARGET_COLUMN] = bundle.target
    return df


def validate_dataset(features: pd.DataFrame, target: pd.Series) -> None:
    if len(features) != len(target):
        raise ValueError("Features and target have different row counts.")
    if features.empty:
        raise ValueError("Feature dataset is empty.")
    if target.empty:
        raise ValueError("Target dataset is empty.")
    missing = features.isna().sum().sum() + target.isna().sum()
    if missing:
        raise ValueError(f"Dataset has {missing} missing values.")
    labels = set(target.unique())
    expected = set(CLASS_LABELS)
    if labels != expected:
        raise ValueError(f"Unexpected target labels: {sorted(labels)}. Expected {sorted(expected)}.")
