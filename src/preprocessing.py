"""Data validation, splitting, and leakage-safe preprocessing."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DataSplits:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def validate_dataset(data: pd.DataFrame, target_column: str = "Class") -> pd.DataFrame:
    """Validate the expected binary-classification table without deleting rows."""
    if target_column not in data.columns:
        raise ValueError(f"Dataset must contain target column {target_column!r}.")
    if data.empty:
        raise ValueError("Dataset is empty.")
    if data.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")

    features = data.drop(columns=target_column)
    if features.empty or not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("All feature columns must be numeric.")
    if data[target_column].isna().any() or features.isna().any().any():
        raise ValueError("Dataset contains missing values; resolve them before training.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Feature columns contain infinite values.")

    target_values = set(data[target_column].unique())
    if not target_values.issubset({0, 1}) or target_values != {0, 1}:
        raise ValueError("Target must contain both binary labels 0 and 1.")
    return data.copy()


def split_dataset(
    data: pd.DataFrame,
    target_column: str = "Class",
    random_state: int = 42,
) -> DataSplits:
    """Create stratified 60/20/20 train, validation, and test partitions."""
    validated = validate_dataset(data, target_column)
    X = validated.drop(columns=target_column)
    y = validated[target_column].astype("int8")

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=random_state,
        stratify=y_development,
    )
    return DataSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    """Scale transaction time and amount, fitting statistics inside each pipeline fit."""
    scale_columns = [name for name in ("Time", "Amount") if name in feature_names]
    return ColumnTransformer(
        transformers=[("scaled", StandardScaler(), scale_columns)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
