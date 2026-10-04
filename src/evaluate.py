"""Fraud-focused evaluation and diagnostic plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def select_f1_threshold(y_true: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    """Choose a probability threshold using validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.5, 0.0
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def calculate_metrics(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, object]:
    predictions = (scores >= threshold).astype("int8")
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": ["legitimate (0)", "fraud (1)"],
    }


def save_diagnostic_plots(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
    output_dir: str | Path,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    predictions = (scores >= threshold).astype("int8")
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])

    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        xlabel="Predicted label",
        ylabel="True label",
        title="Test confusion matrix",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Transactions")
    figure.tight_layout()
    figure.savefig(destination / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(false_positive_rate, true_positive_rate)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].plot(recall, precision)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    figure.tight_layout()
    figure.savefig(destination / "roc_precision_recall.png", dpi=160)
    plt.close(figure)
