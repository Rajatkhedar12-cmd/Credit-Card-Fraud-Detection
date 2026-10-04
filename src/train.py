"""Train baseline and tuned fraud classifiers without resampling held-out data."""

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.evaluate import calculate_metrics, save_diagnostic_plots, select_f1_threshold
from src.preprocessing import build_preprocessor, split_dataset
from src.utils import DEFAULT_DATA_PATH, load_dataset, write_json


def build_model_pipeline(
    classifier: Any,
    feature_names: list[str],
    use_smote: bool,
    random_state: int,
) -> Pipeline:
    steps: list[tuple[str, Any]] = [("preprocessor", build_preprocessor(feature_names))]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state, k_neighbors=5)))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def _validation_row(name: str, y_true: pd.Series, scores: np.ndarray) -> dict[str, float | str]:
    metrics = calculate_metrics(y_true, scores, threshold=0.5)
    return {
        "model": name,
        "threshold": 0.5,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "average_precision": metrics["average_precision"],
    }


def run_training(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = "results/run",
    random_state: int = 42,
    cv_folds: int = 3,
) -> dict[str, Any]:
    data = load_dataset(data_path)
    splits = split_dataset(data, random_state=random_state)
    feature_names = list(splits.X_train.columns)

    baseline = build_model_pipeline(
        LogisticRegression(max_iter=1000), feature_names, use_smote=False, random_state=random_state
    )
    baseline.fit(splits.X_train, splits.y_train)

    smote_baseline = build_model_pipeline(
        LogisticRegression(max_iter=1000), feature_names, use_smote=True, random_state=random_state
    )
    smote_baseline.fit(splits.X_train, splits.y_train)

    cross_validation = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    xgboost_pipeline = build_model_pipeline(
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        ),
        feature_names,
        use_smote=True,
        random_state=random_state,
    )
    parameter_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1],
    }
    search = GridSearchCV(
        estimator=xgboost_pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cross_validation,
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    search.fit(splits.X_train, splits.y_train)

    validation_rows = [
        _validation_row(
            "Logistic regression (no SMOTE)",
            splits.y_validation,
            baseline.predict_proba(splits.X_validation)[:, 1],
        ),
        _validation_row(
            "Logistic regression (SMOTE)",
            splits.y_validation,
            smote_baseline.predict_proba(splits.X_validation)[:, 1],
        ),
        _validation_row(
            "Tuned XGBoost (SMOTE, default threshold)",
            splits.y_validation,
            search.best_estimator_.predict_proba(splits.X_validation)[:, 1],
        ),
    ]
    validation_comparison = pd.DataFrame(validation_rows)

    validation_scores = search.best_estimator_.predict_proba(splits.X_validation)[:, 1]
    threshold, validation_f1 = select_f1_threshold(splits.y_validation, validation_scores)

    X_development = pd.concat([splits.X_train, splits.X_validation])
    y_development = pd.concat([splits.y_train, splits.y_validation])
    final_model = search.best_estimator_
    final_model.fit(X_development, y_development)

    test_scores = final_model.predict_proba(splits.X_test)[:, 1]
    test_metrics = calculate_metrics(splits.y_test, test_scores, threshold=threshold)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    validation_comparison.to_csv(output_path / "validation_comparison.csv", index=False)
    joblib.dump(final_model, output_path / "xgboost_smote_pipeline.joblib")
    save_diagnostic_plots(splits.y_test, test_scores, threshold, output_path)

    results = {
        "dataset_path": str(data_path),
        "random_state": random_state,
        "split_rows": {
            "train": int(len(splits.y_train)),
            "validation": int(len(splits.y_validation)),
            "test": int(len(splits.y_test)),
        },
        "cross_validation": {
            "folds": cv_folds,
            "scoring": "f1",
            "best_mean_f1": float(search.best_score_),
            "best_parameters": search.best_params_,
        },
        "threshold_selection": {
            "source": "validation partition",
            "method": "maximum F1 over precision-recall thresholds",
            "threshold": threshold,
            "validation_f1_at_selected_threshold": validation_f1,
        },
        "test_metrics": test_metrics,
        "validation_comparison_file": "validation_comparison.csv",
        "test_evaluation_note": "The test partition was not used for fitting or threshold selection.",
    }
    write_json(results, output_path / "metrics.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("results/run"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=3)
    arguments = parser.parse_args()
    results = run_training(
        data_path=arguments.data_path,
        output_dir=arguments.output_dir,
        random_state=arguments.random_state,
        cv_folds=arguments.cv_folds,
    )
    print(pd.Series(results["test_metrics"]).to_string())


if __name__ == "__main__":
    main()
