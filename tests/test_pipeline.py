"""Small synthetic checks for split and SMOTE pipeline behavior."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.preprocessing import split_dataset, validate_dataset
from src.train import build_model_pipeline, run_training


class DataPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(7)
        self.data = pd.DataFrame(
            {
                "Time": np.arange(200, dtype=float),
                "Amount": generator.uniform(1, 500, size=200),
                "V1": generator.normal(size=200),
                "Class": np.array([0] * 180 + [1] * 20),
            }
        )

    def test_split_is_stratified_and_disjoint(self) -> None:
        splits = split_dataset(self.data)
        self.assertEqual(len(splits.X_train), 120)
        self.assertEqual(len(splits.X_validation), 40)
        self.assertEqual(len(splits.X_test), 40)
        self.assertEqual(splits.y_train.sum(), 12)
        self.assertEqual(splits.y_validation.sum(), 4)
        self.assertEqual(splits.y_test.sum(), 4)
        self.assertTrue(set(splits.X_train.index).isdisjoint(splits.X_validation.index))
        self.assertTrue(set(splits.X_train.index).isdisjoint(splits.X_test.index))
        self.assertTrue(set(splits.X_validation.index).isdisjoint(splits.X_test.index))

    def test_validation_rejects_infinite_features(self) -> None:
        invalid = self.data.copy()
        invalid.loc[0, "Amount"] = np.inf
        with self.assertRaisesRegex(ValueError, "infinite"):
            validate_dataset(invalid)

    def test_smote_pipeline_fits_only_when_training(self) -> None:
        splits = split_dataset(self.data)
        pipeline = build_model_pipeline(
            LogisticRegression(max_iter=500),
            feature_names=list(splits.X_train.columns),
            use_smote=True,
            random_state=7,
        )
        pipeline.fit(splits.X_train, splits.y_train)
        self.assertIn("smote", pipeline.named_steps)
        self.assertEqual(len(pipeline.predict_proba(splits.X_test)), len(splits.y_test))
        self.assertEqual(int(splits.y_test.sum()), 4)

    def test_training_writes_test_metrics_and_artifacts(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            data_path = temporary_path / "transactions.csv"
            output_path = temporary_path / "results"
            self.data.to_csv(data_path, index=False)

            results = run_training(
                data_path=data_path,
                output_dir=output_path,
                random_state=7,
                cv_folds=2,
            )

            self.assertIn("test_metrics", results)
            self.assertEqual(results["split_rows"]["test"], 40)
            self.assertTrue((output_path / "metrics.json").is_file())
            self.assertTrue((output_path / "validation_comparison.csv").is_file())
            self.assertTrue((output_path / "xgboost_smote_pipeline.joblib").is_file())
            self.assertTrue((output_path / "confusion_matrix.png").is_file())
            self.assertTrue((output_path / "roc_precision_recall.png").is_file())


if __name__ == "__main__":
    unittest.main()
