# Metrics Reporting

Run `python -m src.train` with the Kaggle CSV at `data/creditcard.csv` to generate `results/run/metrics.json` and `results/run/validation_comparison.csv`.

The JSON contains the selected threshold, fraud-class precision/recall/F1, ROC-AUC, average precision, confusion matrix, cross-validation score, and split sizes. The validation comparison reports the baseline, SMOTE baseline, and tuned XGBoost at a common 0.5 threshold. The final test metrics use the threshold selected on validation.

No result values are recorded here because the dataset has not been evaluated in this environment. Do not report the resume figures as reproduced until they match a generated holdout run.
