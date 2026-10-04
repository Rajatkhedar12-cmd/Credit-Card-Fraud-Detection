# Model Artifacts

Trained model files are generated locally in `results/run/` by `python -m src.train` and are excluded from version control. The saved XGBoost pipeline includes preprocessing and SMOTE for consistent training behavior. The selected classification threshold is recorded separately in `results/run/metrics.json`.
