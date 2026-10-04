# Credit Card Fraud Detection – ML Pipeline

An individual machine-learning project for classifying credit-card transactions as legitimate or fraudulent. The emphasis is on minority-class evaluation and keeping preprocessing and oversampling inside training-only workflows.

## Overview

Fraud detection is a binary-classification problem with a highly imbalanced target: fraudulent transactions are a small minority. A model can appear accurate while missing most fraud, so this project compares models using precision, recall, F1-score, ROC-AUC, average precision, and the confusion matrix.

## Problem Statement

Given the anonymized transaction features in the Kaggle dataset, predict whether each transaction is legitimate (`Class = 0`) or fraudulent (`Class = 1`). The intended use here is an educational offline evaluation, not production decisioning.

## Dataset

This project expects the Kaggle [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), including a numeric `Class` target and transaction columns such as `Time`, `Amount`, and `V1` through `V28`.

Download the dataset through Kaggle and place the CSV here:

```text
data/creditcard.csv
```

The dataset is not included in this repository. Please follow Kaggle's terms and do not commit the downloaded data.

## Key Challenges

- Severe class imbalance means an always-legitimate classifier can have deceptively high accuracy.
- Fraud is the minority class, so evaluation must report its precision, recall, and F1-score.
- False negatives represent fraud that was not flagged; false positives can create unnecessary review work.
- The classification threshold changes the precision-recall tradeoff and should be selected without using the test set.

## Approach

```text
Raw transaction data
-> data validation
-> stratified train / validation / test split
-> preprocessing fitted in each training fit
-> SMOTE on training data only
-> baseline and XGBoost models
-> cross-validated hyperparameter tuning
-> validation threshold selection
-> one final test evaluation
```

The split is stratified into 60% training, 20% validation, and 20% test. `Time` and `Amount` are standardized inside an imbalanced-learn pipeline. SMOTE is a pipeline step after preprocessing: during cross-validation it is fitted only on each fold's training partition, and validation and test rows are never resampled. The chosen model is refit on train plus validation only after its threshold has been selected on validation.

No statistical outlier-removal rule is applied. In this dataset, unusual values may carry fraud signal, and the source implementation provides no basis for deleting them. Input validation rejects missing or non-finite values instead of silently changing the dataset.

## Models

- **Baseline:** Logistic regression trained without oversampling.
- **SMOTE comparison:** Logistic regression with training-only SMOTE.
- **Tuned model:** XGBoost classifier with SMOTE, selected by stratified cross-validation using F1-score.

The XGBoost classifier is an ensemble of boosted decision trees. No deep-learning models are used.

## Evaluation Metrics

- **Precision:** Of transactions flagged as fraud, the fraction that are fraud.
- **Recall:** Of actual fraud transactions, the fraction detected.
- **F1-score:** Harmonic mean of precision and recall; used for model selection.
- **ROC-AUC:** Ranking quality across classification thresholds.
- **Average precision:** Summary of the precision-recall curve, useful when positives are rare.
- **Confusion matrix:** Counts of true/false positives and negatives at the selected threshold.

Accuracy is not used as the headline metric because the legitimate class dominates the dataset.

## Results

Results are generated locally by the pipeline and written to `results/run/metrics.json`. No metric values are pre-filled in this repository. The resume-reported figures of approximately 99.4% ROC-AUC and 88% fraud-class F1 are not described here as reproduced results; compare them with the generated holdout metrics after running this project. These are offline holdout results, not production metrics.

The validation comparison uses a fixed 0.5 threshold for consistent comparison. The final XGBoost operating threshold is selected by maximizing F1 on validation data only; the test set is reserved for the final evaluation.

## Why SMOTE?

SMOTE generates synthetic minority-class examples to give a classifier more fraud examples during fitting. It can improve minority-class learning, but it does not guarantee better test performance. In this project, it is applied only to training data. When cross-validation runs, the imbalanced-learn pipeline applies SMOTE separately within each training fold to prevent leakage.

## Project Structure

```text
credit-card-fraud-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── fraud_detection_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── tests/
│   └── test_pipeline.py
├── models/
│   └── README.md
└── results/
    ├── README.md
    └── metrics.md
```

## Installation

Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download `creditcard.csv` from Kaggle and put it in `data/`, then run:

```bash
python -m src.train
```

Optional arguments:

```bash
python -m src.train --data-path data/creditcard.csv --output-dir results/run --random-state 42 --cv-folds 3
```

For the notebook, start Jupyter from the repository root:

```bash
jupyter notebook notebooks/fraud_detection_analysis.ipynb
```

The complete search can take several minutes because SMOTE and cross-validation are run for each parameter combination.

## Tests

```bash
python -m unittest discover -s tests -v
```
