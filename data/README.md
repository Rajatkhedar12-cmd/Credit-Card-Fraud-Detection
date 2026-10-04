# Dataset

Download the Kaggle Credit Card Fraud Detection dataset from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.

Place the downloaded CSV at `data/creditcard.csv`. The data file is intentionally excluded from version control; review Kaggle's dataset terms before use or redistribution.

The loader expects a numeric `Class` target containing both labels `0` (legitimate) and `1` (fraud), plus numeric feature columns. Missing and non-finite values are rejected rather than silently dropped or imputed.
