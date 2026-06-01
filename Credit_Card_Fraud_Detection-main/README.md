# Credit Card Fraud Detection System

## Overview

This project builds a machine learning pipeline to detect fraudulent credit card transactions. Using a Kaggle dataset of 339,607 transaction records, the pipeline handles the main workflow from raw data to trained model comparison. This includes data cleaning, feature engineering, encoding/scaling, optional SMOTE class-balancing, model training, threshold tuning, evaluation, and plot generation.

The dataset is heavily imbalanced (only 0.52% of transactions are fraudulent), making this a realistic and challenging classification problem. The pipeline is modular so new models and preprocessing steps can be added without changing existing code.

---

## Project Structure

```
project-root/
├── data/
│   └── raw/                        # Original dataset (do NOT modify)
├── notebooks/                      # Jupyter notebooks for EDA and experiments
├── results/
│   ├── plots/                      # Visual outputs: confusion matrices, ROC curves, threshold curves
│   └── metrics/                    # Model comparison CSVs and feature importance CSVs
├── src/
│   ├── config.py                   # Central config: column names, toggles, constants
│   ├── preprocessing/
│   │   ├── preprocessor.py         # Orchestrates cleaning, feature engineering, and validation
│   │   ├── cleaning.py             # Column drops, datetime casting, validation
│   │   ├── feature_engineering.py  # Time features, age, amt_log, haversine distance
│   │   └── encoder.py              # One-hot encoding, RobustScaler, StandardScaler
│   ├── models/
│   │   ├── base.py                 # FraudClassifier wrapper for fit/predict/predict_proba logging
│   │   ├── logistic.py             # Logistic regression model builder
│   │   ├── decision_tree.py        # Decision tree model builder
│   │   ├── random_forest.py        # Random forest model builder
│   │   └── xgboost_model.py        # XGBoost model builder
│   └── evaluation/
│       ├── evaluator.py            # Evaluation metrics, CSV saving, threshold tuning
│       └── plots.py                # Confusion matrix, comparison, threshold, feature importance, ROC plots
├── main.py                         # Entry point, runs the full pipeline end-to-end
└── requirements.txt                # Python dependencies
```

---

## Notebooks

The notebooks folder contains Jupyter notebooks used for exploration and experimentation. These are not part of the runnable pipeline; final reusable logic is kept in `src`.

 Step | Notebook                      | Purpose 
------|-------------------------------|---------
 1    | `data_ex.ipynb`               | First look at the raw dataset — structure, column types, sanity checks 
 2    | `credit_card_fraud_eda.ipynb` | Initial EDA — distributions, missing values, fraud rate breakdowns 
 3    | `EDA.ipynb`                   | Deeper exploratory analysis building on the initial EDA 
 4    | `FeatureImportance.ipynb`     | Explores feature importance scores from tree-based models
 5    | `ModelComparison.ipynb`       | Side-by-side comparison of multiple models on the same test set 

---

## Feature Engineering

Features were chosen based on EDA findings then validated by model importance scores.

### Features Used in Modeling
| Feature | Source |
|---------|--------|
| `amt` | Raw transaction amount |
| `category` | Raw category, one-hot encoded |
| `state` | Raw state, one-hot encoded |
| `city_pop` | Raw city population |
| `lat`, `long` | Raw customer coordinates, currently kept |
| `merch_lat`, `merch_long` | Raw merchant coordinates, currently kept |
| `trans_hour` | Engineered from transaction timestamp |
| `trans_day_of_week` | Engineered from transaction timestamp |
| `trans_month` | Engineered from transaction timestamp |
| `is_weekend` | Engineered from transaction timestamp |
| `is_night` | Engineered, 1 if transaction happened from 10pm through 6am |
| `age` | Engineered from transaction date minus date of birth |
| `amt_log` | Engineered log transform of transaction amount |
| `distance_km` | Engineered haversine distance between customer and merchant coordinates |

### Features Excluded and Why

| Feature | Reason |
|---------|--------|
| `merchant`, `job` | High-cardinality categorical columns. Target encoding was considered but left out to keep the pipeline simpler and avoid leakage risk. |
| `city` | Dropped because location information is already represented by coordinates, state, and city population. |
| `trans_num` | Unique transaction ID with no predictive signal. |
| `trans_date_trans_time` | Replaced by engineered time features. |
| `dob` | Replaced by engineered `age`. |

Note: raw coordinate columns are currently kept because `DROP_COORDS_AFTER_DISTANCE = False` in `config.py`. The pipeline still creates `distance_km`, but tree-based models may also use the original coordinate features if they help.

---

## How the Pipeline Works

```
Raw CSV
  └─► pandas.read_csv                                 loads data/raw/credit_card_frauds.csv
        └─► Preprocessor                              clean → engineer features → validate
              └─► Drop high-cardinality columns       merchant, job
                    └─► train/validation/test split   stratified 60/20/20 split
                          └─► Encoder                 OHE + scaling, fit on train only
                                └─► SMOTE             optional, training data only
                                      └─► Models      train logistic, tree, forest, XGBoost
                                            └─► Evaluate
                                                ├─ baseline test results at threshold 0.50
                                                ├─ validation-based threshold tuning
                                                ├─ tuned test results
                                                └─ plots and CSV outputs
```

> **Why is encoding done after the split?**
> Fitting the encoder before splitting would cause data leakage — the test set would
> influence the scaler's mean and variance. The Encoder is always fit_transform'd on
> training data and only transform'd on validation and test sets.

> **Why a validation set?**
> Threshold tuning is performed on the validation set so the test set stays completely
> untouched until final evaluation. This gives an honest measure of real-world performance.

---

## Models

 Model               | Type              | Strength 
---------------------|-------------------|----------
 Logistic Regression | Baseline          | Fast, easy to follow, good probability estimates 
 Decision Tree       | Rule-Based        | Clear decision paths, easy to explain 
 Random Forest       | Ensemble          | Reduces overfitting by averaging 100 independent trees 
 XGBoost             | Gradient Boosting | Best performer — learns from previous mistakes iteratively 

---

## Evaluation

The pipeline compares models in two ways:

1. **Baseline comparison** - evaluates each model on the final test set using the default `0.50` threshold.
2. **Tuned comparison** - tunes thresholds on the validation set, then applies the selected threshold to the final test set.

The tuned threshold is selected using this weighted score:

```
score = 0.7 * recall + 0.3 * precision
```

This keeps recall as the main priority while still penalizing models that create too many false positives.

Metrics saved for each model include:
- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- false positives
- false negatives
- true positives
- true negatives
- selected threshold

Generated metric files include:

```
results/metrics/baseline_model_comparison.csv
results/metrics/tuned_model_comparison.csv
results/metrics/<model>_feature_importance.csv
```

Generated plot files include:

```
results/plots/baseline_model_comparison.png
results/plots/tuned_model_comparison.png
results/plots/roc_curve_comparison.png
results/plots/<model>_confusion_matrix.png
results/plots/<model>_threshold_curve.png
results/plots/<model>_feature_importance.png
```

---

## Key Findings

- Accuracy alone is misleading because the dataset is highly imbalanced. A model could predict nearly every transaction as legitimate and still look accurate while missing fraud.
- Recall matters heavily because false negatives mean fraudulent transactions were missed.
- Precision still matters because too many false positives would incorrectly flag legitimate transactions.
- Engineered features such as `is_night`, `amt_log`, `age`, and `distance_km` give the models more useful information than the raw columns alone.
- Tree-based models can produce feature importance outputs, which are saved and plotted automatically.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.iu.edu/cothomp/Credit_Card_Fraud_Detection.git
cd Credit_Card_Fraud_Detection
```

### 2. Create a virtual environment

**Mac/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**
```bash
python -m venv venv
.\venv\Scripts\Activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download from Kaggle:

https://www.kaggle.com/datasets/dhruvb2028/credit-card-fraud-dataset

Place the downloaded file at this exact path in your project:

```
data/raw/credit_card_frauds.csv
```

### 5. Run the pipeline

```bash
python main.py
```

Progress will be printed at each stage: loading, preprocessing, splitting, encoding, optional SMOTE resampling, model training, threshold tuning, evaluation, and plot generation.
- Results (plots, metrics) are saved under `results/` and are not committed to the repo
