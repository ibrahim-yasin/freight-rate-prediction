# Analysis/high_value_catboost.py

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from ETL.Extract import load_data
from ETL.Transform import add_features


# =========================================================
# SETTINGS
# =========================================================

DATA_PATH = "data/train-test.csv"
HIGH_THRESHOLD = 7000


# =========================================================
# 1. LOAD DATA
# =========================================================

df = load_data(DATA_PATH)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df.loc[
    df["weight"] <= 0,
    "weight"
] = np.nan


# =========================================================
# 2. TIME SPLIT
# =========================================================

train_df = df[
    df["date"] < "2025-10-01"
].copy()

val_df = df[
    df["date"] >= "2025-10-01"
].copy()


# =========================================================
# 3. TRAIN-ONLY IMPUTATION
# =========================================================

weight_median = train_df["weight"].median()
market_median = train_df["market_index"].median()

for data in [train_df, val_df]:

    data["weight"] = (
        data["weight"]
        .fillna(weight_median)
    )

    data["market_index"] = (
        data["market_index"]
        .fillna(market_median)
    )


# =========================================================
# 4. FEATURE ENGINEERING
# =========================================================

train_df = add_features(train_df)
val_df = add_features(val_df)


# =========================================================
# 5. TARGET
# =========================================================

train_df["is_high_value"] = (
    train_df["posted_rate"] > HIGH_THRESHOLD
).astype(int)

val_df["is_high_value"] = (
    val_df["posted_rate"] > HIGH_THRESHOLD
).astype(int)


# =========================================================
# 6. FEATURES
# =========================================================

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
    "route"
]

numeric_features = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "month",
    "day",
    "day_of_week",
    "is_weekend"
]

features = (
    categorical_features
    + numeric_features
)

X_train = train_df[features].copy()
y_train = train_df["is_high_value"].copy()

X_val = val_df[features].copy()
y_val = val_df["is_high_value"].copy()


# =========================================================
# 7. CATBOOST CATEGORICAL CLEANUP
# =========================================================

for col in categorical_features:

    X_train[col] = (
        X_train[col]
        .fillna("Unknown")
        .astype(str)
    )

    X_val[col] = (
        X_val[col]
        .fillna("Unknown")
        .astype(str)
    )


cat_indices = [
    X_train.columns.get_loc(col)
    for col in categorical_features
]


# =========================================================
# 8. CLASS WEIGHT
# =========================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

positive_weight = (
    negative_count
    / positive_count
)

print("=" * 70)
print("TRAINING INFO")
print("=" * 70)

print("Train positives:", positive_count)
print("Train negatives:", negative_count)

print(
    "Positive class weight:",
    round(positive_weight, 2)
)


# =========================================================
# 9. MODEL
# =========================================================

model = CatBoostClassifier(
    iterations=800,
    depth=7,
    learning_rate=0.03,

    loss_function="Logloss",
    eval_metric="PRAUC",

    class_weights=[
        1.0,
        positive_weight
    ],

    random_seed=42,
    verbose=100
)


print("\nTraining CatBoost Classifier...")


model.fit(
    X_train,
    y_train,

    cat_features=cat_indices,

    eval_set=(
        X_val,
        y_val
    ),

    early_stopping_rounds=100
)


# =========================================================
# 10. PROBABILITIES
# =========================================================

prob = (
    model.predict_proba(
        X_val
    )[:, 1]
)


# =========================================================
# 11. BASE METRICS
# =========================================================

pred = (
    prob >= 0.50
).astype(int)


print("\n" + "=" * 70)
print("CATBOOST @ 0.50")
print("=" * 70)

print(
    classification_report(
        y_val,
        pred,
        digits=4,
        zero_division=0
    )
)

print(
    "Confusion Matrix:"
)

print(
    confusion_matrix(
        y_val,
        pred
    )
)

print(
    "\nPR-AUC:",
    round(
        average_precision_score(
            y_val,
            prob
        ),
        4
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_val,
            prob
        ),
        4
    )
)


# =========================================================
# 12. THRESHOLD SEARCH
# =========================================================

print("\n" + "=" * 70)
print("THRESHOLD SEARCH")
print("=" * 70)


best_threshold = None
best_f1 = -1


for threshold in np.arange(
    0.05,
    0.96,
    0.01
):

    threshold_pred = (
        prob >= threshold
    ).astype(int)

    current_f1 = f1_score(
        y_val,
        threshold_pred,
        zero_division=0
    )

    if current_f1 > best_f1:

        best_f1 = current_f1
        best_threshold = threshold


best_pred = (
    prob >= best_threshold
).astype(int)


precision = precision_score(
    y_val,
    best_pred,
    zero_division=0
)

recall = recall_score(
    y_val,
    best_pred,
    zero_division=0
)


print(
    "Best Threshold:",
    round(
        best_threshold,
        2
    )
)

print(
    "Precision:",
    round(
        precision,
        4
    )
)

print(
    "Recall:",
    round(
        recall,
        4
    )
)

print(
    "F1:",
    round(
        best_f1,
        4
    )
)

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_val,
        best_pred
    )
)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_val,
        best_pred,
        digits=4,
        zero_division=0
    )
)