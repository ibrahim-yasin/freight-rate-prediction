import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

df = load_data(
    DATA_PATH
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# =========================================================
# 2. INVALID WEIGHT
# =========================================================

df.loc[
    df["weight"] <= 0,
    "weight"
] = np.nan


# =========================================================
# 3. TIME SPLIT
# =========================================================

train_df = df[
    df["date"] < "2025-10-01"
].copy()

val_df = df[
    df["date"] >= "2025-10-01"
].copy()


# =========================================================
# 4. TRAIN-ONLY IMPUTATION
# =========================================================

weight_median = (
    train_df["weight"]
    .median()
)

market_median = (
    train_df["market_index"]
    .median()
)


for data in [
    train_df,
    val_df
]:

    data["weight"] = (
        data["weight"]
        .fillna(weight_median)
    )

    data["market_index"] = (
        data["market_index"]
        .fillna(market_median)
    )


# =========================================================
# 5. FEATURE ENGINEERING
# =========================================================

train_df = add_features(
    train_df
)

val_df = add_features(
    val_df
)


# =========================================================
# 6. CREATE HIGH-VALUE TARGET
# =========================================================

train_df["is_high_value"] = (
    train_df["posted_rate"]
    > HIGH_THRESHOLD
).astype(int)

val_df["is_high_value"] = (
    val_df["posted_rate"]
    > HIGH_THRESHOLD
).astype(int)


print("=" * 70)
print("HIGH VALUE TARGET")
print("=" * 70)

print(
    "Train positives:",
    train_df["is_high_value"].sum()
)

print(
    "Validation positives:",
    val_df["is_high_value"].sum()
)


# =========================================================
# 7. FEATURES
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


X_train = (
    train_df[features]
    .copy()
)

y_train = (
    train_df["is_high_value"]
    .copy()
)

X_val = (
    val_df[features]
    .copy()
)

y_val = (
    val_df["is_high_value"]
    .copy()
)


# =========================================================
# 8. PREPROCESSOR
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),

        (
            "num",
            StandardScaler(),
            numeric_features
        )
    ]
)


# =========================================================
# 9. LOGISTIC REGRESSION
# =========================================================

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                solver="lbfgs",
                random_state=42
            )
        )
    ]
)


print("\nTraining Logistic Regression...")

logistic_model.fit(
    X_train,
    y_train
)

logistic_pred = (
    logistic_model.predict(
        X_val
    )
)

logistic_prob = (
    logistic_model.predict_proba(
        X_val
    )[:, 1]
)


print("\n" + "=" * 70)
print("LOGISTIC REGRESSION")
print("=" * 70)

print(
    classification_report(
        y_val,
        logistic_pred,
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
        logistic_pred
    )
)

print(
    "\nPR-AUC:",
    round(
        average_precision_score(
            y_val,
            logistic_prob
        ),
        4
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_val,
            logistic_prob
        ),
        4
    )
)


# =========================================================
# 10. LOGISTIC THRESHOLD ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("LOGISTIC THRESHOLD ANALYSIS")
print("=" * 70)


logistic_thresholds = [
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]


for threshold in logistic_thresholds:

    threshold_pred = (
        logistic_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        threshold_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        threshold_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        threshold_pred,
        zero_division=0
    )

    predicted_high = (
        threshold_pred.sum()
    )

    true_positive = (
        (
            (threshold_pred == 1)
            &
            (y_val.values == 1)
        )
        .sum()
    )

    false_positive = (
        (
            (threshold_pred == 1)
            &
            (y_val.values == 0)
        )
        .sum()
    )

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1: {f1:.4f}"
    )

    print(
        f"Predicted High: {predicted_high}"
    )

    print(
        f"True Positives: {true_positive}"
    )

    print(
        f"False Positives: {false_positive}"
    )


# =========================================================
# 11. BEST LOGISTIC THRESHOLD
# =========================================================

best_logistic_threshold = None
best_logistic_f1 = -1


for threshold in np.arange(
    0.05,
    1.00,
    0.01
):

    threshold_pred = (
        logistic_prob >= threshold
    ).astype(int)

    current_f1 = f1_score(
        y_val,
        threshold_pred,
        zero_division=0
    )

    if current_f1 > best_logistic_f1:

        best_logistic_f1 = current_f1
        best_logistic_threshold = threshold


best_logistic_pred = (
    logistic_prob >= best_logistic_threshold
).astype(int)


print("\n" + "=" * 70)
print("BEST LOGISTIC THRESHOLD")
print("=" * 70)

print(
    "Best Threshold:",
    round(
        best_logistic_threshold,
        2
    )
)

print(
    "Best F1:",
    round(
        best_logistic_f1,
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_val,
            best_logistic_pred,
            zero_division=0
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_val,
            best_logistic_pred,
            zero_division=0
        ),
        4
    )
)

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_val,
        best_logistic_pred
    )
)


# =========================================================
# 12. RANDOM FOREST CLASSIFIER
# =========================================================

rf_model = Pipeline(
    steps=[
        (
            "preprocessor",
            ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        ),
                        categorical_features
                    ),

                    (
                        "num",
                        "passthrough",
                        numeric_features
                    )
                ]
            )
        ),

        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


print("\nTraining Random Forest Classifier...")

rf_model.fit(
    X_train,
    y_train
)

rf_pred = (
    rf_model.predict(
        X_val
    )
)

rf_prob = (
    rf_model.predict_proba(
        X_val
    )[:, 1]
)


print("\n" + "=" * 70)
print("RANDOM FOREST CLASSIFIER")
print("=" * 70)

print(
    classification_report(
        y_val,
        rf_pred,
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
        rf_pred
    )
)

print(
    "\nPR-AUC:",
    round(
        average_precision_score(
            y_val,
            rf_prob
        ),
        4
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_val,
            rf_prob
        ),
        4
    )
)


# =========================================================
# 13. RANDOM FOREST THRESHOLD ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("RANDOM FOREST THRESHOLD ANALYSIS")
print("=" * 70)


rf_thresholds = [
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]


for threshold in rf_thresholds:

    threshold_pred = (
        rf_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        threshold_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        threshold_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        threshold_pred,
        zero_division=0
    )

    predicted_high = (
        threshold_pred.sum()
    )

    true_positive = (
        (
            (threshold_pred == 1)
            &
            (y_val.values == 1)
        )
        .sum()
    )

    false_positive = (
        (
            (threshold_pred == 1)
            &
            (y_val.values == 0)
        )
        .sum()
    )

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1: {f1:.4f}"
    )

    print(
        f"Predicted High: {predicted_high}"
    )

    print(
        f"True Positives: {true_positive}"
    )

    print(
        f"False Positives: {false_positive}"
    )


# =========================================================
# 14. BEST RANDOM FOREST THRESHOLD
# =========================================================

best_rf_threshold = None
best_rf_f1 = -1


for threshold in np.arange(
    0.05,
    1.00,
    0.01
):

    threshold_pred = (
        rf_prob >= threshold
    ).astype(int)

    current_f1 = f1_score(
        y_val,
        threshold_pred,
        zero_division=0
    )

    if current_f1 > best_rf_f1:

        best_rf_f1 = current_f1
        best_rf_threshold = threshold


best_rf_pred = (
    rf_prob >= best_rf_threshold
).astype(int)


print("\n" + "=" * 70)
print("BEST RANDOM FOREST THRESHOLD")
print("=" * 70)

print(
    "Best Threshold:",
    round(
        best_rf_threshold,
        2
    )
)

print(
    "Best F1:",
    round(
        best_rf_f1,
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_val,
            best_rf_pred,
            zero_division=0
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_val,
            best_rf_pred,
            zero_division=0
        ),
        4
    )
)

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_val,
        best_rf_pred
    )
)


# =========================================================
# 15. BASELINE
# =========================================================

positive_rate = (
    y_val.mean()
)

print("\n" + "=" * 70)
print("BASELINE")
print("=" * 70)

print(
    "Positive Rate:",
    round(
        positive_rate,
        4
    )
)

print(
    "Random PR-AUC baseline:",
    round(
        positive_rate,
        4
    )
)