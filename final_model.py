# =========================================================
# FINAL TRAINING + VALIDATION PREDICTIONS
# =========================================================

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ETL.Extract import load_data


# =========================================================
# 1. SETTINGS
# =========================================================

TRAIN_PATH = "data/train-test.csv"
VALIDATION_PATH = "data/validation.csv"

OUTPUT_PATH = "validation_predictions.csv"


# =========================================================
# 2. FEATURE ENGINEERING FUNCTION
# =========================================================

def add_features(data):

    data = data.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data["month"] = data["date"].dt.month

    data["day"] = data["date"].dt.day

    data["day_of_week"] = (
        data["date"].dt.dayofweek
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["route"] = (
        data["pickup"].astype(str)
        + "_TO_"
        + data["delivery"].astype(str)
    )

    return data


# =========================================================
# 3. LOAD TRAIN DATA
# =========================================================

train_df = load_data(
    TRAIN_PATH
)

print("=" * 70)
print("FINAL MODEL TRAINING")
print("=" * 70)

print(
    "Training rows:",
    len(train_df)
)


# =========================================================
# 4. CLEAN TRAIN DATA
# =========================================================

train_df["date"] = pd.to_datetime(
    train_df["date"],
    errors="coerce"
)

train_df.loc[
    train_df["weight"] <= 0,
    "weight"
] = np.nan


# =========================================================
# 5. TRAIN IMPUTATION VALUES
# =========================================================

weight_median = (
    train_df["weight"]
    .median()
)

market_median = (
    train_df["market_index"]
    .median()
)

print(
    "Weight median:",
    weight_median
)

print(
    "Market index median:",
    market_median
)


# Fill training missing values

train_df["weight"] = (
    train_df["weight"]
    .fillna(weight_median)
)

train_df["market_index"] = (
    train_df["market_index"]
    .fillna(market_median)
)


# =========================================================
# 6. TRAIN FEATURE ENGINEERING
# =========================================================

train_df = add_features(
    train_df
)


# =========================================================
# 7. LOAD VALIDATION DATA
# =========================================================

validation_df = load_data(
    VALIDATION_PATH
)

print(
    "Validation rows:",
    len(validation_df)
)


# Keep IDs before transformation
validation_ids = (
    validation_df["load_id"]
    .copy()
)


# =========================================================
# 8. CLEAN VALIDATION DATA
# =========================================================

validation_df["date"] = pd.to_datetime(
    validation_df["date"],
    errors="coerce"
)

validation_df.loc[
    validation_df["weight"] <= 0,
    "weight"
] = np.nan


# IMPORTANT:
# Use medians calculated from TRAINING data only

validation_df["weight"] = (
    validation_df["weight"]
    .fillna(weight_median)
)

validation_df["market_index"] = (
    validation_df["market_index"]
    .fillna(market_median)
)


# =========================================================
# 9. VALIDATION FEATURE ENGINEERING
# =========================================================

validation_df = add_features(
    validation_df
)


# =========================================================
# 10. DEFINE FEATURES
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

target = "posted_rate"


# =========================================================
# 11. CREATE TRAIN / VALIDATION MATRICES
# =========================================================

X_train = (
    train_df[features]
    .copy()
)

y_train = (
    train_df[target]
    .copy()
)

X_validation = (
    validation_df[features]
    .copy()
)


# =========================================================
# 12. PREPROCESSOR
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),

        (
            "numerical",
            "passthrough",
            numeric_features
        )
    ]
)


# =========================================================
# 13. FINAL LINEAR REGRESSION MODEL
# =========================================================

final_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            LinearRegression()
        )
    ]
)


# =========================================================
# 14. TRAIN FINAL MODEL
# =========================================================

print("\nTraining final Linear Regression...")

final_model.fit(
    X_train,
    y_train
)

print("Training complete.")


# =========================================================
# 15. PREDICT VALIDATION DATA
# =========================================================

predictions = final_model.predict(
    X_validation
)


# Ensure predictions are positive
predictions = np.maximum(
    predictions,
    0.01
)


# =========================================================
# 16. CREATE SUBMISSION FILE
# =========================================================

submission = pd.DataFrame({
    "load_id": validation_ids,
    "predicted_rate": predictions
})


# =========================================================
# 17. VALIDATE OUTPUT
# =========================================================

print("\n" + "=" * 70)
print("OUTPUT CHECK")
print("=" * 70)

print(
    "Rows:",
    len(submission)
)

print(
    "Columns:",
    submission.columns.tolist()
)

print(
    "Missing predictions:",
    submission["predicted_rate"]
    .isnull()
    .sum()
)

print(
    "Non-positive predictions:",
    (
        submission["predicted_rate"] <= 0
    ).sum()
)

print(
    "\nPrediction statistics:"
)

print(
    submission["predicted_rate"]
    .describe()
)

print(
    "\nFirst 5 predictions:"
)

print(
    submission.head()
)


# =========================================================
# 18. SAVE FILE
# =========================================================

submission.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)