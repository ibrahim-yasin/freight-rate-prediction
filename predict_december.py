# =========================================================
# DECEMBER PREDICTIONS
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
DECEMBER_PATH = "data/december_chart_inputs.csv"

OUTPUT_PATH = "data/december_chart_inputs.csv"


# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

def add_features(data):

    data = data.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek

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

train_df["date"] = pd.to_datetime(
    train_df["date"],
    errors="coerce"
)

# Invalid weight
train_df.loc[
    train_df["weight"] <= 0,
    "weight"
] = np.nan


# =========================================================
# 4. TRAIN MEDIANS
# =========================================================

weight_median = (
    train_df["weight"]
    .median()
)

market_median = (
    train_df["market_index"]
    .median()
)

train_df["weight"] = (
    train_df["weight"]
    .fillna(weight_median)
)

train_df["market_index"] = (
    train_df["market_index"]
    .fillna(market_median)
)


# =========================================================
# 5. FEATURE ENGINEERING - TRAIN
# =========================================================

train_df = add_features(
    train_df
)


# =========================================================
# 6. LOAD DECEMBER DATA
# =========================================================

december_df = pd.read_csv(
    DECEMBER_PATH
)

print("=" * 70)
print("DECEMBER DATA")
print("=" * 70)

print(
    "Rows:",
    len(december_df)
)

print(
    december_df.head()
)


# =========================================================
# 7. CLEAN DECEMBER DATA
# =========================================================

december_df["date"] = pd.to_datetime(
    december_df["date"],
    errors="coerce"
)

december_df.loc[
    december_df["weight"] <= 0,
    "weight"
] = np.nan

december_df["weight"] = (
    december_df["weight"]
    .fillna(weight_median)
)

# December file may not contain market_index.
# If it exists, fill using training median.
if "market_index" in december_df.columns:

    december_df["market_index"] = (
        december_df["market_index"]
        .fillna(market_median)
    )


# =========================================================
# 8. FEATURE ENGINEERING - DECEMBER
# =========================================================

december_features_df = add_features(
    december_df
)


# =========================================================
# 9. FEATURES
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
    "month",
    "day",
    "day_of_week",
    "is_weekend"
]

# Only include these if December file has them
optional_features = []

if "market_index" in december_features_df.columns:
    optional_features.append("market_index")

if "quote_signal" in december_features_df.columns:
    optional_features.append("quote_signal")

numeric_features = (
    numeric_features
    + optional_features
)

features = (
    categorical_features
    + numeric_features
)


# =========================================================
# 10. IMPORTANT:
# TRAIN AND DECEMBER MUST USE SAME FEATURES
# =========================================================

missing_train_features = [
    feature
    for feature in features
    if feature not in train_df.columns
]

if missing_train_features:

    raise ValueError(
        "Missing training features: "
        + str(missing_train_features)
    )


X_train = (
    train_df[features]
    .copy()
)

y_train = (
    train_df["posted_rate"]
    .copy()
)

X_december = (
    december_features_df[features]
    .copy()
)


# =========================================================
# 11. PREPROCESSOR
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
# 12. FINAL MODEL
# =========================================================

model = Pipeline(
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
# 13. TRAIN
# =========================================================

print(
    "\nTraining final Linear Regression..."
)

model.fit(
    X_train,
    y_train
)

print(
    "Training complete."
)


# =========================================================
# 14. PREDICT DECEMBER
# =========================================================

december_predictions = model.predict(
    X_december
)

december_predictions = np.maximum(
    december_predictions,
    0.01
)


# =========================================================
# 15. WRITE PREDICTIONS
# =========================================================

december_df["predicted_rate"] = (
    december_predictions
)


# Keep required original columns only
required_columns = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate"
]

december_output = (
    december_df[required_columns]
    .copy()
)


# =========================================================
# 16. SAVE
# =========================================================

december_output.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# 17. CHECK OUTPUT
# =========================================================

print("\n" + "=" * 70)
print("DECEMBER OUTPUT CHECK")
print("=" * 70)

print(
    "Rows:",
    len(december_output)
)

print(
    "Columns:",
    december_output.columns.tolist()
)

print(
    "\nPrediction Statistics:"
)

print(
    december_output["predicted_rate"]
    .describe()
)

print(
    "\nFirst 5 Rows:"
)

print(
    december_output.head()
)

print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)