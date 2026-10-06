# =========================================================
# MODEL COMPARISON
# =========================================================

import os
import time
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from ETL.Extract import load_data
from ETL.Transform import prepare_data


# =========================================================
# 1. SETTINGS
# =========================================================

DATA_PATH = "data/train-test.csv"
RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# =========================================================
# 2. LOAD + PREPARE DATA
# =========================================================

df = load_data(
    DATA_PATH
)

X_train, X_val, y_train, y_val = prepare_data(
    df
)

print("=" * 60)
print("DATA SPLIT")
print("=" * 60)

print(
    "Train shape:",
    X_train.shape
)

print(
    "Validation shape:",
    X_val.shape
)


# =========================================================
# 3. FEATURE GROUPS
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


# =========================================================
# 4. PREPROCESSOR
# =========================================================

def build_preprocessor():

    return ColumnTransformer(
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
# 5. EVALUATION FUNCTION
# =========================================================

def evaluate_model(
    name,
    y_true,
    y_pred,
    training_time
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(
        f"MAE:  {mae:.2f}"
    )

    print(
        f"RMSE: {rmse:.2f}"
    )

    print(
        f"R²:   {r2:.4f}"
    )

    print(
        f"Training Time: "
        f"{training_time:.2f} sec"
    )

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Training_Time": training_time
    }


# =========================================================
# 6. TRAIN PIPELINE MODEL
# =========================================================

def train_pipeline_model(
    name,
    model
):

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "model",
                model
            )
        ]
    )

    start_time = time.time()

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_val
    )

    training_time = (
        time.time()
        - start_time
    )

    result = evaluate_model(
        name,
        y_val,
        predictions,
        training_time
    )

    return (
        pipeline,
        predictions,
        result
    )


# =========================================================
# 7. STORAGE
# =========================================================

results = []

trained_models = {}

predictions_dict = {}


# =========================================================
# 8. DUMMY BASELINE
# =========================================================

print(
    "\nTraining Dummy Baseline..."
)

baseline_model = DummyRegressor(
    strategy="median"
)

start_time = time.time()

baseline_model.fit(
    np.zeros(
        (len(X_train), 1)
    ),
    y_train
)

baseline_pred = baseline_model.predict(
    np.zeros(
        (len(X_val), 1)
    )
)

baseline_time = (
    time.time()
    - start_time
)

baseline_result = evaluate_model(
    "Dummy Baseline",
    y_val,
    baseline_pred,
    baseline_time
)

results.append(
    baseline_result
)

trained_models[
    "Dummy Baseline"
] = baseline_model

predictions_dict[
    "Dummy Baseline"
] = baseline_pred


# =========================================================
# 9. LINEAR REGRESSION
# =========================================================

print(
    "\nTraining Linear Regression..."
)

linear_model, linear_pred, linear_result = (
    train_pipeline_model(
        "Linear Regression",
        LinearRegression()
    )
)

results.append(
    linear_result
)

trained_models[
    "Linear Regression"
] = linear_model

predictions_dict[
    "Linear Regression"
] = linear_pred


# =========================================================
# 10. RANDOM FOREST
# =========================================================

print(
    "\nTraining Random Forest..."
)

rf_model, rf_pred, rf_result = (
    train_pipeline_model(
        "Random Forest",
        RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
)

results.append(
    rf_result
)

trained_models[
    "Random Forest"
] = rf_model

predictions_dict[
    "Random Forest"
] = rf_pred


# =========================================================
# 11. XGBOOST
# =========================================================

print(
    "\nTraining XGBoost..."
)

xgb_model, xgb_pred, xgb_result = (
    train_pipeline_model(
        "XGBoost",
        XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.90,
            colsample_bytree=0.90,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
    )
)

results.append(
    xgb_result
)

trained_models[
    "XGBoost"
] = xgb_model

predictions_dict[
    "XGBoost"
] = xgb_pred


# =========================================================
# 12. CATBOOST
# =========================================================

print(
    "\nTraining CatBoost..."
)

cat_features = [
    "pickup",
    "delivery",
    "equipment",
    "route"
]

catboost_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=42,
    verbose=0
)

start_time = time.time()

catboost_model.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

catboost_pred = catboost_model.predict(
    X_val
)

catboost_time = (
    time.time()
    - start_time
)

catboost_result = evaluate_model(
    "CatBoost",
    y_val,
    catboost_pred,
    catboost_time
)

results.append(
    catboost_result
)

trained_models[
    "CatBoost"
] = catboost_model

predictions_dict[
    "CatBoost"
] = catboost_pred


# =========================================================
# 13. FINAL COMPARISON
# =========================================================

results_df = pd.DataFrame(
    results
)

results_df = (
    results_df
    .sort_values(
        by="MAE",
        ascending=True
    )
    .reset_index(
        drop=True
    )
)

print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.round(
        {
            "MAE": 2,
            "RMSE": 2,
            "R2": 4,
            "Training_Time": 2
        }
    )
)


# =========================================================
# 14. SELECT BEST MODEL
# =========================================================

best_model_row = (
    results_df.iloc[0]
)

best_model_name = (
    best_model_row["Model"]
)

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"Model: "
    f"{best_model_name}"
)

print(
    f"MAE:  "
    f"{best_model_row['MAE']:.2f}"
)

print(
    f"RMSE: "
    f"{best_model_row['RMSE']:.2f}"
)

print(
    f"R²:   "
    f"{best_model_row['R2']:.4f}"
)


# =========================================================
# 15. SAVE MODEL COMPARISON RESULTS
# =========================================================

comparison_path = os.path.join(
    RESULTS_DIR,
    "model_comparison_results.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

print(
    "\nSaved:"
)

print(
    comparison_path
)


# =========================================================
# 16. SAVE VALIDATION DATA + PREDICTIONS
# =========================================================

# Keep all validation features
validation_results = (
    X_val
    .copy()
    .reset_index(drop=True)
)

# Add actual target
validation_results[
    "actual_rate"
] = (
    y_val
    .reset_index(drop=True)
    .values
)


# Add predictions from every model
for model_name, predictions in predictions_dict.items():

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    validation_results[
        safe_name
    ] = predictions


# Save file
validation_predictions_path = os.path.join(
    RESULTS_DIR,
    "validation_model_predictions.csv"
)

validation_results.to_csv(
    validation_predictions_path,
    index=False
)


print(
    "Saved:"
)

print(
    validation_predictions_path
)


# =========================================================
# 17. VALIDATION FILE CHECK
# =========================================================

print("\n" + "=" * 70)
print("VALIDATION OUTPUT CHECK")
print("=" * 70)

print(
    "Shape:",
    validation_results.shape
)

print(
    "\nColumns:"
)

print(
    validation_results.columns.tolist()
)

print(
    "\nFirst 5 Rows:"
)

print(
    validation_results.head()
)


# =========================================================
# 18. FINAL MESSAGE
# =========================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON COMPLETE")
print("=" * 70)