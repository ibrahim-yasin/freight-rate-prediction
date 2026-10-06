import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error


# =========================================================
# 1. SETTINGS
# =========================================================

RESULTS_DIR = "results"

PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR,
    "validation_model_predictions.csv"
)


# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_csv(
    PREDICTIONS_PATH
)

print("=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# =========================================================
# 3. MODELS
# =========================================================

model_columns = [
    "dummy_baseline",
    "linear_regression",
    "random_forest",
    "xgboost",
    "catboost"
]


# =========================================================
# 4. CREATE ERROR COLUMNS
# =========================================================

for model in model_columns:

    # Residual:
    # positive = model predicted too low
    # negative = model predicted too high
    df[f"{model}_error"] = (
        df["actual_rate"]
        - df[model]
    )

    df[f"{model}_abs_error"] = (
        df[f"{model}_error"]
        .abs()
    )


# =========================================================
# 5. OVERALL ERROR SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("OVERALL ERROR SUMMARY")
print("=" * 70)

summary = []

for model in model_columns:

    summary.append({
        "Model": model,

        "MAE": df[
            f"{model}_abs_error"
        ].mean(),

        "Median_Error": df[
            f"{model}_abs_error"
        ].median(),

        "Max_Error": df[
            f"{model}_abs_error"
        ].max()
    })


summary_df = pd.DataFrame(
    summary
)

summary_df = (
    summary_df
    .sort_values("MAE")
    .reset_index(drop=True)
)

print(
    summary_df.round(2)
)


# =========================================================
# 6. DISTANCE BANDS
# =========================================================

distance_bins = [
    0,
    250,
    500,
    1000,
    1500,
    2000,
    np.inf
]

distance_labels = [
    "0-250",
    "250-500",
    "500-1000",
    "1000-1500",
    "1500-2000",
    "2000+"
]

df["distance_range"] = pd.cut(
    df["distance"],
    bins=distance_bins,
    labels=distance_labels,
    include_lowest=True
)


# =========================================================
# 7. MAE BY DISTANCE
# =========================================================

print("\n" + "=" * 70)
print("MAE BY DISTANCE RANGE")
print("=" * 70)

distance_results = []

for model in model_columns:

    grouped = (
        df.groupby(
            "distance_range",
            observed=False
        )[f"{model}_abs_error"]
        .mean()
    )

    for distance_range, mae in grouped.items():

        distance_results.append({
            "Model": model,
            "Distance_Range": distance_range,
            "MAE": mae
        })


distance_df = pd.DataFrame(
    distance_results
)

print(
    distance_df.round(2)
)


# =========================================================
# 8. MAE BY EQUIPMENT
# =========================================================

print("\n" + "=" * 70)
print("MAE BY EQUIPMENT")
print("=" * 70)

equipment_results = []

for model in model_columns:

    grouped = (
        df.groupby(
            "equipment"
        )[f"{model}_abs_error"]
        .mean()
    )

    for equipment, mae in grouped.items():

        equipment_results.append({
            "Model": model,
            "Equipment": equipment,
            "MAE": mae
        })


equipment_df = pd.DataFrame(
    equipment_results
)

print(
    equipment_df.round(2)
)


# =========================================================
# 9. MAE BY RATE RANGE
# =========================================================

rate_bins = [
    0,
    1000,
    2000,
    3000,
    5000,
    10000,
    np.inf
]

rate_labels = [
    "0-1000",
    "1000-2000",
    "2000-3000",
    "3000-5000",
    "5000-10000",
    "10000+"
]

df["rate_range"] = pd.cut(
    df["actual_rate"],
    bins=rate_bins,
    labels=rate_labels,
    include_lowest=True
)


print("\n" + "=" * 70)
print("MAE BY ACTUAL RATE RANGE")
print("=" * 70)

rate_results = []

for model in model_columns:

    grouped = (
        df.groupby(
            "rate_range",
            observed=False
        )[f"{model}_abs_error"]
        .mean()
    )

    for rate_range, mae in grouped.items():

        rate_results.append({
            "Model": model,
            "Rate_Range": rate_range,
            "MAE": mae
        })


rate_df = pd.DataFrame(
    rate_results
)

print(
    rate_df.round(2)
)


# =========================================================
# 10. LINEAR VS CATBOOST
# =========================================================

print("\n" + "=" * 70)
print("LINEAR REGRESSION VS CATBOOST")
print("=" * 70)

linear_error = df[
    "linear_regression_abs_error"
]

catboost_error = df[
    "catboost_abs_error"
]

linear_better = (
    linear_error
    <
    catboost_error
).sum()

catboost_better = (
    catboost_error
    <
    linear_error
).sum()

ties = (
    linear_error
    ==
    catboost_error
).sum()


print(
    "Linear Regression better:",
    linear_better
)

print(
    "CatBoost better:",
    catboost_better
)

print(
    "Ties:",
    ties
)

print(
    "\nLinear better %:",
    round(
        linear_better
        / len(df)
        * 100,
        2
    )
)

print(
    "CatBoost better %:",
    round(
        catboost_better
        / len(df)
        * 100,
        2
    )
)


# =========================================================
# 11. WORST LINEAR REGRESSION ERRORS
# =========================================================

print("\n" + "=" * 70)
print("WORST LINEAR REGRESSION ERRORS")
print("=" * 70)

worst_linear = (
    df[
        [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "actual_rate",
            "linear_regression",
            "linear_regression_error",
            "linear_regression_abs_error"
        ]
    ]
    .sort_values(
        "linear_regression_abs_error",
        ascending=False
    )
    .head(20)
)

print(
    worst_linear.round(2)
)


# =========================================================
# 12. WORST CATBOOST ERRORS
# =========================================================

print("\n" + "=" * 70)
print("WORST CATBOOST ERRORS")
print("=" * 70)

worst_catboost = (
    df[
        [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "actual_rate",
            "catboost",
            "catboost_error",
            "catboost_abs_error"
        ]
    ]
    .sort_values(
        "catboost_abs_error",
        ascending=False
    )
    .head(20)
)

print(
    worst_catboost.round(2)
)


# =========================================================
# 13. ACTUAL VS PREDICTED - LINEAR
# =========================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    df["actual_rate"],
    df["linear_regression"],
    alpha=0.3,
    s=10
)

minimum = min(
    df["actual_rate"].min(),
    df["linear_regression"].min()
)

maximum = max(
    df["actual_rate"].max(),
    df["linear_regression"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.title(
    "Actual vs Predicted - Linear Regression"
)

plt.xlabel(
    "Actual Rate"
)

plt.ylabel(
    "Predicted Rate"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "actual_vs_predicted_linear.png"
    ),
    dpi=150
)

plt.show()


# =========================================================
# 14. RESIDUAL PLOT - LINEAR
# =========================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    df["linear_regression"],
    df["linear_regression_error"],
    alpha=0.3,
    s=10
)

plt.axhline(
    y=0
)

plt.title(
    "Residual Plot - Linear Regression"
)

plt.xlabel(
    "Predicted Rate"
)

plt.ylabel(
    "Actual - Predicted"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "residual_linear.png"
    ),
    dpi=150
)

plt.show()


# =========================================================
# 15. ACTUAL VS PREDICTED - CATBOOST
# =========================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    df["actual_rate"],
    df["catboost"],
    alpha=0.3,
    s=10
)

minimum = min(
    df["actual_rate"].min(),
    df["catboost"].min()
)

maximum = max(
    df["actual_rate"].max(),
    df["catboost"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.title(
    "Actual vs Predicted - CatBoost"
)

plt.xlabel(
    "Actual Rate"
)

plt.ylabel(
    "Predicted Rate"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "actual_vs_predicted_catboost.png"
    ),
    dpi=150
)

plt.show()


# =========================================================
# 16. SAVE RESULTS
# =========================================================

summary_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "error_summary.csv"
    ),
    index=False
)

distance_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "error_by_distance.csv"
    ),
    index=False
)

equipment_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "error_by_equipment.csv"
    ),
    index=False
)

rate_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "error_by_rate.csv"
    ),
    index=False
)

worst_linear.to_csv(
    os.path.join(
        RESULTS_DIR,
        "worst_linear_errors.csv"
    ),
    index=False
)

worst_catboost.to_csv(
    os.path.join(
        RESULTS_DIR,
        "worst_catboost_errors.csv"
    ),
    index=False
)


# =========================================================
# 17. FINAL MESSAGE
# =========================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nFiles saved in results/"
)