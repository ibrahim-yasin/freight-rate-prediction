import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# SETTINGS
# =========================================================

RESULTS_DIR = "results"

PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR,
    "validation_model_predictions.csv"
)

HIGH_RATE_THRESHOLD = 7000


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv(
    PREDICTIONS_PATH
)

print("=" * 70)
print("HIGH VALUE LOAD ANALYSIS")
print("=" * 70)

print("Total validation rows:", len(df))


# =========================================================
# 2. FILTER HIGH-VALUE LOADS
# =========================================================

high_df = df[
    df["actual_rate"] > HIGH_RATE_THRESHOLD
].copy()

normal_df = df[
    df["actual_rate"] <= HIGH_RATE_THRESHOLD
].copy()


print("\nHigh-value threshold:")
print(HIGH_RATE_THRESHOLD)

print("\nHigh-value rows:")
print(len(high_df))

print(
    "High-value percentage:",
    round(
        len(high_df) / len(df) * 100,
        2
    ),
    "%"
)


# =========================================================
# 3. BASIC STATISTICS
# =========================================================

print("\n" + "=" * 70)
print("HIGH VALUE BASIC STATISTICS")
print("=" * 70)

columns = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "actual_rate"
]

print(
    high_df[columns]
    .describe()
    .T
)


# =========================================================
# 4. NORMAL VS HIGH COMPARISON
# =========================================================

print("\n" + "=" * 70)
print("NORMAL VS HIGH VALUE COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    "Normal_Mean": normal_df[
        [
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "actual_rate"
        ]
    ].mean(),

    "High_Value_Mean": high_df[
        [
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "actual_rate"
        ]
    ].mean()
})

comparison["Difference"] = (
    comparison["High_Value_Mean"]
    -
    comparison["Normal_Mean"]
)

print(
    comparison.round(2)
)


# =========================================================
# 5. EQUIPMENT DISTRIBUTION
# =========================================================

print("\n" + "=" * 70)
print("HIGH VALUE BY EQUIPMENT")
print("=" * 70)

equipment_counts = (
    high_df["equipment"]
    .value_counts()
)

equipment_percent = (
    high_df["equipment"]
    .value_counts(
        normalize=True
    )
    * 100
)

equipment_summary = pd.DataFrame({
    "Count": equipment_counts,
    "Percentage": equipment_percent.round(2)
})

print(
    equipment_summary
)


# =========================================================
# 6. TOP HIGH-VALUE ROUTES
# =========================================================

print("\n" + "=" * 70)
print("TOP HIGH VALUE ROUTES")
print("=" * 70)

route_summary = (
    high_df.groupby("route")
    .agg(
        count=("actual_rate", "size"),
        mean_rate=("actual_rate", "mean"),
        median_rate=("actual_rate", "median"),
        mean_distance=("distance", "mean")
    )
    .sort_values(
        ["count", "mean_rate"],
        ascending=[False, False]
    )
)

print(
    route_summary
    .head(20)
    .round(2)
)


# =========================================================
# 7. HIGH VALUE BY MONTH
# =========================================================

print("\n" + "=" * 70)
print("HIGH VALUE BY MONTH")
print("=" * 70)

print(
    high_df["month"]
    .value_counts()
    .sort_index()
)


# =========================================================
# 8. HIGH VALUE BY DAY OF WEEK
# =========================================================

print("\n" + "=" * 70)
print("HIGH VALUE BY DAY OF WEEK")
print("=" * 70)

print(
    high_df["day_of_week"]
    .value_counts()
    .sort_index()
)


# =========================================================
# 9. MODEL PERFORMANCE ON HIGH VALUE LOADS
# =========================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE ON HIGH VALUE LOADS")
print("=" * 70)

models = [
    "linear_regression",
    "random_forest",
    "xgboost",
    "catboost"
]

performance_rows = []

for model in models:

    abs_error = (
        high_df["actual_rate"]
        - high_df[model]
    ).abs()

    performance_rows.append({
        "Model": model,
        "MAE": abs_error.mean(),
        "Median_Error": abs_error.median(),
        "Max_Error": abs_error.max()
    })


performance_df = pd.DataFrame(
    performance_rows
).sort_values(
    "MAE"
)

print(
    performance_df.round(2)
)


# =========================================================
# 10. UNDERPREDICTION CHECK
# =========================================================

print("\n" + "=" * 70)
print("UNDERPREDICTION CHECK")
print("=" * 70)

for model in models:

    underpredicted = (
        high_df[model]
        <
        high_df["actual_rate"]
    ).sum()

    percentage = (
        underpredicted
        / len(high_df)
        * 100
    )

    print(
        f"{model}: "
        f"{underpredicted}/{len(high_df)} "
        f"({percentage:.2f}%)"
    )


# =========================================================
# 11. RATE PER MILE ANALYSIS
# =========================================================
# Analysis only - do NOT use as a model feature because
# it is calculated using the target actual_rate.

high_df["actual_rate_per_mile"] = (
    high_df["actual_rate"]
    /
    high_df["distance"]
)

normal_df["actual_rate_per_mile"] = (
    normal_df["actual_rate"]
    /
    normal_df["distance"]
)

print("\n" + "=" * 70)
print("RATE PER MILE ANALYSIS")
print("=" * 70)

print(
    "Normal rate/mile mean:",
    round(
        normal_df[
            "actual_rate_per_mile"
        ].mean(),
        2
    )
)

print(
    "High-value rate/mile mean:",
    round(
        high_df[
            "actual_rate_per_mile"
        ].mean(),
        2
    )
)

print(
    "\nHigh-value rate/mile statistics:"
)

print(
    high_df[
        "actual_rate_per_mile"
    ].describe()
)


# =========================================================
# 12. MOST EXTREME HIGH VALUE LOADS
# =========================================================

print("\n" + "=" * 70)
print("TOP 20 HIGHEST ACTUAL RATES")
print("=" * 70)

top_high = (
    high_df[
        [
            "pickup",
            "delivery",
            "route",
            "distance",
            "equipment",
            "weight",
            "market_index",
            "quote_signal",
            "actual_rate",
            "linear_regression",
            "catboost"
        ]
    ]
    .sort_values(
        "actual_rate",
        ascending=False
    )
    .head(20)
)

print(
    top_high.round(2)
)


# =========================================================
# 13. DISTANCE VS ACTUAL RATE
# =========================================================

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    normal_df["distance"],
    normal_df["actual_rate"],
    alpha=0.15,
    s=10,
    label="Normal"
)

plt.scatter(
    high_df["distance"],
    high_df["actual_rate"],
    alpha=0.8,
    s=20,
    label="High Value"
)

plt.axhline(
    HIGH_RATE_THRESHOLD,
    linestyle="--"
)

plt.title(
    "High Value Loads vs Normal Loads"
)

plt.xlabel(
    "Distance"
)

plt.ylabel(
    "Actual Rate ($)"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "high_value_distance.png"
    ),
    dpi=150
)

plt.show()


# =========================================================
# 14. MARKET INDEX VS HIGH VALUE
# =========================================================

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    df["market_index"],
    df["actual_rate"],
    alpha=0.25,
    s=10
)

plt.axhline(
    HIGH_RATE_THRESHOLD,
    linestyle="--"
)

plt.title(
    "Market Index vs Actual Rate"
)

plt.xlabel(
    "Market Index"
)

plt.ylabel(
    "Actual Rate ($)"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "high_value_market_index.png"
    ),
    dpi=150
)

plt.show()


# =========================================================
# 15. QUOTE SIGNAL VS HIGH VALUE
# =========================================================

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    df["quote_signal"],
    df["actual_rate"],
    alpha=0.25,
    s=10
)

plt.axhline(
    HIGH_RATE_THRESHOLD,
    linestyle="--"
)

plt.title(
    "Quote Signal vs Actual Rate"
)

plt.xlabel(
    "Quote Signal"
)

plt.ylabel(
    "Actual Rate ($)"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "high_value_quote_signal.png"
    ),
    dpi=150
)

plt.show()


# =========================================================
# 16. SAVE RESULTS
# =========================================================

high_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "high_value_loads.csv"
    ),
    index=False
)

comparison.to_csv(
    os.path.join(
        RESULTS_DIR,
        "high_value_comparison.csv"
    )
)

performance_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "high_value_model_performance.csv"
    ),
    index=False
)

route_summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "high_value_routes.csv"
    )
)


# =========================================================
# 17. FINAL MESSAGE
# =========================================================

print("\n" + "=" * 70)
print("HIGH VALUE ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nFiles saved in results/"
)