import pandas as pd
import matplotlib.pyplot as plt

from ETL.Extract import load_data


# =========================================================
# LOAD DATA
# =========================================================

df = load_data("data/train-test.csv")

# Work on a copy for EDA only
eda_df = df.copy()

# Convert date
eda_df["date"] = pd.to_datetime(
    eda_df["date"],
    errors="coerce"
)


# =========================================================
# 1. BASIC INFORMATION
# =========================================================

print("=" * 60)
print("DATASET BASIC INFORMATION")
print("=" * 60)

print("\nShape:")
print(eda_df.shape)

print("\nFirst 5 Rows:")
print(eda_df.head())

print("\nColumn Names:")
print(eda_df.columns.tolist())

print("\nData Types:")
print(eda_df.dtypes)

print("\nDataset Info:")
eda_df.info()


# =========================================================
# 2. MISSING VALUES
# =========================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_table = pd.DataFrame({
    "Missing Count": eda_df.isnull().sum(),
    "Missing %": (
        eda_df.isnull().sum()
        / len(eda_df)
        * 100
    ).round(2)
})

print(missing_table)


# =========================================================
# 3. DUPLICATES
# =========================================================

print("\n" + "=" * 60)
print("DUPLICATE CHECK")
print("=" * 60)

print(
    "\nDuplicate Rows:",
    eda_df.duplicated().sum()
)

print(
    "Duplicate Load IDs:",
    eda_df["load_id"].duplicated().sum()
)

print(
    "Unique Load IDs:",
    eda_df["load_id"].nunique()
)


# =========================================================
# 4. DATE CHECK
# =========================================================

print("\n" + "=" * 60)
print("DATE CHECK")
print("=" * 60)

print(
    "\nInvalid Dates:",
    eda_df["date"].isnull().sum()
)

print(
    "Start Date:",
    eda_df["date"].min()
)

print(
    "End Date:",
    eda_df["date"].max()
)

print(
    "Unique Dates:",
    eda_df["date"].nunique()
)


# =========================================================
# 5. NUMERICAL SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

numeric_columns = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "posted_rate"
]

print(
    eda_df[numeric_columns]
    .describe()
    .T
)


# =========================================================
# 6. INVALID VALUE CHECK
# =========================================================

print("\n" + "=" * 60)
print("INVALID VALUE CHECK")
print("=" * 60)

print(
    "\nDistance <= 0:",
    (eda_df["distance"] <= 0).sum()
)

print(
    "Weight <= 0:",
    (eda_df["weight"] <= 0).sum()
)

print(
    "Posted Rate <= 0:",
    (eda_df["posted_rate"] <= 0).sum()
)

print(
    "Market Index <= 0:",
    (eda_df["market_index"] <= 0).sum()
)

print(
    "Quote Signal <= 0:",
    (eda_df["quote_signal"] <= 0).sum()
)


# =========================================================
# 7. CATEGORICAL VARIABLES
# =========================================================

print("\n" + "=" * 60)
print("CATEGORICAL VARIABLES")
print("=" * 60)

print(
    "\nUnique Pickup Cities:",
    eda_df["pickup"].nunique()
)

print(
    "Unique Delivery Cities:",
    eda_df["delivery"].nunique()
)

print("\nEquipment Types:")
print(
    eda_df["equipment"]
    .value_counts()
)

print("\nTop 10 Pickup Cities:")
print(
    eda_df["pickup"]
    .value_counts()
    .head(10)
)

print("\nTop 10 Delivery Cities:")
print(
    eda_df["delivery"]
    .value_counts()
    .head(10)
)


# =========================================================
# 8. TARGET ANALYSIS
# =========================================================

print("\n" + "=" * 60)
print("TARGET ANALYSIS: POSTED RATE")
print("=" * 60)

print(
    eda_df["posted_rate"]
    .describe()
)

print(
    "\nMean:",
    round(
        eda_df["posted_rate"].mean(),
        2
    )
)

print(
    "Median:",
    round(
        eda_df["posted_rate"].median(),
        2
    )
)

print(
    "Skewness:",
    round(
        eda_df["posted_rate"].skew(),
        3
    )
)


# =========================================================
# 9. OUTLIER CHECK
# =========================================================

print("\n" + "=" * 60)
print("OUTLIER CHECK - IQR")
print("=" * 60)

for column in [
    "distance",
    "weight",
    "posted_rate"
]:

    values = eda_df[column].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = eda_df[
        (eda_df[column] < lower_bound)
        |
        (eda_df[column] > upper_bound)
    ]

    print(f"\n{column}")

    print(
        f"Lower Bound: "
        f"{lower_bound:.2f}"
    )

    print(
        f"Upper Bound: "
        f"{upper_bound:.2f}"
    )

    print(
        f"Number of Outliers: "
        f"{len(outliers)}"
    )

    print(
        f"Outlier Percentage: "
        f"{len(outliers) / len(eda_df) * 100:.2f}%"
    )


# =========================================================
# 10. CORRELATION
# =========================================================

print("\n" + "=" * 60)
print("CORRELATION WITH POSTED RATE")
print("=" * 60)

correlation = (
    eda_df[numeric_columns]
    .corr()["posted_rate"]
    .sort_values(
        ascending=False
    )
)

print(correlation)


# =========================================================
# 11. DATE DISTRIBUTION
# =========================================================

print("\n" + "=" * 60)
print("DATE DISTRIBUTION")
print("=" * 60)

eda_df["month"] = (
    eda_df["date"]
    .dt.month
)

eda_df["day_of_week"] = (
    eda_df["date"]
    .dt.day_name()
)

print("\nRows Per Month:")

print(
    eda_df["month"]
    .value_counts()
    .sort_index()
)

print("\nRows Per Day of Week:")

print(
    eda_df["day_of_week"]
    .value_counts()
)


# =========================================================
# 12. POSTED RATE BY EQUIPMENT
# =========================================================

print("\n" + "=" * 60)
print("POSTED RATE BY EQUIPMENT")
print("=" * 60)

equipment_summary = (
    eda_df
    .groupby("equipment")["posted_rate"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max"
        ]
    )
    .sort_values(
        "mean",
        ascending=False
    )
)

print(equipment_summary)


# =========================================================
# 13. PLOTS
# =========================================================


# Posted Rate Distribution
plt.figure(
    figsize=(10, 5)
)

plt.hist(
    eda_df["posted_rate"],
    bins=50
)

plt.title(
    "Distribution of Posted Rate"
)

plt.xlabel(
    "Posted Rate ($)"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()
plt.show()


# Distance Distribution
plt.figure(
    figsize=(10, 5)
)

plt.hist(
    eda_df["distance"],
    bins=50
)

plt.title(
    "Distribution of Distance"
)

plt.xlabel(
    "Distance"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()
plt.show()


# Weight Distribution
plt.figure(
    figsize=(10, 5)
)

plt.hist(
    eda_df["weight"].dropna(),
    bins=50
)

plt.title(
    "Distribution of Weight"
)

plt.xlabel(
    "Weight"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()
plt.show()


# Distance vs Posted Rate
plt.figure(
    figsize=(10, 5)
)

plt.scatter(
    eda_df["distance"],
    eda_df["posted_rate"],
    alpha=0.3,
    s=10
)

plt.title(
    "Distance vs Posted Rate"
)

plt.xlabel(
    "Distance"
)

plt.ylabel(
    "Posted Rate ($)"
)

plt.tight_layout()
plt.show()


# Weight vs Posted Rate
plt.figure(
    figsize=(10, 5)
)

plt.scatter(
    eda_df["weight"],
    eda_df["posted_rate"],
    alpha=0.3,
    s=10
)

plt.title(
    "Weight vs Posted Rate"
)

plt.xlabel(
    "Weight"
)

plt.ylabel(
    "Posted Rate ($)"
)

plt.tight_layout()
plt.show()


# =========================================================
# 14. FINAL SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("FINAL DATA CHECK SUMMARY")
print("=" * 60)

print(
    f"Rows: "
    f"{len(eda_df)}"
)

print(
    f"Original Columns: "
    f"{len(df.columns)}"
)

print(
    f"Missing Values: "
    f"{df.isnull().sum().sum()}"
)

print(
    f"Duplicate Rows: "
    f"{df.duplicated().sum()}"
)

print(
    f"Invalid Weights <= 0: "
    f"{(df['weight'] <= 0).sum()}"
)

print(
    f"Posted Rate Mean: "
    f"{df['posted_rate'].mean():.2f}"
)

print(
    f"Posted Rate Median: "
    f"{df['posted_rate'].median():.2f}"
)

print(
    f"Posted Rate Skewness: "
    f"{df['posted_rate'].skew():.3f}"
)

print(
    f"Date Range: "
    f"{eda_df['date'].min()} "
    f"to "
    f"{eda_df['date'].max()}"
)