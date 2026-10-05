import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. LOAD DATA
# =========================================================

def load_data(path):
    return pd.read_csv(path)


# =========================================================
# 2. EDA FUNCTION
# =========================================================

def run_eda(df):

    df = df.copy()

    # =====================================================
    # BASIC INFORMATION
    # =====================================================

    print("=" * 60)
    print("DATASET BASIC INFORMATION")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nDataset Info:")
    df.info()


    # =====================================================
    # MISSING VALUES
    # =====================================================

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    missing_count = df.isnull().sum()

    missing_percent = (
        df.isnull().sum()
        / len(df)
        * 100
    )

    missing_table = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing %": missing_percent.round(2)
    })

    print(missing_table)


    # =====================================================
    # DUPLICATES
    # =====================================================

    print("\n" + "=" * 60)
    print("DUPLICATE CHECK")
    print("=" * 60)

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nDuplicate Load IDs:")
    print(df["load_id"].duplicated().sum())

    print("\nUnique Load IDs:")
    print(df["load_id"].nunique())


    # =====================================================
    # DATE CHECK
    # =====================================================

    print("\n" + "=" * 60)
    print("DATE CHECK")
    print("=" * 60)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    print("\nInvalid Dates:")
    print(df["date"].isnull().sum())

    print("\nStart Date:")
    print(df["date"].min())

    print("\nEnd Date:")
    print(df["date"].max())

    print("\nNumber of Unique Dates:")
    print(df["date"].nunique())


    # =====================================================
    # NUMERICAL SUMMARY
    # =====================================================

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
        df[numeric_columns]
        .describe()
        .T
    )


    # =====================================================
    # INVALID VALUES
    # =====================================================

    print("\n" + "=" * 60)
    print("INVALID VALUE CHECK")
    print("=" * 60)

    print("\nDistance <= 0:")
    print((df["distance"] <= 0).sum())

    print("\nWeight <= 0:")
    print((df["weight"] <= 0).sum())

    print("\nPosted Rate <= 0:")
    print((df["posted_rate"] <= 0).sum())

    print("\nMarket Index <= 0:")
    print((df["market_index"] <= 0).sum())

    print("\nQuote Signal <= 0:")
    print((df["quote_signal"] <= 0).sum())


    # =====================================================
    # CATEGORICAL VARIABLES
    # =====================================================

    print("\n" + "=" * 60)
    print("CATEGORICAL VARIABLES")
    print("=" * 60)

    print("\nUnique Pickup Cities:")
    print(df["pickup"].nunique())

    print("\nUnique Delivery Cities:")
    print(df["delivery"].nunique())

    print("\nEquipment Types:")
    print(df["equipment"].value_counts())

    print("\nTop 10 Pickup Cities:")
    print(
        df["pickup"]
        .value_counts()
        .head(10)
    )

    print("\nTop 10 Delivery Cities:")
    print(
        df["delivery"]
        .value_counts()
        .head(10)
    )


    # =====================================================
    # TARGET ANALYSIS
    # =====================================================

    print("\n" + "=" * 60)
    print("TARGET: POSTED RATE")
    print("=" * 60)

    print("\nPosted Rate Statistics:")
    print(df["posted_rate"].describe())

    print("\nPosted Rate Skewness:")
    print(df["posted_rate"].skew())

    print("\nPosted Rate Median:")
    print(df["posted_rate"].median())


    # =====================================================
    # OUTLIERS
    # =====================================================

    print("\n" + "=" * 60)
    print("OUTLIER CHECK")
    print("=" * 60)

    for column in [
        "distance",
        "weight",
        "posted_rate"
    ]:

        clean_series = df[column].dropna()

        q1 = clean_series.quantile(0.25)
        q3 = clean_series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df[
            (df[column] < lower_bound)
            |
            (df[column] > upper_bound)
        ]

        print(f"\n{column}")
        print(f"Lower Bound: {lower_bound:.2f}")
        print(f"Upper Bound: {upper_bound:.2f}")
        print(f"Number of Outliers: {len(outliers)}")

        print(
            "Outlier Percentage: "
            f"{len(outliers) / len(df) * 100:.2f}%"
        )


    # =====================================================
    # CORRELATION
    # =====================================================

    print("\n" + "=" * 60)
    print("CORRELATION WITH POSTED RATE")
    print("=" * 60)

    correlation = (
        df[numeric_columns]
        .corr()["posted_rate"]
        .sort_values(ascending=False)
    )

    print(correlation)


    # =====================================================
    # DATE FEATURES FOR EDA ONLY
    # =====================================================

    print("\n" + "=" * 60)
    print("DATE DISTRIBUTION")
    print("=" * 60)

    df["month"] = df["date"].dt.month

    df["day_of_week"] = (
        df["date"]
        .dt.day_name()
    )

    print("\nRows Per Month:")

    print(
        df["month"]
        .value_counts()
        .sort_index()
    )

    print("\nRows Per Day of Week:")

    print(
        df["day_of_week"]
        .value_counts()
    )


    # =====================================================
    # POSTED RATE BY EQUIPMENT
    # =====================================================

    print("\n" + "=" * 60)
    print("POSTED RATE BY EQUIPMENT")
    print("=" * 60)

    equipment_summary = (
        df
        .groupby("equipment")["posted_rate"]
        .agg([
            "count",
            "mean",
            "median",
            "min",
            "max"
        ])
        .sort_values(
            "mean",
            ascending=False
        )
    )

    print(equipment_summary)


    # =====================================================
    # PLOTS
    # =====================================================

    # Posted Rate Distribution
    plt.figure(figsize=(10, 5))

    plt.hist(
        df["posted_rate"].dropna(),
        bins=50
    )

    plt.title("Distribution of Posted Rate")
    plt.xlabel("Posted Rate ($)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


    # Distance Distribution
    plt.figure(figsize=(10, 5))

    plt.hist(
        df["distance"].dropna(),
        bins=50
    )

    plt.title("Distribution of Distance")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


    # Weight Distribution
    plt.figure(figsize=(10, 5))

    plt.hist(
        df["weight"].dropna(),
        bins=50
    )

    plt.title("Distribution of Weight")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


    # Distance vs Posted Rate
    plt.figure(figsize=(10, 5))

    plt.scatter(
        df["distance"],
        df["posted_rate"],
        alpha=0.3,
        s=10
    )

    plt.title("Distance vs Posted Rate")
    plt.xlabel("Distance")
    plt.ylabel("Posted Rate ($)")

    plt.tight_layout()
    plt.show()


    # Weight vs Posted Rate
    plt.figure(figsize=(10, 5))

    plt.scatter(
        df["weight"],
        df["posted_rate"],
        alpha=0.3,
        s=10
    )

    plt.title("Weight vs Posted Rate")
    plt.xlabel("Weight")
    plt.ylabel("Posted Rate ($)")

    plt.tight_layout()
    plt.show()


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("\n" + "=" * 60)
    print("FINAL DATA CHECK SUMMARY")
    print("=" * 60)

    print(
        f"Rows: {df.shape[0]}"
    )

    print(
        f"Original Columns: 14"
    )

    print(
        f"EDA Columns After Date Features: "
        f"{df.shape[1]}"
    )

    print(
        "Total Missing Values: "
        f"{df.isnull().sum().sum()}"
    )

    print(
        "Duplicate Rows: "
        f"{df.duplicated().sum()}"
    )

    print(
        "Duplicate Load IDs: "
        f"{df['load_id'].duplicated().sum()}"
    )

    print(
        "Invalid Weights <= 0: "
        f"{(df['weight'] <= 0).sum()}"
    )

    print(
        "Posted Rate Mean: "
        f"{df['posted_rate'].mean():.2f}"
    )

    print(
        "Posted Rate Median: "
        f"{df['posted_rate'].median():.2f}"
    )

    print(
        "Posted Rate Skewness: "
        f"{df['posted_rate'].skew():.3f}"
    )

    print(
        f"Date Range: "
        f"{df['date'].min()} "
        f"to "
        f"{df['date'].max()}"
    )


# =========================================================
# 3. RUN FILE
# =========================================================

if __name__ == "__main__":

    df = load_data(
        "data/train-test.csv"
    )

    run_eda(df)