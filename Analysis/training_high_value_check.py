import pandas as pd

from ETL.Extract import load_data


# =========================================================
# LOAD DATA
# =========================================================

df = load_data("data/train-test.csv")

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# =========================================================
# TIME SPLIT
# =========================================================

train_df = df[
    df["date"] < "2025-10-01"
].copy()

val_df = df[
    df["date"] >= "2025-10-01"
].copy()


# =========================================================
# HIGH VALUE THRESHOLD
# =========================================================

THRESHOLD = 7000


train_high = train_df[
    train_df["posted_rate"] > THRESHOLD
].copy()

val_high = val_df[
    val_df["posted_rate"] > THRESHOLD
].copy()


# =========================================================
# COUNTS
# =========================================================

print("=" * 70)
print("HIGH VALUE FREQUENCY")
print("=" * 70)

print(
    "Training rows:",
    len(train_df)
)

print(
    "Training high-value rows:",
    len(train_high)
)

print(
    "Training high-value %:",
    round(
        len(train_high) / len(train_df) * 100,
        3
    )
)

print()

print(
    "Validation rows:",
    len(val_df)
)

print(
    "Validation high-value rows:",
    len(val_high)
)

print(
    "Validation high-value %:",
    round(
        len(val_high) / len(val_df) * 100,
        3
    )
)


# =========================================================
# HIGH VALUE BY MONTH
# =========================================================

print("\n" + "=" * 70)
print("HIGH VALUE BY MONTH")
print("=" * 70)

train_high["month"] = (
    train_high["date"].dt.month
)

monthly = (
    train_high["month"]
    .value_counts()
    .sort_index()
)

print(monthly)


# =========================================================
# HIGH VALUE STATS
# =========================================================

print("\n" + "=" * 70)
print("TRAIN HIGH VALUE STATISTICS")
print("=" * 70)

columns = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "posted_rate"
]

print(
    train_high[columns]
    .describe()
    .T
)


# =========================================================
# EQUIPMENT
# =========================================================

print("\n" + "=" * 70)
print("TRAIN HIGH VALUE BY EQUIPMENT")
print("=" * 70)

print(
    train_high["equipment"]
    .value_counts()
)


# =========================================================
# RATE PER MILE
# =========================================================

train_high["rate_per_mile"] = (
    train_high["posted_rate"]
    /
    train_high["distance"]
)

print("\n" + "=" * 70)
print("TRAIN HIGH VALUE RATE PER MILE")
print("=" * 70)

print(
    train_high["rate_per_mile"]
    .describe()
)


# =========================================================
# TOP HIGH VALUE LOADS
# =========================================================

print("\n" + "=" * 70)
print("TOP TRAIN HIGH VALUE LOADS")
print("=" * 70)

print(
    train_high[
        [
            "date",
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "market_index",
            "quote_signal",
            "posted_rate",
            "rate_per_mile"
        ]
    ]
    .sort_values(
        "posted_rate",
        ascending=False
    )
    .head(30)
    .round(2)
)