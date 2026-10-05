import numpy as np
import pandas as pd


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def add_features(data):

    data = data.copy()

    # Date features
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek

    # Weekend flag
    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # Route feature
    data["route"] = (
        data["pickup"].astype(str)
        + "_TO_"
        + data["delivery"].astype(str)
    )

    return data


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data(df):

    df = df.copy()

    # =====================================================
    # 1. DATE CONVERSION
    # =====================================================

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )


    # =====================================================
    # 2. INVALID VALUES
    # =====================================================

    # Negative or zero weight is invalid
    df.loc[
        df["weight"] <= 0,
        "weight"
    ] = np.nan


    # =====================================================
    # 3. TIME-BASED SPLIT
    # =====================================================
    # Train: Jan -> Sep
    # Validation: October

    train_df = df[
        df["date"] < "2025-10-01"
    ].copy()

    val_df = df[
        df["date"] >= "2025-10-01"
    ].copy()


    # =====================================================
    # 4. TRAIN-ONLY IMPUTATION VALUES
    # =====================================================

    weight_median = (
        train_df["weight"]
        .median()
    )

    market_median = (
        train_df["market_index"]
        .median()
    )


    # =====================================================
    # 5. FILL MISSING VALUES
    # =====================================================

    train_df["weight"] = (
        train_df["weight"]
        .fillna(weight_median)
    )

    val_df["weight"] = (
        val_df["weight"]
        .fillna(weight_median)
    )


    train_df["market_index"] = (
        train_df["market_index"]
        .fillna(market_median)
    )

    val_df["market_index"] = (
        val_df["market_index"]
        .fillna(market_median)
    )


    # =====================================================
    # 6. FEATURE ENGINEERING
    # =====================================================

    train_df = add_features(
        train_df
    )

    val_df = add_features(
        val_df
    )


    # =====================================================
    # 7. DEFINE FEATURES
    # =====================================================

    features = [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "market_index",
        "quote_signal",
        "month",
        "day",
        "day_of_week",
        "is_weekend",
        "route"
    ]

    target = "posted_rate"


    # =====================================================
    # 8. CREATE X / y
    # =====================================================

    X_train = (
        train_df[features]
        .copy()
    )

    y_train = (
        train_df[target]
        .copy()
    )

    X_val = (
        val_df[features]
        .copy()
    )

    y_val = (
        val_df[target]
        .copy()
    )


    # =====================================================
    # 9. RETURN DATA
    # =====================================================

    return (
        X_train,
        X_val,
        y_train,
        y_val
    )