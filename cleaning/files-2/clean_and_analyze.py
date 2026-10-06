"""
Task: Data Cleaning + Initial Data Analytics
Dataset: sample_customer_data.csv
Libraries: pandas, numpy

This script is split into two clearly separated stages, matching the task:
  STAGE 1 - DATA CLEANING ONLY
  STAGE 2 - INITIAL STEP OF DATA ANALYTICS (basic EDA on the cleaned data)
"""

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

INPUT_PATH = "sample_customer_data.csv"
CLEANED_OUTPUT_PATH = "sample_customer_data_cleaned.csv"

# =========================================================
# STAGE 1: DATA CLEANING ONLY
# =========================================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[Load] Raw shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"[Clean] Removed {before - len(df)} duplicate rows")

    # 2. Standardize text columns: strip whitespace, fix casing
    df["customer_name"] = df["customer_name"].str.strip().str.title()
    df["region"] = df["region"].str.strip().str.title()
    df["product_category"] = df["product_category"].str.strip().str.title()

    # 3. Fix invalid ages (negative values -> absolute value; unrealistic ages -> NaN)
    df["age"] = df["age"].apply(lambda x: abs(x) if pd.notna(x) else x)
    df.loc[(df["age"] < 0) | (df["age"] > 100), "age"] = np.nan

    # 4. Handle outliers in purchase_amount using the IQR method
    q1 = df["purchase_amount"].quantile(0.25)
    q3 = df["purchase_amount"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (df["purchase_amount"] < lower_bound) | (df["purchase_amount"] > upper_bound)
    print(f"[Clean] Flagged {outlier_mask.sum()} purchase_amount outliers (capped to bounds)")
    df["purchase_amount"] = np.clip(df["purchase_amount"], lower_bound, upper_bound)

    # 5. Convert signup_date to proper datetime
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

    # 6. Impute missing numeric values with median
    for col in ["age", "purchase_amount"]:
        missing = df[col].isna().sum()
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"[Clean] Filled {missing} missing values in '{col}' with median ({median_val:.2f})")

    # 7. Handle missing categorical values
    df["region"] = df["region"].fillna("Unknown")
    df["product_category"] = df["product_category"].fillna("Unknown")

    # 8. Missing satisfaction_score -> treat as NaN (kept, since imputing an opinion score is risky);
    #    but ensure dtype is consistent (nullable integer)
    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce").astype("Int64")

    # 9. Final dtype tidy-up
    df["age"] = df["age"].round(0).astype(int)
    df["purchase_amount"] = df["purchase_amount"].round(2)

    print(f"[Clean] Final shape: {df.shape}")
    return df


# =========================================================
# STAGE 2: INITIAL STEP OF DATA ANALYTICS
# =========================================================

def initial_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("INITIAL DATA ANALYTICS")
    print("=" * 60)

    print("\n--- Structure & dtypes ---")
    print(df.dtypes)

    print("\n--- Summary statistics (numeric) ---")
    print(df.describe())

    print("\n--- Missing value check (post-clean) ---")
    print(df.isna().sum())

    print("\n--- Records per region ---")
    print(df["region"].value_counts())

    print("\n--- Records per product category ---")
    print(df["product_category"].value_counts())

    print("\n--- Average purchase amount by region ---")
    print(df.groupby("region")["purchase_amount"].mean().round(2).sort_values(ascending=False))

    print("\n--- Average satisfaction score by product category ---")
    print(df.groupby("product_category")["satisfaction_score"].mean().round(2))

    print("\n--- Correlation (numeric columns) ---")
    print(df[["age", "purchase_amount", "satisfaction_score"]].corr(numeric_only=True).round(2))


def main():
    raw_df = load_data(INPUT_PATH)
    cleaned_df = clean_data(raw_df)
    cleaned_df.to_csv(CLEANED_OUTPUT_PATH, index=False)
    print(f"[Save] Cleaned data written to {CLEANED_OUTPUT_PATH}")

    initial_analysis(cleaned_df)


if __name__ == "__main__":
    main()
