import json
import os
import sys
import time
import warnings

from data_pipeline import load_and_merge_data, compute_analytics

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=UserWarning)

RAW_DIR = os.path.join("data_raw")
MAIN_FILE = os.path.join(RAW_DIR, "DataCoSupplyChainDataset.csv")
JOIN_FILES = [
    os.path.join(RAW_DIR, "orders_and_shipments.csv"),
    os.path.join(RAW_DIR, "inventory.csv"),
    os.path.join(RAW_DIR, "fulfillment.csv"),
    os.path.join(RAW_DIR, "tokenized_access_logs.csv"),
]

MODEL_PATH = "model.pkl"
CHART_DATA_PATH = "chart_data.json"

SELECTED_COLUMNS = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Late_delivery_risk",
    "Market",
    "Department Name",
    "Shipping Mode",
    "Sales",
    "Benefit per order",
    "Order Profit Per Order",
]


def safe_read_csv(path: str, encoding: str = "latin1") -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding=encoding)
    except Exception:
        return pd.DataFrame()


def main():
    print("[1/8] Building unified dataset via data_pipeline...", flush=True)
    df = load_and_merge_data()
    print(f"      Unified rows: {len(df):,}", flush=True)

    # Keep only the required columns for modeling
    missing_cols = [c for c in SELECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required columns in dataset: {missing_cols}")

    print("[3/8] Selecting required columns...", flush=True)
    df = df[SELECTED_COLUMNS].copy()
    print(f"      Columns selected. Current rows: {len(df):,}.", flush=True)

    # Drop rows with missing values
    print("[4/8] Dropping rows with missing values...", flush=True)
    df = df.dropna(axis=0, how="any")
    print(f"      Remaining rows after dropna: {len(df):,}.", flush=True)

    # Ensure numeric columns are numeric
    numeric_features = [
        "Days for shipment (scheduled)",
        "Late_delivery_risk",
        "Sales",
        "Benefit per order",
        "Order Profit Per Order",
    ]
    print("[5/8] Coercing numeric features...", flush=True)
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows that became NaN after coercion
    df = df.dropna(axis=0, how="any")
    print(f"      Rows after numeric coercion and dropna: {len(df):,}.", flush=True)

    # Define target and features
    target_col = "Days for shipping (real)"
    X = df.drop(columns=[target_col])
    y = pd.to_numeric(df[target_col], errors="coerce")
    # Drop rows where target is NaN
    valid_mask = y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    categorical_features = [
        "Market",
        "Department Name",
        "Shipping Mode",
    ]

    # Preprocess: OneHotEncode categorical columns; pass through remainder
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    model = RandomForestRegressor(random_state=42, n_estimators=200)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # Train/test split
    print("[6/8] Splitting train/test (80/20)...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
        # shuffle True by default
    )
    print(f"      Train: {len(X_train):,} rows | Test: {len(X_test):,} rows.", flush=True)

    # Fit
    print("[7/8] Fitting model (RandomForestRegressor, n_estimators=200)...", flush=True)
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"      Model trained in {time.time() - t0:.1f}s.", flush=True)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    # Use sqrt of MSE for compatibility with older scikit-learn versions
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(f"      Test RMSE: {rmse:.4f}", flush=True)

    # Save model (pipeline includes preprocessing)
    print("[8/8] Saving artifacts...", flush=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"      Saved model to {MODEL_PATH}", flush=True)

    # Build richer analytics payload for dashboard
    analytics = compute_analytics(df)
    chart_payload = {
        "byMarket": analytics.get("byMarket", {"labels": [], "data": []}),
        "byDepartment": analytics.get("byDepartment", {"labels": [], "data": []}),
        "stockVsDelay": analytics.get("stockVsDelay", {"x": [], "y": []}),
    }

    with open(CHART_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chart_payload, f, ensure_ascii=False, indent=2)
    print(f"      Wrote chart data to {CHART_DATA_PATH}", flush=True)


if __name__ == "__main__":
    main()
