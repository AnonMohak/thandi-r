import os
import json
import pandas as pd


RAW_DIR = os.path.join("data_raw")
MAIN_FILE = os.path.join(RAW_DIR, "DataCoSupplyChainDataset.csv")
ORDERS_FILE = os.path.join(RAW_DIR, "orders_and_shipments.csv")
INVENTORY_FILE = os.path.join(RAW_DIR, "inventory.csv")
FULFILLMENT_FILE = os.path.join(RAW_DIR, "fulfillment.csv")
ACCESS_LOGS_FILE = os.path.join(RAW_DIR, "tokenized_access_logs.csv")


def safe_read_csv(path: str, encoding: str = "latin1") -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding=encoding)
    except Exception:
        return pd.DataFrame()


def _best_key(df: pd.DataFrame) -> str:
    if "Product ID" in df.columns:
        return "Product ID"
    return "Product Name" if "Product Name" in df.columns else None


def load_and_merge_data() -> pd.DataFrame:
    main_df = safe_read_csv(MAIN_FILE, encoding="latin1")
    if main_df.empty:
        raise RuntimeError(f"Could not read main dataset at {MAIN_FILE}")

    base_key = _best_key(main_df)
    if base_key is None:
        raise RuntimeError("Main dataset missing 'Product Name' and 'Product ID'")

    # Normalize key presence
    work_df = main_df.copy()

    # Orders/Shipments (light join)
    ord_df = safe_read_csv(ORDERS_FILE, encoding="latin1")
    k = _best_key(ord_df)
    if not ord_df.empty and k is not None and k in work_df.columns:
        ord_df = ord_df.drop_duplicates(subset=[k])[[k]]
        work_df = work_df.merge(ord_df, on=k, how="left")

    # Inventory features
    inv_df = safe_read_csv(INVENTORY_FILE, encoding="latin1")
    k = _best_key(inv_df)
    if not inv_df.empty and k is not None and k in work_df.columns:
        # Detect alternative column names for stock/reorder
        stock_candidates = [
            "Stock Level", "StockLevel", "Stock", "Quantity", "On Hand",
            "OnHand", "Inventory", "Inventory Level", "Units"
        ]
        reorder_candidates = [
            "Reorder Level", "ReorderLevel", "Reorder", "Min Stock", "MinStock"
        ]
        stock_col = next((c for c in stock_candidates if c in inv_df.columns), None)
        reorder_col = next((c for c in reorder_candidates if c in inv_df.columns), None)
        keep_cols = [k]
        if stock_col:
            keep_cols.append(stock_col)
        if reorder_col:
            keep_cols.append(reorder_col)
        if len(keep_cols) > 1:
            inv_df = inv_df[keep_cols].drop_duplicates(subset=[k])
            # Normalize column names after merge
            rename_map = {}
            if stock_col and stock_col != "Stock Level":
                rename_map[stock_col] = "Stock Level"
            if reorder_col and reorder_col != "Reorder Level":
                rename_map[reorder_col] = "Reorder Level"
            inv_df = inv_df.rename(columns=rename_map)
            work_df = work_df.merge(inv_df, on=k, how="left")

    # Fulfillment KPIs
    ful_df = safe_read_csv(FULFILLMENT_FILE, encoding="latin1")
    k = _best_key(ful_df)
    if not ful_df.empty and k is not None and k in work_df.columns:
        ful_small_cols = [
            c for c in [k, "Fulfillment Time", "Warehouse Efficiency"] if c in ful_df.columns
        ]
        if len(ful_small_cols) >= 2:
            ful_df = ful_df[ful_small_cols].drop_duplicates(subset=[k])
            work_df = work_df.merge(ful_df, on=k, how="left")

    # Access logs -> Access Count per product
    log_df = safe_read_csv(ACCESS_LOGS_FILE, encoding="latin1")
    k = _best_key(log_df)
    if not log_df.empty and k is not None and k in work_df.columns:
        access_counts = log_df.groupby(k).size().reset_index(name="Access Count")
        work_df = work_df.merge(access_counts, on=k, how="left")

    return work_df


def compute_analytics(df: pd.DataFrame) -> dict:
    analytics = {}

    # Average delivery days by Market
    if "Market" in df.columns and "Days for shipping (real)" in df.columns:
        s = (
            df.dropna(subset=["Market", "Days for shipping (real)"])
            .groupby("Market")["Days for shipping (real)"]
            .mean()
            .sort_values(ascending=False)
            .round(2)
        )
        analytics["byMarket"] = {
            "labels": s.index.tolist(),
            "data": s.values.tolist(),
        }

    # Average delivery days by Department
    if "Department Name" in df.columns and "Days for shipping (real)" in df.columns:
        s = (
            df.dropna(subset=["Department Name", "Days for shipping (real)"])
            .groupby("Department Name")["Days for shipping (real)"]
            .mean()
            .sort_values(ascending=False)
            .round(2)
        )
        analytics["byDepartment"] = {
            "labels": s.index.tolist(),
            "data": s.values.tolist(),
        }

    # Stock Level vs Delivery Delay scatter (if available)
    if (
        "Stock Level" in df.columns
        and "Days for shipping (real)" in df.columns
    ):
        scatter_df = df[["Stock Level", "Days for shipping (real)"]].dropna()
        if len(scatter_df) > 0:
            # Coerce to numeric and drop non-numeric
            scatter_df["Stock Level"] = pd.to_numeric(scatter_df["Stock Level"], errors="coerce")
            scatter_df["Days for shipping (real)"] = pd.to_numeric(scatter_df["Days for shipping (real)"], errors="coerce")
            scatter_df = scatter_df.dropna()
            if len(scatter_df) > 0:
                # Sample to avoid huge payloads
                sample = scatter_df.sample(n=min(1000, len(scatter_df)), random_state=42)
                analytics["stockVsDelay"] = {
                    "x": sample["Stock Level"].astype(float).tolist(),
                    "y": sample["Days for shipping (real)"].astype(float).tolist(),
                }

    return analytics


