import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

MODEL_PATH = "model.pkl"
CHART_DATA_PATH = "chart_data.json"

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load model (sklearn Pipeline with preprocessing)
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. Please run 'python train_model.py' first."
    )
model = joblib.load(MODEL_PATH)

# Columns expected by the model (must match training order)
FEATURE_COLUMNS = [
    "Market",
    "Department Name",
    "Shipping Mode",
    "Days for shipment (scheduled)",
    "Late_delivery_risk",
    "Sales",
    "Benefit per order",
    "Order Profit Per Order",
]


def _coerce_payload_to_frame(payload: dict) -> pd.DataFrame:
    # Create single-row DataFrame with correct columns
    # Categorical as-is, numeric coerced
    def get_val(key, default=None):
        return payload.get(key, default)

    row = {
        "Market": get_val("Market", ""),
        "Department Name": get_val("Department Name", ""),
        "Shipping Mode": get_val("Shipping Mode", ""),
        "Days for shipment (scheduled)": pd.to_numeric(
            get_val("Days for shipment (scheduled)", None), errors="coerce"
        ),
        "Late_delivery_risk": pd.to_numeric(
            get_val("Late_delivery_risk", None), errors="coerce"
        ),
        "Sales": pd.to_numeric(get_val("Sales", None), errors="coerce"),
        "Benefit per order": pd.to_numeric(
            get_val("Benefit per order", None), errors="coerce"
        ),
        "Order Profit Per Order": pd.to_numeric(
            get_val("Order Profit Per Order", None), errors="coerce"
        ),
    }
    df = pd.DataFrame([row], columns=[
        "Market",
        "Department Name",
        "Shipping Mode",
        "Days for shipment (scheduled)",
        "Late_delivery_risk",
        "Sales",
        "Benefit per order",
        "Order Profit Per Order",
    ])
    return df


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chart-data", methods=["GET"])
def chart_data():
    if not os.path.exists(CHART_DATA_PATH):
        return jsonify({"error": f"{CHART_DATA_PATH} not found. Run training first."}), 404
    with open(CHART_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/predict", methods=["POST"])
def predict():
    # Accept JSON or form
    if request.is_json:
        payload = request.get_json(silent=True) or {}
    else:
        payload = request.form.to_dict(flat=True)

    try:
        X = _coerce_payload_to_frame(payload)
        # Basic validation: no missing critical numerics
        if X.isna().any().any():
            return jsonify({"error": "Invalid or missing input fields."}), 400

        y_pred = model.predict(X)[0]
        result = {
            "predicted_days_for_shipping": round(float(y_pred), 2),
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)