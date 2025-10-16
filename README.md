# Delivery Time Dashboard (Scalable Data Integration + Flask + scikit-learn)

An end-to-end example that integrates product data from multiple CSV sources, builds enriched features, trains a model to predict `Days for shipping (real)`, and serves a modern analytics and prediction dashboard.

## Features
- Data integration (`data_pipeline.py`): merges `DataCoSupplyChainDataset.csv`, `orders_and_shipments.csv`, `inventory.csv`, `fulfillment.csv`, and `tokenized_access_logs.csv` by `Product ID` (if present) or `Product Name`. Derives demand metric `Access Count` from logs.
- Model (`train_model.py`): trains a `RandomForestRegressor` with `OneHotEncoder` preprocessing; uses enriched features (inventory, fulfillment, demand) alongside shipment details.
- Analytics: exports `chart_data.json` with:
  - Average delivery days by Market
  - Average delivery days by Department
  - Scatter: Stock Level vs Delivery Delay
- Flask backend (`app.py`):
  - `GET /` dashboard UI
  - `GET /chart-data` analytics JSON
  - `POST /predict` delivery time prediction (keeps existing input fields)
- Frontend: Chart.js visualizations (bar + bar + scatter) and axios-powered prediction form.

## Project Structure
```
.
├─ data_raw/
│  ├─ DataCoSupplyChainDataset.csv
│  ├─ orders_and_shipments.csv
│  ├─ inventory.csv
│  ├─ fulfillment.csv
│  └─ tokenized_access_logs.csv
├─ data_pipeline.py        # merges + computes analytics
├─ train_model.py
├─ model.pkl               # created after training
├─ chart_data.json         # created after training
├─ app.py
├─ templates/
│  └─ index.html
└─ static/
   ├─ style.css
   └─ script.js
```

## Requirements
- Python 3.9+
- pip

Install dependencies:
```
pip install flask pandas scikit-learn joblib numpy
```

## Step-by-step: How to Run

1) Prepare data
- Place the provided CSV files in `data_raw/` as shown above.

2) Train the model and generate analytics
From the project root:
```
python train_model.py
```
You will see step-by-step logs:
- Building unified dataset via `data_pipeline.py`
- Cleaning, coercing numerics, splitting train/test
- Fitting model and reporting RMSE
- Writing `model.pkl` and enriched `chart_data.json`

3) Run the Flask app
Start the server:
```
python app.py
```
Open the dashboard at:
```
http://127.0.0.1:5000
```

## API
- `GET /chart-data`
  - Returns enriched analytics from `chart_data.json` as JSON
- `POST /predict`
  - Accepts JSON or form fields:
    - `Market` (str)
    - `Department Name` (str)
    - `Shipping Mode` (str)
    - `Days for shipment (scheduled)` (number)
    - `Late_delivery_risk` (0/1)
    - `Sales` (number)
    - `Benefit per order` (number)
    - `Order Profit Per Order` (number)
  - Returns `{ "predicted_days_for_shipping": number }`

## Notes
- Run `train_model.py` first to generate `model.pkl` and `chart_data.json`.
- The model pipeline handles categorical encoding; the service expects raw categorical strings and numeric values as listed above.
- CSV encoding is set to `latin1` to accommodate the provided dataset.
- For scalable storage, you can extend `data_pipeline.py` to persist intermediate tables in SQLite/PostgreSQL and have `app.py` query them.
