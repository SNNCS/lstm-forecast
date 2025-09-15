# residual-ar-lstm

Residual AR-LSTM for long-horizon time-series forecasting with trend/seasonality decomposition and a recursive LSTM decoder.

---

## Overview

This project implements a **Residual AutoRegressive LSTM** forecaster:
- Decompose the series into **trend** (robust median) + **weekly seasonality** (smoothed profile from training slice).
- Model the **residual** part with an **LSTM encoder** and a **recursive LSTMCell decoder** (1-step x H steps).
- Use **day-of-week embedding** as a conditioning signal to guide the decoder.
- Avoid **data leakage** by fitting scalers/profiles **only on the training segment**.
- Train with **Huber loss** for robustness; add light jitter on seasonal amplitude for regularization.

Target use case: sales/demand forecasting with weekly seasonality and long horizons (e.g., 90 days).

---

## Key features

- **LSTM encoder + LSTMCell decoder**: step-by-step recursive decoding for HORIZON steps.
- **Seasonal-trend decomposition**: stable median trend + smoothed weekly profile.
- **Leakage-safe engineering**: fit scalers/seasonality on train slice only.
- **Residual AR design**: residual lags, rolling stats, DOW embedding.
- **Robust training**: Huber loss, weight decay, dropout, CUDA support.

---

## Repository structure

```
residual-ar-lstm/
├─ Sales_Prediction_LSTM.py
├─ requirements.txt
├─ README.md
└─ train.csv
```

---

## Data

The script expects a CSV file named `train.csv` with the following columns:

- `date` (YYYY-MM-DD or any pandas-parseable date)
- `store` (int)
- `item` (int)
- `sales` (float or int)

## How To Run

# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data
# Place train.csv in the repo root with columns: date, store, item, sales

# 3. Run training + forecasting
python residual_ar_lstm.py

# 4. Check outputs
# - Console logs training losses
# - Matplotlib plot shows last 30d actual + 90d forecast
