# -*- coding: utf-8 -*-
# Residual AR with recursive decoding:
# LSTM encoder + LSTMCell decoder (1-step x HORIZON), DOW embedding, no leakage in scalers/profiles.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ===== Device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
torch.backends.cudnn.benchmark = True

# ===== Hyperparams =====
WINDOW_SIZE = 60
HORIZON     = 90
EPOCHS      = 80
LR          = 1e-3
BATCH_SIZE  = 128
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
DOW_DIM     = 4
WEIGHT_DECAY= 1e-4

# ===== Helpers =====
def weekly_profile(df_group):
    """Compute smoothed weekly profile over training segment (shape: (7,))."""
    prof = df_group.groupby("dow")["sales"].mean().reindex(range(7)).fillna(method="bfill").values
    prof = pd.Series(prof).rolling(3, center=True, min_periods=1).mean().values
    return np.maximum(prof, 0.0)

def seasonality_strength(sales, dow, week_prof):
    """Fraction of variance explained by weekly seasonality."""
    if len(sales) < 8:
        return 0.0
    season = week_prof[dow]
    resid  = sales - season
    v_all, v_res = np.var(sales), np.var(resid)
    return float(0.0 if v_all <= 1e-8 else np.clip(1.0 - v_res / v_all, 0.0, 1.0))

def stable_trend_baseline(sales, k=30):
    """Robust trend: median of the last k points."""
    k = min(k, len(sales))
    return float(np.median(sales[-k:]))

def add_rolling_on_residual(residual):
    """Rolling mean/std on residuals (shifted by 1 to avoid leakage)."""
    r = pd.Series(residual)
    m7  = r.rolling(7,  min_periods=1).mean().shift(1).bfill().values
    s14 = r.rolling(14, min_periods=1).std(ddof=0).shift(1).bfill().fillna(0.0).values
    return np.vstack([m7, s14]).T

def lag_feats_on_residual(residual):
    """Lag features on residuals."""
    r = pd.Series(residual)
    lag1  = r.shift(1).bfill().values
    lag7  = r.shift(7).bfill().values
    lag14 = r.shift(14).bfill().values
    return np.vstack([lag1, lag7, lag14]).T

# ===== Dataset =====
class SeqDS(Dataset):
    def __init__(self, X, Y, last_dow, future_dow):
        self.X = torch.from_numpy(X)                 
        self.Y = torch.from_numpy(Y)                 
        self.last_dow = torch.from_numpy(last_dow)   
        self.future_dow = torch.from_numpy(future_dow)  
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.last_dow[i], self.future_dow[i]

# ===== Model =====
class ResidualAR(nn.Module):
    def __init__(self, input_size, hidden=128, layers=2, dropout=0.2, horizon=90, dow_dim=4):
        super().__init__()
        self.horizon = horizon
        self.dow_emb = nn.Embedding(7, dow_dim)
        enc_in = input_size + dow_dim
        self.encoder = nn.LSTM(enc_in, hidden, layers, batch_first=True, dropout=dropout)
        self.dec_cell = nn.LSTMCell(1 + dow_dim, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, last_dow_idx, future_dow_idx, y0=None):
        B, T, _ = x.size()
        last_dow_emb = self.dow_emb(last_dow_idx).unsqueeze(1).repeat(1, T, 1)
        x = torch.cat([x, last_dow_emb], dim=-1)

        _, (h, c) = self.encoder(x)
        hx, cx = h[-1], c[-1]

        y_prev = torch.zeros(B, 1, device=x.device) if y0 is None else y0.view(B, 1)
        outs = []
        for t in range(self.horizon):
            dow_emb = self.dow_emb(future_dow_idx[:, t])
            dec_in = torch.cat([y_prev, dow_emb], dim=-1)
            hx, cx = self.dec_cell(dec_in, (hx, cx))
            y_t = self.out(hx) 
            outs.append(y_t)
            y_prev = y_t
        return torch.cat(outs, dim=1)  # (B, H)

# ===== Load data =====
df = pd.read_csv("train.csv")
df["date"] = pd.to_datetime(df["date"])
df["dow"]  = df["date"].dt.dayofweek
df = df.sort_values(["store", "item", "date"])

results = []

# You can adjust the ranges below.
for store_id in range(1, 2):
    for item_id in range(1, 51):
        g_full = df[(df["store"] == store_id) & (df["item"] == item_id)].copy()
        if len(g_full) < WINDOW_SIZE + HORIZON + 30:
            print(f"⏭️ Skip S{store_id} I{item_id} — not enough data")
            continue

        g_full = g_full.sort_values("date").reset_index(drop=True)
        sales_all = g_full["sales"].values.astype(float)
        dow_all   = g_full["dow"].values.astype(int)

        # Split: last HORIZON for forecast; the rest for training.
        T_total = len(g_full)
        T_train = T_total - HORIZON
        sales_tr = sales_all[:T_train]
        dow_tr   = dow_all[:T_train]

        # Weekly profile, trend and strength from training segment only.
        week_prof_tr = weekly_profile(g_full.iloc[:T_train])
        strength  = seasonality_strength(sales_tr, dow_tr, week_prof_tr)
        lam_base  = float(np.clip(0.2 + 0.6 * strength, 0.2, 0.9))
        trend_base_tr = stable_trend_baseline(sales_tr, k=30)

        # Residuals using training parameters.
        season_all = week_prof_tr[dow_all]
        residual_all = sales_all - lam_base * season_all - trend_base_tr

        # Features: rolling + lags (fit scalers on training segment).
        roll_feat_all = add_rolling_on_residual(residual_all)
        lag_feat_all  = lag_feats_on_residual(residual_all)

        roll_scaler = StandardScaler().fit(roll_feat_all[:T_train])
        lag_scaler  = StandardScaler().fit(lag_feat_all[:T_train])
        res_scaler  = StandardScaler().fit(residual_all[:T_train].reshape(-1, 1))

        roll_scaled_all = roll_scaler.transform(roll_feat_all)
        lag_scaled_all  = lag_scaler.transform(lag_feat_all)
        resid_scaled_all= res_scaler.transform(residual_all.reshape(-1, 1))[:, 0]

        # 6-D input features.
        X_feat_all = np.concatenate([
            resid_scaled_all.reshape(-1, 1),
            roll_scaled_all, lag_scaled_all
        ], axis=1)

        # Supervised samples from training segment.
        Xs, Ys, last_dow_list, fut_dow_list = [], [], [], []
        jitter_low, jitter_high = max(0.0, lam_base * 0.7), min(1.0, lam_base * 1.2)

        for i in range(T_train - WINDOW_SIZE - HORIZON + 1):
            Xs.append(X_feat_all[i:i + WINDOW_SIZE])

            lam_train = np.random.uniform(jitter_low, jitter_high)
            season_curr = week_prof_tr[dow_all]
            resid_true = sales_all - lam_train * season_curr - trend_base_tr
            y_raw = resid_true[i + WINDOW_SIZE : i + WINDOW_SIZE + HORIZON]
            y_scaled = res_scaler.transform(y_raw.reshape(-1, 1))[:, 0]
            Ys.append(y_scaled)

            last_dow = dow_all[i + WINDOW_SIZE - 1]
            fut_dow  = [(last_dow + k + 1) % 7 for k in range(HORIZON)]
            last_dow_list.append(last_dow)
            fut_dow_list.append(fut_dow)

        if len(Xs) == 0:
            # Fallback: baseline only.
            last_dow = dow_all[T_train - 1]
            future_dow = [(last_dow + k + 1) % 7 for k in range(HORIZON)]
            season_future = lam_base * week_prof_tr[future_dow]
            trend_future  = np.full(HORIZON, stable_trend_baseline(sales_tr, k=30), dtype=float)
            preds = np.clip(trend_future + season_future, 0, None)
        else:
            X_np = np.asarray(Xs, np.float32)
            Y_np = np.asarray(Ys, np.float32)
            last_dow_np = np.asarray(last_dow_list, np.int64)
            fut_dow_np  = np.asarray(fut_dow_list, np.int64)

            ds = SeqDS(X_np, Y_np, last_dow_np, fut_dow_np)
            dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

            model = ResidualAR(
                input_size=X_np.shape[2], hidden=HIDDEN_SIZE,
                layers=NUM_LAYERS, dropout=DROPOUT, horizon=HORIZON, dow_dim=DOW_DIM
            ).to(device)

            opt  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            crit = nn.HuberLoss(delta=1.0)

            for ep in range(1, EPOCHS + 1):
                model.train()
                loss_sum = 0.0
                for xb, yb, ldb, fdb in dl:
                    xb, yb = xb.to(device), yb.to(device)
                    ldb, fdb = ldb.to(device), fdb.to(device)
                    pred = model(xb, ldb, fdb)
                    loss = crit(pred, yb)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    loss_sum += loss.item() * xb.size(0)
                if ep % 10 == 0:
                    print(f"[S{store_id:02d} I{item_id:02d}] Epoch {ep:03d}/{EPOCHS}  Huber={loss_sum/len(ds):.4f}")

            # Inference on the last training window.
            start_last = T_train - WINDOW_SIZE
            X_last = X_feat_all[start_last : start_last + WINDOW_SIZE][None, ...].astype(np.float32)
            last_dow_last = np.array([dow_all[start_last + WINDOW_SIZE - 1]], np.int64)
            future_dow_last = np.array([[(last_dow_last[0] + k + 1) % 7 for k in range(HORIZON)]], np.int64)

            model.eval()
            with torch.no_grad():
                y_res_scaled = model(
                    torch.from_numpy(X_last).to(device),
                    torch.from_numpy(last_dow_last).to(device),
                    torch.from_numpy(future_dow_last).to(device)
                ).cpu().numpy()[0]
            y_res = res_scaler.inverse_transform(y_res_scaled.reshape(-1, 1))[:, 0]

            trend_future = np.full(HORIZON, stable_trend_baseline(sales_tr, k=30), dtype=float)
            season_future = lam_base * week_prof_tr[future_dow_last[0]]
            preds = np.clip(trend_future + season_future + y_res, 0, None)

        for d, p in enumerate(preds, 1):
            results.append({"store": store_id, "item": item_id, "day": d,
                            "predicted_sales": float(np.round(p, 2))})

# ===== Plot one example =====
ex_s, ex_i = 1, 1
hist = df[(df["store"] == ex_s) & (df["item"] == ex_i)].sort_values("date")
recent_30 = hist["sales"].values[-30:]
ex_pred = [r["predicted_sales"] for r in results
           if r["store"] == ex_s and r["item"] == ex_i][:HORIZON]

plt.figure(figsize=(12, 5))
plt.plot(range(30), recent_30, label="Recent 30d (Actual)")
plt.plot(range(30, 30 + len(ex_pred)), ex_pred, label="Forecast 90d")
plt.title(f"Store {ex_s}, Item {ex_i} — 30d Actual + 90d Forecast")
plt.xlabel("Day"); plt.ylabel("Sales"); plt.legend(); plt.tight_layout(); plt.show()

