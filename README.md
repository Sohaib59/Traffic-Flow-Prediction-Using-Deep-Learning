# Traffic Flow Prediction Using Deep Learning

Predicting short-horizon traffic speed on the **METR-LA** dataset (Los Angeles County highway loop
detectors, 207 sensors, 5-minute readings) with LSTM, GRU, and Transformer architectures, compared
head-to-head on the same features, splits, and evaluation metrics.

## Dataset note (read this first)

The full METR-LA release (Mar–Jun 2012, 34,272 timesteps × 207 sensors) is distributed by the original
authors (Li et al., *DCRNN*, ICLR 2018) as a single HDF5 file hosted on Google Drive — not reachable
from a sandboxed / offline environment, and Kaggle's mirrors require API auth that wasn't available
here either. This project uses a **real, unmodified 1,025-timestep (~3.5 day) slice of the same
official 207-sensor dataset**, sourced from a public GitHub mirror (`data/METR-LA.csv`).

Every step of the pipeline — preprocessing, feature engineering, sequence generation, all three
models — is written to run unchanged on the full 34,272-row file: swap the CSV path in the notebook
and re-run. Using the smaller slice here keeps the whole project fast and fully reproducible without a
GPU, at the cost of a smaller train/test set (noted explicitly wherever it affects interpretation of
results below).

## Problem framing

METR-LA records **speed** (mph), not volume. This project predicts speed because that's what the
dataset actually contains, and it's the standard target in the traffic-forecasting literature on this
exact dataset (DCRNN, Graph WaveNet, GMAN, etc. all forecast speed) — via the fundamental traffic-flow
diagram, speed drops are the direct signature of rising congestion/flow breakdown, so a speed forecast
answers the same operational question a flow forecast would.

To keep the project a clean, explainable single-series problem, the target is the **network-wide
average speed** across all 207 sensors at each timestep (a standard simplification of the full 207-node
spatio-temporal problem — modeling all sensors jointly is the graph-neural-network extension this
project points to as future work).

## Repository structure

```
traffic_project/
├── data/
│   └── METR-LA.csv                  # raw sensor data (207 sensors, 5-min cadence)
├── notebooks/
│   ├── traffic_flow_prediction.ipynb  # MAIN DELIVERABLE — full pipeline, executed with outputs
│   └── traffic_flow_prediction.py     # same content in plain-script (jupytext) form, for diffing/review
├── src/
│   └── model_defs.py                 # standalone model classes for clean inference (see below)
├── models/
│   ├── lstm_model.pt / gru_model.pt / transformer_model.pt
│   ├── gru_30min_model.pt / gru_60min_model.pt   # bonus multi-step models
│   └── x_scaler.pkl / y_scaler.pkl               # fitted MinMaxScalers (needed to use the models)
├── outputs/
│   ├── figures/                      # all 8 saved plots (EDA + results)
│   ├── model_comparison_metrics.csv
│   ├── multistep_metrics.csv
│   └── hyperparameter_tuning_results.csv
├── requirements.txt
└── README.md
```

## Methodology

### 1. Data repair
The mirrored CSV's header row was corrupted by a prior export (a data row had been used as the column
header, so pandas auto-suffixed duplicate values). That row is dropped; clean `sensor_000..sensor_206`
columns are rebuilt, and a synthetic 5-minute `DatetimeIndex` is attached starting 2012-03-01 — the
documented real start date and sampling rate of METR-LA.

### 2. EDA
- `0.0 mph` readings (~2.8% of all values) are METR-LA's documented sensor-dropout sentinel, not real
  near-zero speeds — confirmed by their random, non-clustered distribution across sensors and time.
- Speed distribution is left-skewed: most readings sit in the 55–70 mph free-flow band with a heavier
  low-speed tail during congestion.
- A clear diurnal (hour-of-day) pattern is present, motivating calendar features.
- Nearby sensors are highly correlated — expected given they measure the same corridor.

### 3. Preprocessing
- `0.0` → `NaN`, then **time-aware interpolation** (`limit=6`, i.e. up to 30 min of missing data
  bridged) followed by edge-fill for any boundary gaps.
- Physical-domain clipping to `[0, 80]` mph as an outlier guard.

### 4. Feature engineering
- Calendar: `hour_sin` / `hour_cos` (cyclical encoding, avoids the 23→0 discontinuity a raw `hour`
  integer creates), `is_weekend`.
- Lag features: `t-1, t-2, t-3, t-6, t-12` (5–60 min back).
- Rolling statistics: 30-min and 60-min rolling mean/std (computed on `shift(1)` to avoid leakage).

### 5. Sequence generation
60-minute lookback window (12 steps) → single-step-ahead (5 min) target for the main comparison;
6-step (30 min) and 12-step (60 min) horizons for the bonus multi-step section. **Chronological split**
(70/15/15, no shuffling) — standard practice for time series to avoid leaking future information into
training.

### 6. Models (PyTorch)
- **LSTM** — 3-gate recurrent unit, `nn.LSTM` + linear head.
- **GRU** — 2-gate recurrent unit, `nn.GRU` + linear head.
- **Transformer** — learned input projection + sinusoidal positional encoding + `TransformerEncoder`
  (multi-head self-attention) + linear head on the final timestep's representation.

### 7. Hyperparameter tuning
Grid search over `hidden_size ∈ {32, 64, 128}` × `num_layers ∈ {1, 2}` for **LSTM and GRU**, selected
by validation MSE. Kept intentionally narrow (4 configs each) — with ~700 training sequences, an
exhaustive search would tune to validation-set noise more than it would inform architecture choice.

### 8. Evaluation
MAE, MSE, RMSE, MAPE, R² — all computed after inverse-transforming predictions back to real mph units.

## Results

| Model       | MAE (mph) | RMSE (mph) | MAPE (%) | R²   |
|-------------|-----------|------------|----------|------|
| **GRU**     | **0.34**  | **0.45**   | **0.54** | **0.90** |
| LSTM        | 0.64      | 0.78       | 0.99     | 0.69 |
| Transformer | 0.78      | 0.96       | 1.21     | 0.53 |

*(Exact numbers regenerate on each run in `outputs/model_comparison_metrics.csv` — this table is
from the committed run.)*

**GRU wins clearly.** With only ~700 training sequences, GRU's 2-gate design has ~25% fewer parameters
than LSTM's 3-gate design at the same hidden size, so it converges to a lower-variance fit instead of
overfitting — visible directly in the validation-loss curves. LSTM's extra cell-state gate is built for
*longer* dependencies than a 60-minute lookback actually exercises here, so its added capacity mostly
adds estimation noise at this data volume rather than predictive power. The Transformer is weakest for
the same reason in a stronger form: self-attention has no recurrent inductive bias and has to learn
temporal locality purely from data — a disadvantage on ~1k rows that inverts on the full 34k-row
dataset, where Transformers (and graph-attention variants like GMAN) are the published state of the
art. **Practical takeaway:** for short-horizon forecasting on limited data, use GRU; Transformers need
the volume of the full dataset (and ideally a graph-aware spatial component, as in DCRNN/Graph
WaveNet) to earn their extra flexibility.

### Bonus: multi-step forecasting
GRU (the strongest single-step model) extended to direct 30-minute (6-step) and 60-minute (12-step)
horizons — see `outputs/multistep_metrics.csv` and `outputs/figures/08_multistep_forecast.png`.
Error grows and R² drops with horizon length, as expected: uncertainty compounds the further out the
forecast reaches, and a direct (non-recursive) multi-step head has to learn a harder mapping with the
same amount of training data.

## How to run

```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace traffic_flow_prediction.ipynb
# or open traffic_flow_prediction.ipynb in Jupyter Lab and run all cells
```
Runs in well under 5 minutes on CPU — no GPU required at this data scale.

## Loading a saved model for inference

Use `src/model_defs.py` — a standalone module containing just the model class definitions. Don't
import the notebook's `.py` export for this: it's a top-to-bottom script, so importing it would
re-run the entire training pipeline as a side effect.

```python
import sys, torch, joblib, numpy as np
sys.path.insert(0, "src")
from model_defs import GRUForecaster

x_scaler = joblib.load("models/x_scaler.pkl")
y_scaler = joblib.load("models/y_scaler.pkl")

model = GRUForecaster(n_features=13, hidden_size=32, num_layers=1, horizon=1)
model.load_state_dict(torch.load("models/gru_model.pt", map_location="cpu"))
model.eval()

# x: shape (1, 12, 13) -- 12 lookback steps x 13 features, in FEATURE_COLS order
# (avg_speed, hour_sin, hour_cos, is_weekend, lag_1, lag_2, lag_3, lag_6, lag_12,
#  roll_mean_6, roll_std_6, roll_mean_12, roll_std_12), scaled with x_scaler first.
with torch.no_grad():
    pred_scaled = model(torch.tensor(x, dtype=torch.float32)).numpy()
pred_mph = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)
```

This was verified end-to-end (weights load, forward pass runs, output is a plausible mph value).

## Limitations & future work
- **Data volume**: 1,025 timesteps is a fraction of the full 34,272-row dataset; conclusions on
  architecture ranking (esp. Transformer vs. recurrent nets) are expected to shift with more data —
  the code is written to scale directly to the full file.
- **Spatial structure is discarded**: averaging across 207 sensors ignores road-network topology.
  A graph neural network (DCRNN, Graph WaveNet, GMAN) operating on all sensors jointly is the natural
  next step and is what the published state-of-the-art on this exact dataset actually uses.
- **Multi-step forecasting** here is direct (one head per horizon); a recursive or seq2seq decoder
  would likely improve longer-horizon accuracy.
- **Streamlit dashboard** (optional bonus in the brief) was not built in this pass to keep the
  deliverable tight — the saved models/scalers in `models/` are ready to be wrapped in one directly.
