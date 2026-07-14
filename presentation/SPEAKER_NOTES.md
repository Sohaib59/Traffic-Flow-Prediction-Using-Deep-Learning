# Speaker Notes — Traffic Flow Prediction (5–10 min)

**Slide 1 (Title, ~20s)** — Introduce the task: predicting short-horizon highway speed with three
deep learning architectures on METR-LA, and that you'll walk through preprocessing → modeling →
results → why the winner won.

**Slide 2 (Problem & Dataset, ~1 min)** — Explain: METR-LA has 207 real sensors, 5-min cadence, but
the full 34k-row file needed Google Drive access you didn't have, so you used a real 1,025-row slice
of the same official dataset — say this proactively, don't wait to be asked. Clarify you're predicting
*speed*, not raw flow, because that's what METR-LA measures and what the entire published literature on
this dataset (DCRNN, Graph WaveNet, GMAN) forecasts — congestion shows up as a speed drop.

**Slide 3 (EDA, ~1 min)** — Point to the hourly chart: commute-hour dips are visible. Mention the 0.0
mph sentinel-missing values (~2.8%) had to be identified and imputed, not treated as real data — this
is the kind of domain knowledge that separates a correct pipeline from a naive one.

**Slide 4 (Preprocessing & Features, ~1.5 min)** — Walk the two columns left to right: imputation →
outlier clipping → calendar features (cyclical hour encoding, explain *why* sin/cos avoids the 23→0
discontinuity) → lag features → rolling stats (computed on shift(1) to avoid leakage) → sequence
windowing (60-min lookback, 5-min-ahead target).

**Slide 5 (Models, ~1 min)** — One line per architecture: LSTM's 3 gates, GRU's 2 gates (fewer
params), Transformer's self-attention (no recurrent bias, positional encoding added manually).

**Slide 6 (Hyperparameter Tuning, ~1 min)** — Explain the grid was deliberately narrow (4 configs ×
2 models) because ~700 training sequences would let an exhaustive search overfit the *validation* set
too. Point to the loss curves: GRU converges to a lower, more stable plateau than LSTM.

**Slide 7 (Results, ~1.5 min)** — Read the table: GRU wins on every metric (MAE 0.34, RMSE 0.45, R²
0.90) vs LSTM (R² 0.69) and Transformer (R² 0.53). Point to the bar chart for the visual gap.

**Slide 8 (Why GRU wins, ~1.5 min)** — This is the "explain your reasoning" slide — the one most
likely to get follow-up questions. Key points, in order: (1) fewer parameters than LSTM at same hidden
size → less overfitting on 700 samples, (2) LSTM's long-range gate is underused by a 60-min lookback,
(3) Transformer's lack of recurrent inductive bias needs more data to overcome, (4) this ranking is
expected to invert on the full 34k-row dataset — be ready to defend this as a data-volume argument, not
an architecture-quality claim.

**Slide 9 (Conclusion, ~30s)** — Bottom line + bonus (multi-step forecasting) + future work (full
dataset, graph neural network for spatial structure, Streamlit dashboard). End on the GNN point — it
signals you know where this project sits relative to the actual published state of the art on METR-LA.

## Anticipated questions
- *"Why not use the full dataset?"* → Network sandboxing; the official file is Google-Drive-hosted
  with an API key requirement not available in this environment. Code is written to scale to it directly.
- *"Why network-average speed instead of per-sensor?"* → Keeps the problem a clean, explainable single
  time series; per-sensor/graph modeling is explicitly named as the next step, matching how DCRNN/Graph
  WaveNet actually approach this dataset.
- *"Why not use recursive multi-step forecasting?"* → Direct multi-step (separate output head per
  horizon) is simpler and was sufficient to demonstrate the extension; recursive/seq2seq is named as
  future work for tighter long-horizon accuracy.
