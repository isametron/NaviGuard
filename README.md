# NaviGuard: AI-Driven Satellite Clock Intelligence
**Dept. of AI & DS, BMS College of Engineering | 2025–26**

> An end-to-end attention-LSTM prediction service for NavIC/GNSS satellite clock bias
> and ephemeris error forecasting — enabling proactive correction over reactive post-hoc adjustment.

---

## Project Structure

```
NaviGuard/
├── data/
│   └── satellite_telemetry.csv     # NavIC/GNSS telemetry (synthetic, regenerable)
├── models/                          # Auto-generated after training
│   ├── scaler.pkl                  # Fitted MinMaxScaler
│   ├── lstm_attention_satellite.keras
│   └── model_meta.json             # seq_len, horizon, best_val_loss, trained_at, ...
├── outputs/                         # Auto-generated after `predict --save-plot`
│   └── prediction_plot.png
├── src/naviguard/                   # The package
│   ├── config.py                   # paths, SEQ_LEN, HORIZON, LLM settings
│   ├── cli.py                      # `naviguard <subcommand>` entry point
│   ├── data/generate.py            # configurable synthetic telemetry simulation
│   ├── preprocessing/sequences.py  # scaling + sliding-window sequence construction
│   ├── models/                     # attention-LSTM definition + training
│   ├── inference/                  # evaluation, forecasting, plotting
│   ├── llm/                        # LM Studio client + operator-report generation
│   └── api/                        # FastAPI service (health, predict, anomaly-report)
├── scripts/run_pipeline.py         # convenience: generate -> preprocess -> train -> predict
├── frontend/dashboard.py           # Streamlit dashboard consuming the FastAPI service (optional)
├── tests/                          # pytest suite (hermetic — no trained model/LM Studio needed)
├── architecture_diagram.py         # generates outputs/architecture_diagram.png
├── pyproject.toml
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Create a venv and install the package (editable, with dev/test extras)
python -m venv .venv
.venv\Scripts\activate            # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -e ".[dev]"

# 2. Run the full pipeline in one command
python scripts/run_pipeline.py

# 3. Launch the API
uvicorn naviguard.api.main:app --reload
# -> docs at http://127.0.0.1:8000/docs
```

Or run each stage manually via the `naviguard` CLI:
```bash
naviguard generate                  # Step 0: (re)generate telemetry CSV
naviguard preprocess                # Step 1: scale + build sequences (seq_len=20, horizon=6)
naviguard train                     # Step 2: train the attention-LSTM
naviguard predict --save-plot       # Step 3: evaluate + save plot
naviguard serve                     # Step 4: launch the FastAPI service
naviguard clean                     # (optional) remove generated models/outputs/sequences
```

The system is API-first — the FastAPI service above is the source of truth. An optional
Streamlit dashboard consumes it purely over HTTP (no direct file reads):
```bash
pip install -e ".[frontend]"
streamlit run frontend/dashboard.py
# -> http://localhost:8501  (requires the API running — see step 3 above)
```
A TypeScript/React frontend consuming the same JSON endpoints is a possible later phase.

---

## Model Architecture — Attention-Enhanced LSTM

| Layer  | Type              | Units | Parameters                              |
|--------|-------------------|-------|-------------------------------------------|
| 1      | LSTM              | 64    | return_sequences=True, input=(20,3)      |
| 2      | Dropout           | —     | rate=0.2                                  |
| 3      | LSTM              | 32    | return_sequences=True                    |
| 4      | Dropout           | —     | rate=0.2                                  |
| 5      | AttentionPooling  | 32    | additive (Bahdanau-style) attention pool |
| 6      | Dense             | 16    | activation='relu'                         |
| 7      | Dense             | horizon (6) | Linear output, multi-step forecast  |

- **Optimizer:** Adam (lr=0.001, β₁=0.9, β₂=0.999)
- **Loss:** Mean Squared Error (MSE)
- **Callbacks:** EarlyStopping (patience=10) + ModelCheckpoint + ReduceLROnPlateau

The attention layer learns a per-timestep importance score over the 20-step lookback window
and pools the LSTM outputs accordingly, rather than relying solely on the final hidden state —
consistent with the LSTM-Attention literature (Cai & Liu, 2024) cited below. The output layer
forecasts `horizon` steps ahead (default 6 -> 1.5h at 15-min cadence) instead of a single step.

---

## Key Specs

| Parameter        | Value                                              |
|------------------|-----------------------------------------------------|
| Input features   | clock_bias_s, clock_drift_s_per_s, ephemeris_error_m |
| Sequence length  | 20 steps (5 hours of history)                       |
| Forecast horizon | 6 steps (1.5 hours ahead), configurable              |
| Train/Val split  | 80/20 (chronological, no shuffle)                    |
| Sampling rate    | 15 minutes (900 seconds)                             |
| MAE target       | ≤ 50 nanoseconds on the test split, **step 1 only** — later horizon steps are expected to degrade |
| Output unit      | Seconds → converted to nanoseconds                   |

---

## API

Once trained, `naviguard serve` (or `uvicorn naviguard.api.main:app`) exposes:

| Method | Path               | Description |
|--------|--------------------|-------------|
| GET    | `/health`          | Always 200 — reports whether a model/scaler are present |
| GET    | `/model/info`       | Trained model metadata (503 if not trained yet) |
| GET    | `/predict/evaluate` | Per-horizon-step MAE/RMSE + actual/predicted/residual series (JSON form of the old PNG) |
| POST   | `/predict`          | Forecast `horizon` steps ahead from an optional raw telemetry window |
| POST   | `/anomaly-report`   | Numeric evaluation + threshold check, optionally layered with a local-LLM narrative report and severity assessment |

Interactive docs: `http://127.0.0.1:8000/docs`.

---

## Frontend (Dashboard)

`frontend/dashboard.py` is a Streamlit dashboard (visual design by Pratyush Narain) that talks
to the FastAPI service exclusively over HTTP — telemetry table/charts via `GET /telemetry`,
the actual-vs-predicted chart via `GET /predict/evaluate`, and the local-LLM operator report
+ severity assessment via `POST /anomaly-report`. It degrades gracefully: a friendly banner if
the API isn't running, a different one if a request is just slow (e.g. local LLM generation)
rather than actually unreachable, and a "model not trained yet" banner if artifacts are missing.

Configure the API it points at via `NAVIGUARD_API_URL` (default `http://127.0.0.1:8000`).

---

## Local LLM (LM Studio) Integration

`POST /anomaly-report` can call a local LLM via [LM Studio](https://lmstudio.ai) for two things,
layered on top of — never replacing — the numeric MAE-threshold check:

1. **Operator report** — a plain-English narration of already-computed prediction stats.
2. **Severity second-opinion** — a structured `{"severity": "nominal|watch|anomalous", "reasoning": "..."}` reasoning check.

Setup:
1. Install [LM Studio](https://lmstudio.ai), load a small instruct model (e.g. Llama-3.2-3B-Instruct).
2. Start its local server (Developer tab → Start Server) — default `http://localhost:1234/v1`.
3. Call `POST /anomaly-report` with `{"include_llm": true}`.

If LM Studio isn't running, the endpoint still returns `200` with the full numeric analysis —
`llm_report`/`llm_severity` are `null` and `llm_status` explains what to do. Configure via
`LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT_S` environment variables if needed.

---

## Pipeline Data Flow

```
NavIC/GNSS Source (synthetic)
       ↓  [naviguard generate]
satellite_telemetry.csv
       ↓  [naviguard preprocess]
X_seq.npy + y_seq.npy + scaler.pkl   (seq_len=20, horizon=6)
       ↓  [naviguard train]
lstm_attention_satellite.keras + model_meta.json
       ↓  [naviguard predict]
Per-step MAE/RMSE + prediction_plot.png
       ↓  [FastAPI service]
JSON endpoints  +  optional local-LLM anomaly report (LM Studio)
       ↓  [Streamlit dashboard]  (optional, HTTP-only)
```

---

## Tests

```bash
pytest -q
```

The suite is hermetic — it does not require a trained model, generated telemetry, or a running
LM Studio server. API tests use dependency overrides and fake model/scaler stubs; LLM tests use
a fake client and an unreachable port rather than a live server.

---

## References

Key papers this work builds upon:
1. Huang B., Ji Z. (2021) — SL-LSTM for GPS clock bias, *GPS Solutions*
2. He S., Liu J. (2023) — LSTM for BDS-3 clock prediction, *GPS Solutions*
3. Cai C., Liu M. (2024) — LSTM-Attention for BDS, *GPS Solutions*
4. Bhatt A., Mehta I. (2024) — LSTM for Galileo clock bias, *arXiv:2411.07015*

---

## Roadmap (not yet built)

Real NavIC/RINEX/IGS data ingestion · classical baselines (ARIMA/SARIMA/Prophet) for comparison ·
AWS/PySpark ETL for large-scale ingestion · Supabase persistence for historical predictions ·
Docker + CI · a richer TypeScript/React frontend consuming the same API.

---

*NaviGuard directly addresses SDG 9 (Industry & Innovation), SDG 11 (Sustainable Cities),
and SDG 13 (Climate Action) through AI-augmented satellite navigation reliability.*
