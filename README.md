# NaviGuard: Clock-Bias Forecasting and Anomaly Monitoring for NavIC Satellites
**Dept. of AI & DS, BMS College of Engineering | 2025–26**

> A reproducible benchmark and monitoring service for **satellite clock-bias forecasting** and
> **clock-anomaly detection**, built on **real NavIC broadcast clock data** and evaluated honestly
> against physical and classical baselines.

## What this project is (and is not)

- **Forecasting.** Predict a satellite's clock bias 1.5 h to 24 h ahead from its recent broadcast clock
  history. Compared against persistence, broadcast-drift extrapolation, linear extrapolation, ARIMA,
  ridge regression, and LSTM / GRU / attention-LSTM networks, using leakage-free rolling-origin evaluation.
- **Anomaly monitoring.** A residual-based detector, evaluated on injected faults in real series and
  checked against independent broadcast metadata.
- **Service.** A FastAPI service (and Streamlit dashboard) exposing forecasts, evaluation, and anomaly
  reports, optionally narrated by a local LLM.
- **Not** a precise-clock product. The real data is the clock each satellite **broadcasts** (a fitted
  polynomial from the ground segment); no public precise NavIC clock product exists (the IGS MGEX
  products do not include IRNSS). Results describe forecasting of that broadcast clock.
- Ephemeris/orbit error is **not** forecast. Earlier versions carried a synthetic ephemeris feature; the
  real broadcast data has no such column.

---

## Key findings

Real data: 68 days (15 Jul – 21 Sep 2026) of broadcast clocks for three NavIC satellites (I02: 6,633
samples, I09: 2,379, I10: 2,699; ~912 s cadence). MAE in nanoseconds, walk-forward evaluation over 3
folds; neural models are averaged over 3 seeds (± = seed std).

**1. The satellite's own broadcast drift is a very strong baseline.** Averaged over satellites,
1.5 h ahead (6 steps):

| model | step 1 | step 3 | step 6 |
|---|---|---|---|
| persistence | 48.3 | 143.7 | 286.6 |
| **broadcast drift (af1)** | **4.2** | **10.4** | **20.0** |
| ARIMA(p,1,0) | 6.3 | 15.2 | 29.7 |
| ridge | 6.7 | 13.2 | 23.5 |
| LSTM | 11.1 ± 2.9 | 33.3 ± 5.3 | 58.3 ± 23.1 |
| attention-LSTM | 10.3 ± 1.8 | 31.0 ± 16.9 | 67.8 ± 22.8 |

**2. On the cleanest satellite, attention helps — and the benefit grows with horizon.** On I02 (long,
gap-free series) the attention-LSTM is best at steps 1, 12 and 24 of the 6 h (24-step) horizon and the seed
spread is tiny, while ridge/broadcast-drift are next:

| I02, 24-step horizon | step 1 | step 12 | step 24 |
|---|---|---|---|
| broadcast drift | 2.72 | 20.10 | 35.29 |
| ridge | 2.68 | 16.92 | 31.96 |
| LSTM | 2.73 ± 0.01 | 18.85 ± 0.28 | 30.85 ± 3.04 |
| **attention-LSTM** | **2.51 ± 0.06** | **15.70 ± 0.04** | **23.08 ± 0.22** |

At the 1.5 h horizon the attention-LSTM and ridge are statistically indistinguishable on I02
(Diebold–Mariano p ≈ 0.25).

**3. Neural models fail on short, gappy series.** On I09 and I10 (≈2.5 k samples with many gaps) the
learned models are roughly 2–4× worse than broadcast drift and vary a lot across seeds, and attention
is not uniformly better than a plain LSTM. Any claim that learned models "win" is satellite-specific.

**4. Broadcast-clock residuals are heavy-tailed.** One-step residuals have excess kurtosis of 80–96.
Even in the *nominal* training period 3.8% (I02), 11.2% (I09) and 29.6% (I10) of samples exceed 4σ,
because ground-segment refits create frequent discontinuities. A Gaussian z-score threshold therefore
produces many alerts. On injected faults the detectors reach ≥0.99 recall at ≥10σ, but precision is low
(best F1 ≈ 0.25–0.35; ~35–50 alarms per 1000 clean held-out windows for the z-score/CUSUM detectors, ~3 for
Isolation Forest). Flagged real events did **not** coincide with URA-index/health changes (lift 0.5× on
I02, 1.7× on I09 from only 33 flags), so they cannot be labelled as uploads. Separating natural refits
from true faults is the main open problem.

**Caveats.** Neural models are not tuned; the folds share data so significance tests are optimistic;
the satellite average is dominated by the two hard satellites; NavIC coverage varies day to day (I06
appears only in January data).

Reproduce everything:
```bash
naviguard fetch --start 2026-07-15 --end 2026-09-21
naviguard benchmark --seeds 3 --epochs 40 --folds 3 --horizon 6 --out-dir outputs/benchmark_h6
naviguard benchmark --seeds 3 --epochs 40 --folds 3 --seq-len 48 --horizon 24 --out-dir outputs/benchmark_h24
naviguard anomaly-eval
naviguard report --bench "horizon 6 (1.5 h)=outputs/benchmark_h6" --bench "horizon 24 (6 h)=outputs/benchmark_h24"
```

---

## Project Structure

```
NaviGuard/
├── data/
│   ├── satellite_telemetry.csv        # synthetic telemetry (regenerable)
│   ├── navic_telemetry.csv            # real NavIC broadcast clocks (from `naviguard fetch`, gitignored)
│   └── raw/                           # per-day extracted IRNSS records (gitignored)
├── models/                            # generated: synthetic model + models/navic/ (real-data model)
├── outputs/                           # generated: benchmark/, anomaly/, figures/, prediction plot
├── src/naviguard/
│   ├── config.py                      # paths, split fractions, profiles, service/LLM settings
│   ├── cli.py                         # `naviguard <subcommand>` entry point
│   ├── data/
│   │   ├── generate.py                # synthetic telemetry (+ labelled anomaly injection)
│   │   └── broadcast.py               # real NavIC data: download + RINEX parse (DLR BRDM via BKG)
│   ├── preprocessing/sequences.py     # scaling, per-satellite chronological split, windowing (synthetic profile)
│   ├── models/                        # attention-LSTM layers + training (synthetic profile)
│   ├── baselines.py                   # persistence / linear / ridge (synthetic profile)
│   ├── anomaly/detect.py              # robust-z residual detector
│   ├── benchmark/
│   │   ├── windows.py                 # level-free windows, purged rolling-origin folds
│   │   ├── models.py                  # persistence, broadcast-drift, linear, ARIMA, ridge, LSTM, GRU, attn-LSTM
│   │   ├── stats.py                   # Diebold–Mariano test, block bootstrap
│   │   ├── run.py                     # per-satellite and pooled benchmarks (resumable)
│   │   ├── anomaly_eval.py            # injected-fault + real-event detector evaluation
│   │   └── figures.py                 # paper figures and LaTeX tables
│   ├── inference/
│   │   ├── artifacts.py               # thread-safe model cache (profile-aware)
│   │   ├── predict.py                 # evaluation, forecasting, anomaly scan (synthetic profile; dispatches to navic)
│   │   └── navic.py                   # real-data profile: train/select/persist/evaluate/forecast
│   ├── llm/                           # LM Studio client + operator-report generation
│   └── api/                           # FastAPI service
├── frontend/dashboard.py              # Streamlit dashboard (HTTP-only)
├── scripts/run_pipeline.py            # wrapper for `naviguard pipeline`
├── tests/                             # pytest suite (hermetic)
└── architecture_diagram.py
```

---

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows; `source .venv/bin/activate` on macOS/Linux
pip install -e ".[dev]"
```

### A. Real NavIC data (the main path)
```bash
naviguard fetch --days 60              # download + extract per-satellite broadcast clocks (~1.4 MB/day, raw deleted)
naviguard train-navic                  # trains 3 candidates, serves the lowest *validation* MAE
NAVIGUARD_PROFILE=navic naviguard serve        # PowerShell: $env:NAVIGUARD_PROFILE="navic"; naviguard serve
```

### B. Synthetic demo (no downloads)
```bash
python scripts/run_pipeline.py         # generate -> preprocess -> train -> predict
naviguard serve
```

Docs at `http://127.0.0.1:8000/docs`. Dashboard (optional):
```bash
pip install -e ".[frontend]"
streamlit run frontend/dashboard.py    # http://localhost:8501; set NAVIGUARD_API_URL if the API is elsewhere
```

### CLI reference

| command | purpose |
|---|---|
| `generate` | synthetic telemetry (`--n-satellites`, `--anomaly-count` injects labelled faults in the test region) |
| `preprocess` / `train` / `predict` | synthetic-profile pipeline (70/15/15 split, val-based early stopping, baselines, anomaly scan) |
| `pipeline` | generate → preprocess → train → predict |
| `fetch` | download real NavIC broadcast clocks (`--start/--end` or `--days`, `--min-records`, `--keep-raw`) |
| `train-navic` | train + select a model on real data for the `navic` profile |
| `benchmark` | rolling-origin benchmark (`--horizon`, `--seq-len`, `--folds`, `--seeds`, `--models`, `--pooled`, `--resume`) |
| `anomaly-eval` | detector evaluation: injected faults + real events vs metadata |
| `report` | figures and LaTeX tables from benchmark/anomaly outputs |
| `serve` / `clean` | run the API / remove generated artifacts |

---

## Data

**Real (navic profile).** The daily multi-GNSS broadcast-ephemeris product `BRDM00DLR` (DLR/GSOC, RINEX
3.04) mirrored by BKG's IGS archive (public, ~1.4 MB gzipped per day). Each IRNSS record carries a clock
polynomial (af0, af1, af2) at its reference epoch; continuously tracked satellites have a record about
every 15 min (912 s), so af0 at each epoch is a per-satellite clock-bias series. `naviguard fetch` also
keeps the drift (af1), URA index, health flag and TGD. Only the extracted IRNSS rows are cached
(~1 MB for 68 days). Downloads have a hard per-file deadline and retries; failed days are skipped and
reported.

**Synthetic.** 24 h clock-bias sinusoid + random-walk wander + noise, cosine drift, 12 h ephemeris
sinusoid; optional labelled anomalies (spike, drift excursion, ephemeris jump) in the test region.

**Level-free windows.** Real bias sits at hundreds of µs with ns-scale dynamics, so windows are expressed
relative to the last observed bias (bias − last, broadcast drift × step, first difference), and
targets are the change from that last value. Windows never span data gaps. This makes satellites
comparable and enables cross-satellite pooling.

**Evaluation protocol.** Chronological per satellite; windows are assigned to a split by where their
*targets* lie and any straddling a boundary are dropped, so no training target overlaps validation/test.
The scaler (synthetic profile) is fit on training rows only. Benchmarks use purged, expanding-window
rolling-origin folds with the newest 15% of each training set held out for early stopping.

---

## Models

Neural forecasters: 2-layer LSTM (64→32) or GRU, dropout 0.2, Dense(16), multi-step Dense head, Adam
1e-3, MSE. The **attention-LSTM** replaces the last hidden state with additive (Bahdanau-style)
attention pooling over per-timestep outputs. The synthetic-profile model also adds `LastValueSkip`, so
the network predicts the change from the last observed bias.

The `navic` profile trains broadcast-drift, ridge and attention-LSTM pooled across satellites and serves
the one with the lowest **validation** MAE; test metrics for every candidate are recorded in
`models/navic/model_meta.json` but never used for selection. (On current data it selects ridge although
broadcast drift is marginally better on test — selection is by validation.)

---

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Always 200: model/telemetry present, active `profile`; `?check_llm=true` probes LM Studio |
| GET | `/model/info` | Model metadata: test MAE, hyperparameters, candidates and satellites (navic) |
| GET | `/telemetry` | Latest rows; `satellite_id` filter; lists available `satellites` |
| GET | `/predict/evaluate` | Test-split MAE/RMSE per step, actual/predicted/residual series, baseline comparison, `skill_vs_persistence`; `satellite_id` (navic) |
| POST | `/predict` | Forecast from a raw window or the latest rows of `satellite_id` (navic windows are `[clock_bias_s, clock_drift_s_per_s]` rows) |
| POST | `/anomaly-report` | Numeric evaluation + residual anomaly detection (deterministic `nominal/watch/anomalous`, optional `z_threshold`, `satellite_id`), optionally narrated by a local LLM |

Response changes since v0.2 are additive. Error handling: missing model/telemetry → `503`, bad input
(wrong window shape, unknown satellite, data gap in the latest window) → `422`.

**Configuration (environment)**

| variable | effect |
|---|---|
| `NAVIGUARD_PROFILE` | `synthetic` (default) or `navic` |
| `NAVIGUARD_API_KEY` | if set, every endpoint except `/health` requires an `X-API-Key` header (the dashboard sends it from the same variable) |
| `NAVIGUARD_CORS_ORIGINS` | JSON list of allowed browser origins (defaults to local Streamlit/React/Vite ports) |
| `NAVIGUARD_API_URL` | where the dashboard finds the API |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT_S` | local LLM (LM Studio) settings |

**Anomaly detector.** Step-1 residuals on the validation split are calibrated to a robust median/MAD
baseline; test residuals are scored as robust z and flagged above a threshold. In the `navic` profile the
threshold is the larger of z = 4 and the 99.5th percentile of validation |z| (see finding 4), calibration
is per satellite, and flags are candidates for review, not confirmed faults.

**Local LLM (optional).** `POST /anomaly-report` can ask a local model via [LM Studio](https://lmstudio.ai)
to narrate the already-computed statistics and give a severity second opinion. It never sees raw
telemetry or forecasts anything. If LM Studio is not running the endpoint still returns `200` with the
numeric analysis and an explanatory `llm_status`.

---

## Dashboard

`frontend/dashboard.py` is a Streamlit dashboard (visual design by Pratyush Narain) that talks to the API
over HTTP only. It adapts to the API's profile: satellite selector, telemetry, actual-vs-predicted,
baseline comparison with skill vs persistence, per-satellite forecast, residual anomaly detector, and the
optional LLM report. It shows friendly banners for an unreachable API, timeouts, an untrained model, and a
missing/invalid API key.

---

## Tests and CI

```bash
pytest -q
ruff check src tests scripts
```

The suite (~85 tests) is hermetic: no downloads, trained models, or LM Studio needed. It covers
leakage-free splitting, the RINEX parser, fold purging, model shapes, the Diebold–Mariano test, the
anomaly evaluation, the navic train/save/load round trip, and the API in both profiles. GitHub Actions
(`.github/workflows/ci.yml`) runs both commands on every push and pull request.

---

## Related work

Clock-bias prediction with recurrent networks is an active area (verify details before citing):
1. Huang B., Ji Z. (2021) — SL-LSTM for GPS clock bias, *GPS Solutions*
2. He S., Liu J. (2023) — LSTM for BDS-3 clock prediction, *GPS Solutions*
3. Cai C., Liu M. (2024) — LSTM-Attention for BDS, *GPS Solutions*
4. Bhatt A., Mehta I. (2024) — LSTM for Galileo clock bias, *arXiv:2411.07015*

---

## Roadmap

In progress: cross-satellite pooled models and the 24 h (96-step) horizon. Next: separating natural
refit discontinuities from true faults; precise clocks for GPS/Galileo/BeiDou (IGS `.CLK`) as an
external validation set; probabilistic (conformal) forecasts; scheduled data refresh and retraining;
Docker.

---

## Contributors

Siddhant — modelling, benchmark, API, real-data pipeline · Om — preprocessing · Pratyush Narain —
dashboard design.

*Relevant to SDG 9 (Industry, Innovation and Infrastructure): dependable satellite-navigation timing.*
