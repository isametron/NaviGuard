# NaviGuard: Clock-Bias Forecasting and Anomaly Monitoring for NavIC Satellites

**Dept. of AI & DS, BMS College of Engineering | 2025–26**

> A reproducible benchmark and monitoring service for **satellite clock-bias forecasting** and
> **clock-anomaly detection**, built on **real NavIC broadcast clock data** and evaluated honestly
> against physical and classical baselines.

## What this project is (and is not)

- **Forecasting.** Predict a satellite's clock bias 1.5 h to 24 h ahead from its recent broadcast clock
  history. Compared against persistence, broadcast-drift extrapolation, linear extrapolation, ARIMA,
  a two-state clock Kalman filter, ridge regression, and LSTM / GRU / attention-LSTM networks, using
  leakage-free rolling-origin evaluation.
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

| model                     | step 1     | step 3      | step 6      |
| ------------------------- | ---------- | ----------- | ----------- |
| persistence               | 48.3       | 143.7       | 286.6       |
| **broadcast drift (af1)** | **4.2**    | **10.4**    | **20.0**    |
| Kalman filter             | 5.2        | 11.1        | 20.4        |
| ARIMA(p,1,0)              | 6.3        | 15.2        | 29.7        |
| ridge                     | 6.7        | 13.2        | 23.5        |
| LSTM                      | 11.1 ± 2.9 | 33.3 ± 5.3  | 58.3 ± 23.1 |
| attention-LSTM            | 10.3 ± 1.8 | 31.0 ± 16.9 | 67.8 ± 22.8 |

A two-state clock Kalman filter is second only to broadcast drift on average. Between uploads the
broadcast clock is an exact polynomial, so af1 is hard to beat at short range. Further ahead, upcoming
uploads matter more than the current slope, and the filter's longer-run drift estimate beats broadcast
drift on I02: 14% lower MAE at 6 h, 38% lower at 24 h. Both 95% block-bootstrap intervals exclude zero.
On the gappy I09/I10, broadcast drift stays best (Kalman is 1–32% worse), but the filter is well ahead
of every learned model there.

**2. On the cleanest satellite, attention helps — and the benefit grows with horizon.** On I02 (long,
gap-free series) the attention-LSTM is best at steps 1, 12 and 24 of the 6 h (24-step) horizon and the seed
spread is tiny, while ridge/broadcast-drift are next:

| I02, 24-step horizon | step 1          | step 12          | step 24          |
| -------------------- | --------------- | ---------------- | ---------------- |
| broadcast drift      | 2.72            | 20.10            | 35.29            |
| Kalman filter        | 2.61            | 17.83            | 30.34            |
| ridge                | 2.68            | 16.92            | 31.96            |
| LSTM                 | 2.73 ± 0.01     | 18.85 ± 0.28     | 30.85 ± 3.04     |
| **attention-LSTM**   | **2.51 ± 0.06** | **15.70 ± 0.04** | **23.08 ± 0.22** |

At the 1.5 h horizon the attention-LSTM and ridge are statistically indistinguishable on I02
(Diebold–Mariano p ≈ 0.25).

**3. Neural models fail on short, gappy series.** On I09 and I10 (≈2.5 k samples with many gaps) the
learned models are roughly 2–4× worse than broadcast drift and vary a lot across seeds, and attention
is not uniformly better than a plain LSTM. Any claim that learned models "win" is satellite-specific.

**4. Broadcast-clock residuals are heavy-tailed.** One-step residuals have excess kurtosis of 80–96.
Even in the _nominal_ training period 3.8% (I02), 11.2% (I09) and 29.6% (I10) of samples exceed 4σ,
because ground-segment refits create frequent discontinuities. A Gaussian z-score threshold therefore
produces many alerts. On injected faults the detectors reach ≥0.99 recall at ≥10σ, but precision is low
(best F1 ≈ 0.25–0.35; ~35–50 alarms per 1000 clean held-out windows for the z-score/CUSUM detectors, ~3 for
Isolation Forest). Flagged real events did **not** coincide with URA-index/health changes (lift 0.5× on
I02, 1.7× on I09 from only 33 flags), so they cannot be labelled as uploads. Most natural jumps are
isolated level shifts from routine refits (96% / 66% / 87% of |z| > 4 jumps on I02 / I09 / I10). A
refit-aware detector that alarms only on spikes and sustained excursions cuts false alarms to 2.3 per
1000 (F1 0.77–0.82 on spikes ≥ 10σ, 0.89 on ramps ≥ 5σ), but by design misses persistent step faults,
which look identical to refits.

**Caveats.** Neural models are not tuned; the folds share data so significance tests are optimistic;
the satellite average is dominated by the two hard satellites; NavIC coverage varies day to day (I06
appears only in January data).

Reproduce everything:

```bash
naviguard fetch --start 2026-07-15 --end 2026-09-21
naviguard benchmark --seeds 3 --epochs 40 --folds 3 --horizon 6 --out-dir outputs/benchmark_h6
naviguard benchmark --seeds 3 --epochs 40 --folds 3 --seq-len 48 --horizon 24 --out-dir outputs/benchmark_h24
naviguard benchmark --seeds 3 --epochs 40 --folds 3 --seq-len 96 --horizon 96 --min-windows 150 --out-dir outputs/benchmark_h96
# add a classical model to a finished run without retraining the neural models (same --seq-len/--horizon/--folds)
naviguard benchmark --extend --models kalman --seq-len 48 --horizon 24 --out-dir outputs/benchmark_h24
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
│   │   └── broadcast.py               # real NavIC data: download + RINEX parse (DLR BRDM via BKG or ESA)
│   ├── preprocessing/sequences.py     # scaling, per-satellite chronological split, windowing (synthetic profile)
│   ├── models/                        # attention-LSTM layers + training (synthetic profile)
│   ├── baselines.py                   # persistence / linear / ridge (synthetic profile)
│   ├── anomaly/detect.py              # robust-z residual detector
│   ├── benchmark/
│   │   ├── windows.py                 # level-free windows, purged rolling-origin folds
│   │   ├── models.py                  # persistence, broadcast-drift, linear, ARIMA, Kalman, ridge, LSTM, GRU, attn-LSTM
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

### C. Docker

One image runs both services (`docker-compose.yml`); `data/` and `models/` are shared with your
machine, so existing data and a trained model are used as they are.

```bash
docker compose build
docker compose up                      # API :8000, dashboard :8501
```

LM Studio stays on your machine; the containers reach it at `host.docker.internal:1234`. The compose file
passes `NAVIGUARD_PROFILE` (default `navic`), `LLM_MODEL` (default `qwen2.5-7b-instruct`),
`LLM_TEMPERATURE` (default 0.3) and `NAVIGUARD_SOURCE` through from your environment. Without data or a
model yet, run `docker compose run --rm api naviguard fetch --start 2026-07-15 --end 2026-09-21` and
then `... naviguard train-navic` once.

### CLI reference

| command                            | purpose                                                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `generate`                         | synthetic telemetry (`--n-satellites`, `--anomaly-count` injects labelled faults in the test region)          |
| `preprocess` / `train` / `predict` | synthetic-profile pipeline (70/15/15 split, val-based early stopping, baselines, anomaly scan)                |
| `pipeline`                         | generate → preprocess → train → predict                                                                       |
| `fetch`                            | download real NavIC broadcast clocks (`--start/--end` or `--days`, `--min-records`, `--keep-raw`, `--source`) |
| `coverage`                         | per-satellite sample/gap summary and gap-aware eligibility by forecast horizon                                |
| `train-navic`                      | train + select a model on real data for the `navic` profile                                                   |
| `benchmark`                        | rolling-origin benchmark (`--horizon`, `--seq-len`, `--folds`, `--seeds`, `--models`, `--pooled`, `--resume`, `--extend`, `--params`) |
| `tune`                             | settings search for ridge and the neural models, using only data before the first test block (`--trials`, `--resume`) |
| `anomaly-eval`                     | detector evaluation: injected faults + real events vs metadata                                                |
| `report`                           | figures and LaTeX tables from benchmark/anomaly outputs                                                       |
| `serve` / `clean`                  | run the API / remove generated artifacts                                                                      |

---

## Data

### Six-month telemetry checklist

- [ ] Fetch the inclusive date range into a dedicated CSV:
      `naviguard fetch --start 2026-04-01 --end 2026-09-21 --out data/navic_telemetry_6mo.csv`
- [ ] Generate the per-satellite coverage and gap report:
      `naviguard coverage --csv data/navic_telemetry_6mo.csv --out-dir outputs/coverage`
- [ ] Review `outputs/coverage/coverage.md` and the detailed `outputs/coverage/coverage.csv`.

The coverage report counts a gap when the interval exceeds 1.5× the satellite's median cadence.
Horizon qualification requires at least 200 complete gap-free windows, using the configured
sequence length (default 20 samples) and the forecast horizons 1, 6, 12, and 24 samples.

**Real (navic profile).** The daily multi-GNSS broadcast-ephemeris product `BRDM00DLR` (DLR/GSOC, RINEX
3.04) mirrored by BKG's IGS archive (public, ~1.4 MB gzipped per day). Each IRNSS record carries a clock
polynomial (af0, af1, af2) at its reference epoch; continuously tracked satellites have a record about
every 15 min (912 s), so af0 at each epoch is a per-satellite clock-bias series. `naviguard fetch` also
keeps the drift (af1), URA index, health flag and TGD. Only the extracted IRNSS rows are cached
(~1 MB for 68 days). Downloads have a hard per-file deadline and retries; failed days are skipped and
reported.

The same file is mirrored by ESA's GNSS data centre, with identical records. `--source` (or the
`NAVIGUARD_SOURCE` environment variable) picks the mirror: `bkg`, `esa`, or `auto` (the default), which
tries BKG and, once BKG is unreachable, uses ESA first for the rest of the run. The fetch log reports
how many days came from each mirror.

**Synthetic.** 24 h clock-bias sinusoid + random-walk wander + noise, cosine drift, 12 h ephemeris
sinusoid; optional labelled anomalies (spike, drift excursion, ephemeris jump) in the test region.

**Level-free windows.** Real bias sits at hundreds of µs with ns-scale dynamics, so windows are expressed
relative to the last observed bias (bias − last, broadcast drift × step, first difference), and
targets are the change from that last value. Windows never span data gaps. This makes satellites
comparable and enables cross-satellite pooling.

**Evaluation protocol.** Chronological per satellite; windows are assigned to a split by where their
_targets_ lie and any straddling a boundary are dropped, so no training target overlaps validation/test.
The scaler (synthetic profile) is fit on training rows only. Benchmarks use purged, expanding-window
rolling-origin folds with the newest 15% of each training set held out for early stopping.

---

## Models

Neural forecasters: 2-layer LSTM (64→32) or GRU, dropout 0.2, Dense(16), multi-step Dense head, Adam
1e-3, MSE. The **attention-LSTM** replaces the last hidden state with additive (Bahdanau-style)
attention pooling over per-timestep outputs. The synthetic-profile model also adds `LastValueSkip`, so
the network predicts the change from the last observed bias.

Kalman filter: the standard two-state clock model. It tracks bias and drift, with white-FM and
random-walk-FM process noise, and measures the bias at every record, with noise floored at the 2⁻³¹ s
af0 resolution. The broadcast drift enters once, as a drift measurement at the forecast origin.
Feeding it in at every record would be wrong: af1 only changes at uploads, and each repeat would
count as fresh evidence. The four variances are fitted by maximum likelihood on each fold's training
windows.

Tuning (`naviguard tune`): the benchmark's folds expand, so a later fold's training data is an earlier
fold's test block. Tuning therefore uses only the windows before the first test block, runs the same
rolling-origin scheme inside them, and scores each candidate on those held-out inner blocks (MAE
relative to persistence, averaged over folds and satellites). Neural models get a random search over
layer sizes, dropout, learning rate, batch size, dense units and look-back (the untuned default is
always trial 0); ridge gets a grid over its regularisation strength. Settings are chosen per
satellite, since each satellite's models are trained separately; averaging instead would let the
satellite with the largest errors (I10) pick settings that hurt the clean one (I02). A setting
averaged over satellites is kept for pooled models. `naviguard benchmark --params best.json` then
runs the benchmark with the chosen settings. The untuned defaults reproduce the published results
exactly.

The `navic` profile trains broadcast-drift, ridge and attention-LSTM pooled across satellites and serves
the one with the lowest **validation** MAE; test metrics for every candidate are recorded in
`models/navic/model_meta.json` but never used for selection. (On current data it selects ridge although
broadcast drift is marginally better on test — selection is by validation.)

---

## API

| Method | Path                | Description                                                                                                                                                           |
| ------ | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GET    | `/health`           | Always 200: model/telemetry present, active `profile`; `?check_llm=true` probes LM Studio                                                                             |
| GET    | `/model/info`       | Model metadata: test MAE, hyperparameters, candidates and satellites (navic)                                                                                          |
| GET    | `/telemetry`        | Latest rows; `satellite_id` filter; lists available `satellites`                                                                                                      |
| GET    | `/predict/evaluate` | Test-split MAE/RMSE per step, actual/predicted/residual series, baseline comparison, `skill_vs_persistence`; `satellite_id` (navic)                                   |
| POST   | `/predict`          | Forecast from a raw window or the latest rows of `satellite_id` (navic windows are `[clock_bias_s, clock_drift_s_per_s]` rows)                                        |
| POST   | `/anomaly-report`   | Numeric evaluation + residual anomaly detection (deterministic `nominal/watch/anomalous`, optional `z_threshold`, `satellite_id`; refit-aware verdict in `detection.refit_aware` for navic), optionally narrated by a local LLM |

Response changes since v0.2 are additive. Error handling: missing model/telemetry → `503`, bad input
(wrong window shape, unknown satellite, data gap in the latest window) → `422`.

**Configuration (environment)**

| variable                                     | effect                                                                                                                 |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `NAVIGUARD_PROFILE`                          | `synthetic` (default) or `navic`                                                                                       |
| `NAVIGUARD_API_KEY`                          | if set, every endpoint except `/health` requires an `X-API-Key` header (the dashboard sends it from the same variable) |
| `NAVIGUARD_CORS_ORIGINS`                     | JSON list of allowed browser origins (defaults to local Streamlit/React/Vite ports)                                    |
| `NAVIGUARD_API_URL`                          | where the dashboard finds the API                                                                                      |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT_S`, `LLM_TEMPERATURE` | local LLM (LM Studio) settings; temperature (default 0.7) applies to the report only |

**Anomaly detector.** Step-1 residuals on the validation split are calibrated to a robust median/MAD
baseline; test residuals are scored as robust z and flagged above a threshold. In the `navic` profile the
threshold is the larger of z = 4 and the 99.5th percentile of validation |z| (see finding 4), calibration
is per satellite, and flags are candidates for review, not confirmed faults.

In the `navic` profile, `detection.refit_aware` adds the refit-aware verdict. It scores the same test
windows on the broadcast-drift residual at z = 4 (or the given `z_threshold`) and labels every jump
`spike`, `sustained`, `refit` or `edge` (data gap), with its satellite. Only spikes and sustained
excursions alarm or count toward its severity. The LLM is told to treat this as the primary verdict.

**Local LLM (optional).** `POST /anomaly-report` can ask a local model via [LM Studio](https://lmstudio.ai)
to narrate the already-computed statistics and give a severity second opinion. It never sees raw
telemetry or forecasts anything. It is told which data, satellites and model are in service, and is
given values precomputed for it: errors in metres of range, lead times, and the best baseline. The
report comes back as four plain-text lines (`Status:`, `Accuracy:`, `Anomalies:`, `Action:`). The
severity check runs at temperature 0.1 so its JSON stays parseable. A 7–8B instruct model (e.g.
Qwen2.5-7B or Llama-3.1-8B, 4-bit) writes noticeably better reports than the 3B default. If LM Studio
is not running the endpoint still returns `200` with the numeric analysis and an explanatory
`llm_status`.

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

1. Huang B., Ji Z. (2021) — SL-LSTM for GPS clock bias, _GPS Solutions_
2. He S., Liu J. (2023) — LSTM for BDS-3 clock prediction, _GPS Solutions_
3. Cai C., Liu M. (2024) — LSTM-Attention for BDS, _GPS Solutions_
4. Bhatt A., Mehta I. (2024) — LSTM for Galileo clock bias, _arXiv:2411.07015_

---

## Roadmap

Done: cross-satellite pooled models, the 24 h (96-step) horizon, per-satellite coverage reports,
block-bootstrap confidence intervals for model comparisons, and a refit-aware anomaly detector that
separates routine ground-segment refits from faults (~95% fewer false alarms; in `anomaly-eval` and
`/anomaly-report`), a two-state Kalman-filter clock baseline, Docker (API + dashboard via docker
compose), and a BKG/ESA mirror switch for downloads.

In progress: six-month dataset (Apr–Sep 2026).

Next:

- Hyperparameter tuning of the neural models.
- Probabilistic (quantile or conformal) forecasts.
- Precise clocks for GPS/Galileo/BeiDou (IGS `.CLK`) as an external validation set.
- Allan deviation (frequency-stability) reporting alongside MAE, the standard metric in clock science.
- Versioned dataset releases with a DOI (e.g. Zenodo), so results stay reproducible even if the source
  archive changes.
- A short guide to adding a new forecaster or detector, with config-driven experiments.
- Lower-latency data sources (CDDIS sub-hourly IRNSS files, real-time streams, or a NavIC receiver),
  scheduled data refresh and retraining.

---

## Contributors

Siddhant — modelling, benchmark, API, real-data pipeline · Om — preprocessing · Pratyush Narain —
dashboard design.

_Relevant to SDG 9 (Industry, Innovation and Infrastructure): dependable satellite-navigation timing._
