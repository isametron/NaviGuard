# 🛰️ NaviGuard: AI-Driven Satellite Clock Intelligence
**Dept. of AI & DS, BMS College of Engineering | 2025–26**

> An end-to-end LSTM-based prediction pipeline for NavIC/GNSS satellite clock bias
> and ephemeris error forecasting — enabling proactive correction over reactive post-hoc adjustment.

---

## 📁 Project Structure

```
satellite_clock_project/
│
├── data/
│   └── satellite_telemetry.csv     # NavIC/GNSS telemetry (200 samples, 15-min intervals)
│
├── models/                          # Auto-generated after training
│   ├── scaler.pkl                  # Fitted MinMaxScaler
│   └── lstm_satellite.h5           # Best-checkpoint LSTM model
│
├── outputs/                         # Auto-generated after prediction
│   ├── prediction_plot.png          # Actual vs Predicted + Residuals chart
│   └── architecture_diagram.png     # System architecture visualization
│
├── generate_data.py                 # (Re)generate synthetic telemetry CSV
├── preprocess.py                    # Scale + sliding-window sequence construction
├── train_lstm.py                    # LSTM model training with EarlyStopping
├── predict.py                       # Inference, MAE evaluation, plot generation
├── dashboard.py                     # Streamlit interactive dashboard
├── run_pipeline.py                  # 🚀 One-click full pipeline runner
├── architecture_diagram.py          # Generate architecture PNG
├── NaviGuard_Architecture.drawio    # Draw.io system architecture diagram
└── requirements.txt
```

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline in one command
python run_pipeline.py

# 3. Launch the dashboard
streamlit run dashboard.py
```

Or run each step manually:
```bash
python generate_data.py    # Step 0: (re)generate CSV
python preprocess.py       # Step 1: normalize + build sequences
python train_lstm.py       # Step 2: train LSTM
python predict.py          # Step 3: evaluate + save plot
streamlit run dashboard.py # Step 4: launch dashboard
```

---

## 🧠 Model Architecture

| Layer  | Type    | Units | Parameters                          |
|--------|---------|-------|--------------------------------------|
| 1      | LSTM    | 64    | return_sequences=True, input=(20,3) |
| 2      | Dropout | —     | rate=0.2                             |
| 3      | LSTM    | 32    | return_sequences=False               |
| 4      | Dropout | —     | rate=0.2                             |
| 5      | Dense   | 16    | activation='relu'                    |
| 6      | Dense   | 1     | Linear output                        |

- **Optimizer:** Adam (lr=0.001, β₁=0.9, β₂=0.999)
- **Loss:** Mean Squared Error (MSE)
- **Callbacks:** EarlyStopping (patience=10) + ModelCheckpoint + ReduceLROnPlateau

---

## 📊 Key Specs

| Parameter        | Value                              |
|------------------|------------------------------------|
| Input features   | clock_bias_s, clock_drift_s_per_s, ephemeris_error_m |
| Sequence length  | 20 steps (5 hours of history)      |
| Train/Val split  | 80/20 (chronological, no shuffle)  |
| Sampling rate    | 15 minutes (900 seconds)           |
| MAE target       | ≤ 50 nanoseconds on test split     |
| Output unit      | Seconds → converted to μs and ns  |
| Dashboard URL    | localhost:8501                     |

---

## 🔗 Pipeline Data Flow

```
NavIC/GNSS Source
       ↓
satellite_telemetry.csv
       ↓  [preprocess.py]
X_seq.npy + y_seq.npy + scaler.pkl
       ↓  [train_lstm.py]
lstm_satellite.h5
       ↓  [predict.py]
prediction_plot.png + MAE report
       ↓  [dashboard.py]
Streamlit UI @ localhost:8501
```

---

## 📚 References

Key papers this work builds upon:
1. Huang B., Ji Z. (2021) — SL-LSTM for GPS clock bias, *GPS Solutions*
2. He S., Liu J. (2023) — LSTM for BDS-3 clock prediction, *GPS Solutions*
3. Cai C., Liu M. (2024) — LSTM-Attention for BDS, *GPS Solutions*
4. Bhatt A., Mehta I. (2024) — LSTM for Galileo clock bias, *arXiv:2411.07015*

---

*NaviGuard directly addresses SDG 9 (Industry & Innovation), SDG 11 (Sustainable Cities),
and SDG 13 (Climate Action) through AI-augmented satellite navigation reliability.*
