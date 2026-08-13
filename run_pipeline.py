"""
run_pipeline.py  ·  NaviGuard — Full Pipeline Runner
──────────────────────────────────────────────────────
Runs the complete NaviGuard pipeline in sequence:
  1. generate_data.py   → data/satellite_telemetry.csv
  2. preprocess.py      → data/X_seq.npy  |  y_seq.npy  |  models/scaler.pkl
  3. train_lstm.py      → models/lstm_satellite.h5
  4. predict.py         → outputs/prediction_plot.png
  5. Print instructions to launch dashboard

Usage: python run_pipeline.py
"""

import subprocess, sys, time

STEPS = [
    ("Data Generation",   ["python", "generate_data.py"]),
    ("Preprocessing",     ["python", "preprocess.py"]),
    ("LSTM Training",     ["python", "train_lstm.py"]),
    ("Prediction & Eval", ["python", "predict.py"]),
]

def run_step(name, cmd):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[pipeline] ✗ FAILED at step: {name}")
        sys.exit(result.returncode)
    print(f"\n[pipeline] ✓ {name} completed in {elapsed:.1f}s")

if __name__ == "__main__":
    print("\n NaviGuard — Full Pipeline Run")
    print("  Dept. of AI & DS, BMSCE  |  NavIC/GNSS Clock Prediction\n")
    total_start = time.time()
    for name, cmd in STEPS:
        run_step(name, cmd)
    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ALL STEPS COMPLETE  |  Total time: {total:.1f}s")
    print(f"{'='*60}")
    print("\n  \U0001f680  Launch the Streamlit dashboard:")
    print("     streamlit run dashboard.py\n")
