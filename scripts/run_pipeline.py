"""
run_pipeline.py · NaviGuard — Full Pipeline Runner
──────────────────────────────────────────────────────
Convenience wrapper around the `naviguard` CLI. Runs the complete pipeline
in sequence, each stage as its own subprocess (keeps TensorFlow/NumPy global
seed state from leaking across stages):
  1. naviguard generate    -> data/satellite_telemetry.csv
  2. naviguard preprocess  -> data/X_seq.npy | y_seq.npy | models/scaler.pkl
  3. naviguard train       -> models/lstm_attention_satellite.keras
  4. naviguard predict     -> outputs/prediction_plot.png

Usage: python scripts/run_pipeline.py
"""

import subprocess
import sys
import time

STEPS = [
    ("Data Generation",   ["naviguard", "generate"]),
    ("Preprocessing",     ["naviguard", "preprocess"]),
    ("Attention-LSTM Training", ["naviguard", "train"]),
    ("Prediction & Eval", ["naviguard", "predict", "--save-plot"]),
]


def run_step(name, cmd):
    print(f"\n{'=' * 60}")
    print(f"  STEP: {name}")
    print(f"{'=' * 60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[pipeline] FAILED at step: {name}")
        sys.exit(result.returncode)
    print(f"\n[pipeline] {name} completed in {elapsed:.1f}s")


if __name__ == "__main__":
    print("\n NaviGuard — Full Pipeline Run")
    print("  Dept. of AI & DS, BMSCE  |  NavIC/GNSS Clock Prediction\n")
    total_start = time.time()
    for name, cmd in STEPS:
        run_step(name, cmd)
    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  ALL STEPS COMPLETE  |  Total time: {total:.1f}s")
    print(f"{'=' * 60}")
    print("\n  Launch the API: uvicorn naviguard.api.main:app --reload\n")
