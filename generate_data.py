"""
generate_data.py  ·  NaviGuard — Telemetry Simulation
───────────────────────────────────────────────────────
Generates satellite_telemetry.csv using physics-motivated functions:
  clock_bias      → 24-hour sinusoidal + Gaussian noise  (NavIC Rubidium spec)
  clock_drift     → 24-hour cosine derivative + noise
  ephemeris_error → 12-hour sinusoidal + noise  (NavIC L5-band orbit accuracy)
Run  : python generate_data.py
Out  : data/satellite_telemetry.csv
"""

import os, random, math, csv
import numpy as np

SEED       = 42
N_SAMPLES  = 200
INTERVAL_S = 900   # 15 minutes

random.seed(SEED)
np.random.seed(SEED)

os.makedirs("data", exist_ok=True)
rows = []
for i in range(N_SAMPLES):
    t  = i * INTERVAL_S
    cb = 1e-6  * math.sin(2 * math.pi * t / 86400) + np.random.normal(0, 2e-8)
    cd = 1e-10 * math.cos(2 * math.pi * t / 86400) + np.random.normal(0, 5e-12)
    ee = 0.30  * math.sin(2 * math.pi * t / 43200) + np.random.normal(0, 0.05)
    rows.append([i, t, round(cb, 12), round(cd, 14), round(ee, 6)])

out = "data/satellite_telemetry.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample_id","timestamp_s","clock_bias_s",
                "clock_drift_s_per_s","ephemeris_error_m"])
    w.writerows(rows)

print(f"[generate_data] {N_SAMPLES} rows written → {out}")
print(f"[generate_data] Interval: {INTERVAL_S}s  |  Total span: "
      f"{N_SAMPLES * INTERVAL_S / 3600:.1f} hours")
print("[generate_data] ✓ Complete")
