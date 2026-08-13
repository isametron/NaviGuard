import os
import shutil

base = r"C:\Users\siddd\Downloads\satellite_clock_project_v2\satellite_clock_project_v2"
paths = [
    os.path.join(base, "models", "lstm_satellite.keras"),
    os.path.join(base, "models", "scaler.pkl"),
    os.path.join(base, "outputs"),
    os.path.join(base, "data", "X_seq.npy"),
    os.path.join(base, "data", "y_seq.npy"),
]

for p in paths:
    if os.path.isfile(p):
        try:
            os.remove(p)
            print(f"Removed file: {p}")
        except Exception as e:
            print(f"Failed to remove file: {p} -> {e}")
    elif os.path.isdir(p):
        try:
            shutil.rmtree(p)
            print(f"Removed dir: {p}")
        except Exception as e:
            print(f"Failed to remove dir: {p} -> {e}")
    else:
        print(f"Not found: {p}")
