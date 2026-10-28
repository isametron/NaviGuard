"""naviguard.benchmark.intervals — split-conformal forecast ranges from saved benchmark predictions.

For every model, fold and forecast step, the half-width of the range is the conformal quantile of
the absolute errors on that fold's validation windows; the range is then applied to the same
fold's test windows. Nothing is refitted: the inputs are the predictions_Ixx.npz files written by
`naviguard benchmark` (test predictions plus val_<model>, val_y, val_fold and fold).

Scored on test by coverage (share of true values inside the range; nominal 1 - alpha), mean width,
and the interval score of Gneiting & Raftery (width plus 2/alpha times any miss distance; lower is
better). Neural models also use their validation windows for early stopping, so their ranges can be
slightly narrow; classical models never see the validation windows when fitting.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

DEFAULT_ALPHA = 0.1


def conformal_halfwidth(abs_err: np.ndarray, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """Per-column split-conformal quantile of absolute calibration errors (n, steps) -> (steps,).

    Uses the finite-sample rank ceil((n + 1)(1 - alpha)); with too few calibration windows for
    that rank the range is unbounded (inf), as conformal prediction requires.
    """
    abs_err = np.sort(np.asarray(abs_err, dtype=np.float64), axis=0)
    n = abs_err.shape[0]
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if n == 0 or k > n:
        return np.full(abs_err.shape[1], np.inf)
    return abs_err[k - 1]


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    return (hi - lo) + (2 / alpha) * np.clip(lo - y, 0, None) + (2 / alpha) * np.clip(y - hi, 0, None)


def satellite_intervals(z: dict, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Score conformal ranges for every model in one satellite's saved predictions."""
    y, fold, val_y, val_fold = z["y"], z["fold"], z["val_y"], z["val_fold"]
    models = [k[4:] for k in z if k.startswith("val_") and k not in ("val_y", "val_fold")]
    rows = []
    for name in models:
        pred, val_pred = z[name], z[f"val_{name}"]
        half = np.empty_like(y, dtype=np.float64)
        for k in np.unique(fold):
            half[fold == k] = conformal_halfwidth(np.abs(val_y[val_fold == k] - val_pred[val_fold == k]), alpha)
        lo, hi = pred - half, pred + half
        inside = (y >= lo) & (y <= hi)
        score = interval_score(y, lo, hi, alpha)
        for step in range(y.shape[1]):
            rows.append(dict(model=name, step=step + 1, n=len(y), coverage=float(inside[:, step].mean()),
                             width=float(2 * half[:, step].mean()), interval_score=float(score[:, step].mean())))
    return pd.DataFrame(rows)


def build_interval_report(bench_dir: str, out_dir: str | None = None, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Score ranges for every satellite in a benchmark folder; writes intervals.csv and intervals.md."""
    if not 0 < alpha < 1:
        raise ValueError("--alpha must be between 0 and 1 (0.1 gives 90% ranges)")
    out_dir = out_dir or bench_dir
    frames = []
    for path in sorted(glob.glob(os.path.join(bench_dir, "predictions_I*.npz"))):
        with np.load(path) as npz:
            z = {k: npz[k] for k in npz.files}
        if "val_y" not in z:
            raise ValueError(f"{path} has no validation predictions; rerun the benchmark (it saves them since "
                             "26b8f54)")
        df = satellite_intervals(z, alpha)
        df.insert(0, "satellite", int(re.search(r"I(\d+)\.npz$", path).group(1)))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no predictions_I*.npz in {bench_dir}; run `naviguard benchmark` first")
    res = pd.concat(frames, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    res.to_csv(os.path.join(out_dir, "intervals.csv"), index=False)
    write_interval_summary(res, out_dir, alpha)
    return res


def write_interval_summary(res: pd.DataFrame, out_dir: str, alpha: float = DEFAULT_ALPHA) -> None:
    nominal = round(100 * (1 - alpha))
    lines = [f"# {nominal}% split-conformal forecast ranges\n",
             f"Coverage should be close to {nominal}%. Width and interval score in ns (lower is better). "
             "Calibrated per fold on validation windows, scored on test.\n"]
    for sat, g in res.groupby("satellite"):
        horizon = int(g["step"].max())
        for step in sorted({1, horizon}):
            gs = g[g["step"] == step].sort_values("interval_score")
            lines += ["", f"## I{int(sat):02d}, step {step}\n", "| model | coverage | width | interval score |",
                      "|---|---|---|---|"]
            lines += [f"| {r.model} | {100 * r.coverage:.1f}% | {r.width:.2f} | {r.interval_score:.2f} |"
                      for r in gs.itertuples()]
    with open(os.path.join(out_dir, "intervals.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
