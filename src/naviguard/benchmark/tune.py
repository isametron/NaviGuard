"""naviguard.benchmark.tune — leak-free hyperparameter search that never sees any test period.

The benchmark's rolling-origin folds expand: a later fold trains and validates on data that is an
earlier fold's test block. So tuning uses only the windows whose targets end before the benchmark's
*first* test block, and runs the same rolling-origin scheme again inside that span (inner folds).
Each candidate trains on an inner fold's training part, stops early on its validation part, and is
scored on its held-out inner block. The score is relative MAE (model MAE over persistence MAE on the
same windows), averaged over inner folds and then satellites, so satellites count equally whatever
their error scale.

Neural models get a random search over the settings below; trial 0 is always the untuned default,
so the chosen setting is never worse than the default on the tuning score. Ridge gets its full
alpha grid.
Results are saved after every trial and `resume` continues an interrupted run.
"""

import itertools
import json
import os
import random
import time

import numpy as np
import pandas as pd

from naviguard.benchmark.models import NEURAL, TUNABLE, make_model
from naviguard.benchmark.windows import Windows, load_series, make_windows, rolling_origin_folds

NEURAL_DEFAULT = {"units": [64, 32], "dropout": 0.2, "lr": 1e-3, "batch": 64, "dense": 16, "lookback": None}
NEURAL_SPACE = {
    "units": [[32, 16], [64, 32], [64, 64], [128, 32], [128, 64]],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "lr": [3e-4, 1e-3, 3e-3],
    "batch": [32, 64, 128],
    "dense": [16, 32],
    "lookback": ["half", None],          # "half" = the most recent half of each window
}
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def candidates(model: str, n_trials: int, seq_len: int, seed: int = 0) -> list[dict]:
    """Deterministic candidate settings for one model (trial 0 = default for neural models)."""
    if model == "ridge":
        return [{"alpha": a} for a in RIDGE_ALPHAS]
    if model not in NEURAL:
        raise ValueError(f"{model} has no settings to tune; tunable models: {TUNABLE}")
    keys = list(NEURAL_SPACE)
    grid = [dict(zip(keys, combo)) for combo in itertools.product(*(NEURAL_SPACE[k] for k in keys))]
    random.Random(seed).shuffle(grid)
    out = [dict(NEURAL_DEFAULT)]
    for g in grid:
        if len(out) >= n_trials:
            break
        if g != NEURAL_DEFAULT:
            out.append(g)
    for c in out:                        # store a concrete record count, valid for this seq_len only
        if c["lookback"] == "half":
            c["lookback"] = seq_len // 2
    return out


def pre_test_windows(w: Windows, horizon: int, n_folds: int) -> Windows:
    """The windows whose targets end before the benchmark's first test block (purged)."""
    first_test = rolling_origin_folds(w, horizon, n_folds)[0].test[0]
    keep = np.flatnonzero(w.i + horizon - 1 < w.i[first_test])
    return Windows(X=w.X[keep], y=w.y[keep], i=w.i[keep], last_bias_ns=w.last_bias_ns[keep],
                   t_target=w.t_target[keep])


def _windows(csv_path: str, seq_len: int, horizon: int, n_folds: int, min_windows: int, verbose: bool):
    data = {}
    for sat, s in sorted(load_series(csv_path).items()):
        try:
            w = make_windows(s, seq_len, horizon)
            if len(w.i) < min_windows:
                continue                 # the benchmark skips it too
            pre = pre_test_windows(w, horizon, n_folds)
            data[sat] = (pre, rolling_origin_folds(pre, horizon, n_folds))
        except ValueError as e:
            if verbose:
                print(f"[tune] I{sat:02d}: not used for tuning ({e})")
    if not data:
        raise ValueError("no satellite has enough pre-test windows for these settings")
    return data


def _score(model: str, params: dict, data: dict, horizon: int, epochs: int, seed: int) -> dict:
    per_sat = {}
    for sat, (w, folds) in data.items():
        mae, rel = [], []
        for f in folds:                  # inner folds, all before the benchmark's first test block
            kw = dict(params, epochs=epochs) if model in NEURAL else dict(params)
            m = make_model(model, horizon, seed=seed, **kw).fit(w.X[f.train], w.y[f.train], w.X[f.val], w.y[f.val])
            err = float(np.abs(w.y[f.test] - m.predict(w.X[f.test])).mean())
            mae.append(err)
            rel.append(err / max(float(np.abs(w.y[f.test]).mean()), 1e-9))  # persistence MAE on the same windows
            if model in NEURAL:
                import tensorflow as tf
                tf.keras.backend.clear_session()                           # keep memory flat over many fits
        per_sat[int(sat)] = {"val_mae": float(np.mean(mae)), "val_rel": float(np.mean(rel))}
    return per_sat


def run_tuning(csv_path: str, out_dir: str, models: tuple[str, ...] = ("ridge", *NEURAL), seq_len: int = 20,
               horizon: int = 6, n_folds: int = 3, n_trials: int = 20, epochs: int = 40, min_windows: int = 200,
               seed: int = 0, resume: bool = False, verbose: bool = True) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    trials_path = os.path.join(out_dir, "trials.csv")
    done = pd.read_csv(trials_path).to_dict("records") if resume and os.path.exists(trials_path) else []
    seen = {(r["model"], int(r["trial"])) for r in done}
    data = _windows(csv_path, seq_len, horizon, n_folds, min_windows, verbose)
    if verbose:
        sizes = {f"I{s:02d}": len(w.i) for s, (w, _) in data.items()}
        print(f"[tune] h{horizon} seq_len {seq_len}: pre-test windows {sizes}; no benchmark test data is used")

    for model in models:
        cands = candidates(model, n_trials, seq_len, seed)
        for k, params in enumerate(cands):
            if (model, k) in seen:
                continue
            t0 = time.time()
            per_sat = _score(model, params, data, horizon, epochs, seed)
            score = float(np.mean([v["val_rel"] for v in per_sat.values()]))
            done.append({"model": model, "trial": k, "score": score, "params": json.dumps(params),
                         "per_satellite": json.dumps(per_sat), "seconds": round(time.time() - t0, 1)})
            pd.DataFrame(done).to_csv(trials_path, index=False)              # survives an interrupted run
            if verbose:
                print(f"[tune] {model:<10} trial {k + 1:>2}/{len(cands)}  relative val MAE {score:.4f}  "
                      f"({time.time() - t0:.0f}s)  {params}")

    trials = pd.DataFrame(done)
    overall, per_sat = select_best(trials, models)
    result = {"horizon": horizon, "seq_len": seq_len, "n_folds": n_folds,
              "selected_on": "inner folds before the first benchmark test block",
              "models": overall, "per_satellite": per_sat}
    with open(os.path.join(out_dir, "best.json"), "w") as f:
        json.dump(result, f, indent=2)
    write_tuning_summary(trials, out_dir, horizon, seq_len)
    return result


def select_best(trials: pd.DataFrame, models) -> tuple[dict, dict]:
    """Best settings overall (averaged over satellites; used for pooled models and satellites that
    were not tuned) and per satellite (each satellite's models are trained separately, so each gets
    the setting that did best on its own pre-test data; an average would let the satellite with the
    largest errors decide for all)."""
    overall, per_sat = {}, {}
    for model in models:
        t = trials[trials["model"] == model]
        overall[model] = json.loads(t.loc[t["score"].idxmin(), "params"])
        rel = pd.DataFrame([{s: v["val_rel"] for s, v in json.loads(p).items()} for p in t["per_satellite"]],
                           index=t.index)
        for sat in rel.columns:
            per_sat.setdefault(str(sat), {})[model] = json.loads(t.loc[rel[sat].idxmin(), "params"])
    return overall, per_sat


def write_tuning_summary(trials: pd.DataFrame, out_dir: str, horizon: int, seq_len: int) -> None:
    lines = [f"# Tuning summary (horizon {horizon}, seq_len {seq_len})\n",
             "Score = MAE relative to persistence on held-out inner folds, all before the benchmark's first "
             "test block, averaged over folds and satellites (lower is better). No benchmark test data was used.\n",
             "| model | trials | default score | best score | change | best settings |", "|---|---|---|---|---|---|"]
    for model, t in trials.groupby("model", sort=False):
        best = t.loc[t["score"].idxmin()]
        default = t[t["trial"] == 0]["score"]
        d = float(default.iloc[0]) if len(default) else float("nan")
        change = f"{(best['score'] / d - 1):+.1%}" if np.isfinite(d) else "—"
        lines.append(f"| {model} | {len(t)} | {d:.4f} | {best['score']:.4f} | {change} | `{best['params']}` |")
    lines += ["", "## Per satellite (used by the benchmark; each satellite's models are trained separately)\n",
              "| model | satellite | default score | best score | change | best settings |", "|---|---|---|---|---|---|"]
    for model, t in trials.groupby("model", sort=False):
        rel = pd.DataFrame([{s: v["val_rel"] for s, v in json.loads(p).items()} for p in t["per_satellite"]],
                           index=t.index)
        for sat in rel.columns:
            b = rel[sat].idxmin()
            d0 = rel.loc[t.index[t["trial"] == 0], sat]
            d = float(d0.iloc[0]) if len(d0) else float("nan")
            change = f"{(rel.loc[b, sat] / d - 1):+.1%}" if np.isfinite(d) else "—"
            lines.append(f"| {model} | I{int(sat):02d} | {d:.4f} | {rel.loc[b, sat]:.4f} | {change} | "
                         f"`{t.loc[b, 'params']}` |")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def load_params(path: str, horizon: int, seq_len: int) -> tuple[dict, dict]:
    """(overall, per-satellite) tuned settings from best.json, checked against the benchmark's
    horizon and window length."""
    with open(path) as f:
        best = json.load(f)
    if (best.get("horizon"), best.get("seq_len")) != (horizon, seq_len):
        raise ValueError(f"{path} was tuned for horizon {best.get('horizon')}, seq_len {best.get('seq_len')}; "
                         f"this run uses horizon {horizon}, seq_len {seq_len}")
    return best["models"], best.get("per_satellite", {})
