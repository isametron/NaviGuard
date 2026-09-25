"""naviguard.benchmark.run — rolling-origin benchmark of clock-bias forecasters on real series.

For every satellite, windows are built level-free (see windows.py), split into purged
rolling-origin folds, and every model is fitted per fold (neural models over several
seeds). Outputs a tidy results CSV, a Markdown/LaTeX summary, and Diebold–Mariano tests
of the attention-LSTM against every other model.
"""

import os
import time

import numpy as np
import pandas as pd

from naviguard.benchmark.models import ALL_MODELS, NEURAL, make_model
from naviguard.benchmark.stats import diebold_mariano
from naviguard.benchmark.windows import load_series, make_windows, rolling_origin_folds


def run_benchmark(
    csv_path: str,
    out_dir: str,
    satellites: list[int] | None = None,
    models: tuple[str, ...] = ALL_MODELS,
    seq_len: int = 20,
    horizon: int = 6,
    n_folds: int = 3,
    seeds: int = 3,
    min_windows: int = 200,
    epochs: int = 60,
    verbose: bool = True,
    resume: bool = False,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    series = load_series(csv_path)
    rows, dm_rows = [], []
    done: set[int] = set()
    if resume and os.path.exists(os.path.join(out_dir, "results.csv")):
        prev = pd.read_csv(os.path.join(out_dir, "results.csv"))
        rows = prev.to_dict("records")
        done = set(prev["satellite"].unique())
        dm_path = os.path.join(out_dir, "dm_tests.csv")
        dm_rows = pd.read_csv(dm_path).to_dict("records") if os.path.exists(dm_path) else []
        if verbose:
            print(f"[bench] resuming; already done: {sorted(done)}")
    t_start = time.time()

    for sat, s in sorted(series.items()):
        if (satellites and sat not in satellites) or sat in done:
            continue
        try:
            w = make_windows(s, seq_len, horizon)
            if len(w.i) < min_windows:
                raise ValueError(f"only {len(w.i)} windows (< {min_windows})")
            folds = rolling_origin_folds(w, horizon, n_folds)
        except ValueError as e:
            if verbose:
                print(f"[bench] I{sat:02d}: skipped — {e}")
            continue
        test_idx = np.concatenate([f.test for f in folds])
        y_test = w.y[test_idx]
        preds: dict[str, list[np.ndarray]] = {}
        if verbose:
            print(f"[bench] I{sat:02d}: {len(w.i)} windows, {n_folds} folds, {len(test_idx)} test windows")

        for name in models:
            n_seed = seeds if name in NEURAL else 1
            per_seed = []
            for seed in range(n_seed):
                fold_preds = []
                for k, f in enumerate(folds):
                    kw = {"epochs": epochs} if name in NEURAL else {}
                    m = make_model(name, horizon, seed=seed, **kw).fit(
                        w.X[f.train], w.y[f.train], w.X[f.val], w.y[f.val])
                    p = m.predict(w.X[f.test])
                    fold_preds.append(p)
                    err = w.y[f.test] - p
                    for step in range(horizon):
                        rows.append(dict(satellite=sat, fold=k, model=name, seed=seed, step=step + 1,
                                         n=len(f.test), mae=float(np.abs(err[:, step]).mean()),
                                         rmse=float(np.sqrt((err[:, step] ** 2).mean()))))
                per_seed.append(np.concatenate(fold_preds))
            preds[name] = per_seed
            if verbose:
                mae1 = np.mean([np.abs(y_test[:, 0] - p[:, 0]).mean() for p in per_seed])
                print(f"[bench]   {name:<16} step-1 MAE {mae1:8.2f} ns   ({time.time() - t_start:.0f}s elapsed)")

        # Diebold–Mariano: attention-LSTM vs each other model (neural = seed-ensemble mean).
        if "attn_lstm" in preds:
            ref = np.mean(preds["attn_lstm"], axis=0)
            for name, ps in preds.items():
                if name == "attn_lstm":
                    continue
                other = np.mean(ps, axis=0)
                for step in (0, horizon - 1):
                    stat, p = diebold_mariano(y_test[:, step] - ref[:, step], y_test[:, step] - other[:, step],
                                              horizon=step + 1)
                    dm_rows.append(dict(satellite=sat, vs=name, step=step + 1, dm_stat=stat, p_value=p))

        # Save after every satellite so an interrupted run keeps what it finished.
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "results.csv"), index=False)
        if dm_rows:
            pd.DataFrame(dm_rows).to_csv(os.path.join(out_dir, "dm_tests.csv"), index=False)

    if not rows:
        raise ValueError("no satellite had enough windows; fetch more days (naviguard fetch --days N)")
    res = pd.DataFrame(rows)
    write_summary(res, out_dir, horizon)
    return res


def pooled(res: pd.DataFrame) -> pd.DataFrame:
    """Pool folds (weighted by test size), then average over seeds; keep std over seeds."""
    def agg(g):
        return pd.Series({
            "mae": np.average(g["mae"], weights=g["n"]),
            "rmse": np.sqrt(np.average(g["rmse"] ** 2, weights=g["n"])),
        })
    per_seed = res.groupby(["satellite", "model", "seed", "step"]).apply(agg, include_groups=False).reset_index()
    out = per_seed.groupby(["satellite", "model", "step"]).agg(
        mae=("mae", "mean"), mae_std=("mae", "std"), rmse=("rmse", "mean")).reset_index()
    out["mae_std"] = out["mae_std"].fillna(0.0)
    return out


def write_summary(res: pd.DataFrame, out_dir: str, horizon: int) -> None:
    p = pooled(res)
    steps = sorted({1, max(1, horizon // 2), horizon})
    lines = ["# Benchmark summary — MAE (ns), mean over satellites; ± = seed std (neural models)\n"]
    for sat_label, sub in [("all satellites", p)] + [(f"I{s:02d}", g) for s, g in p.groupby("satellite")]:
        tab = sub[sub["step"].isin(steps)].groupby(["model", "step"]).agg(
            mae=("mae", "mean"), sd=("mae_std", "mean")).reset_index()
        wide = tab.pivot(index="model", columns="step", values="mae")
        sd = tab.pivot(index="model", columns="step", values="sd")
        lines.append(f"\n## {sat_label}\n")
        lines.append("| model | " + " | ".join(f"step {s}" for s in steps) + " |")
        lines.append("|---|" + "---|" * len(steps))
        best = {s: wide[s].min() for s in steps}
        for model in wide.index:
            cells = []
            for s in steps:
                v, e = wide.loc[model, s], sd.loc[model, s]
                cell = f"{v:.2f}" + (f" ± {e:.2f}" if e > 0 else "")
                cells.append(f"**{cell}**" if np.isclose(v, best[s]) else cell)
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # LaTeX table (all satellites) for the paper.
    tab = p.groupby(["model", "step"])["mae"].mean().unstack("step")[steps]
    with open(os.path.join(out_dir, "summary.tex"), "w", encoding="utf-8") as f:
        f.write(tab.round(2).to_latex(caption="Mean MAE (ns) by forecast step, averaged over satellites.",
                                      label="tab:bench", column_format="l" + "r" * len(steps)))


def run_pooled_benchmark(
    csv_path: str,
    out_dir: str,
    models: tuple[str, ...] = ("ridge", "arima_p10", "lstm", "gru", "attn_lstm"),
    seq_len: int = 20,
    horizon: int = 6,
    n_folds: int = 3,
    seeds: int = 3,
    min_windows: int = 200,
    epochs: int = 60,
    val_frac: float = 0.15,
    verbose: bool = True,
    resume: bool = False,
) -> pd.DataFrame:
    """Cross-satellite pooling: one model per model-type trained on the windows of *all*
    satellites, evaluated on each satellite's own test folds (identical to the per-satellite
    benchmark, so results are directly comparable).

    Windows are level-free, so satellites can share a model. To stay leak-free, only windows
    whose targets end before the test block's first target *time* are used, from every
    satellite. Rows are appended to out_dir/results.csv as `<model>_pooled`.
    """
    series = load_series(csv_path)
    wins, folds = {}, {}
    for sat, s in sorted(series.items()):
        try:
            w = make_windows(s, seq_len, horizon)
            if len(w.i) < min_windows:
                raise ValueError(f"only {len(w.i)} windows (< {min_windows})")
            folds[sat] = rolling_origin_folds(w, horizon, n_folds)
            wins[sat] = w
        except ValueError as e:
            if verbose:
                print(f"[pooled] I{sat:02d}: not scored — {e}")
    if not wins:
        raise ValueError("no satellite had enough windows")
    step_s = float(np.median([series[s].step_s for s in wins]))
    horizon_s = (horizon - 1) * step_s
    rows, done = [], set()
    prev_path = os.path.join(out_dir, "results_pooled.csv")
    if resume and os.path.exists(prev_path):
        prev = pd.read_csv(prev_path)
        rows, done = prev.to_dict("records"), set(prev["satellite"].unique())
        if verbose:
            print(f"[pooled] resuming; already done: {sorted(done)}")
    t0 = time.time()

    for sat, w in wins.items():
        if sat in done:
            continue
        for k, f in enumerate(folds[sat]):
            t_cut = w.t_target[f.test[0]]
            X = np.concatenate([w2.X[w2.t_target + horizon_s < t_cut] for w2 in wins.values()])
            y = np.concatenate([w2.y[w2.t_target + horizon_s < t_cut] for w2 in wins.values()])
            t = np.concatenate([w2.t_target[w2.t_target + horizon_s < t_cut] for w2 in wins.values()])
            order = np.argsort(t, kind="stable")
            X, y, t = X[order], y[order], t[order]
            n_val = max(int(len(X) * val_frac), 1)
            val = np.arange(len(X) - n_val, len(X))
            train = np.flatnonzero(t + horizon_s < t[val[0]])
            for name in models:
                for seed in range(seeds if name in NEURAL else 1):
                    kw = {"epochs": epochs} if name in NEURAL else {}
                    m = make_model(name, horizon, seed=seed, **kw).fit(X[train], y[train], X[val], y[val])
                    err = w.y[f.test] - m.predict(w.X[f.test])
                    for step in range(horizon):
                        rows.append(dict(satellite=sat, fold=k, model=f"{name}_pooled", seed=seed, step=step + 1,
                                         n=len(f.test), mae=float(np.abs(err[:, step]).mean()),
                                         rmse=float(np.sqrt((err[:, step] ** 2).mean()))))
            if verbose:
                print(f"[pooled] I{sat:02d} fold {k}: trained on {len(train)} pooled windows "
                      f"({time.time() - t0:.0f}s elapsed)")
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "results_pooled.csv"), index=False)

    pooled_rows = pd.DataFrame(rows)
    path = os.path.join(out_dir, "results.csv")
    base = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if len(base):
        base = base[~base["model"].str.endswith("_pooled")]
    combined = pd.concat([base, pooled_rows], ignore_index=True)
    combined.to_csv(path, index=False)
    write_summary(combined, out_dir, horizon)
    return combined
