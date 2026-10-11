"""naviguard.benchmark.run — rolling-origin benchmark of clock-bias forecasters on real series.

For every satellite, windows are built level-free (see windows.py), split into purged
rolling-origin folds, and every model is fitted per fold (neural models over several
seeds). Outputs a tidy results CSV, a Markdown/LaTeX summary, Diebold–Mariano tests of the
attention-LSTM against every other model, moving-block bootstrap 95% confidence intervals
(ci.csv / ci.md), and the per-window test predictions (predictions_Ixx.npz) so the tests
and intervals can be recomputed without retraining.
"""

import os
import time

import numpy as np
import pandas as pd

from naviguard.benchmark.models import ALL_MODELS, NEURAL, make_model
from naviguard.benchmark.stats import block_bootstrap_ci, diebold_mariano
from naviguard.benchmark.windows import load_series, make_windows, rolling_origin_folds

CI_REFERENCE = "attn_lstm"
CI_N_BOOT = 2000


def _error_rows(sat: int, fold: int, model: str, seed: int, err: np.ndarray) -> list[dict]:
    return [dict(satellite=sat, fold=fold, model=model, seed=seed, step=step + 1, n=len(err),
                 mae=float(np.abs(err[:, step]).mean()), rmse=float(np.sqrt((err[:, step] ** 2).mean())))
            for step in range(err.shape[1])]


def _dm_rows(sat: int, y_test: np.ndarray, ref: np.ndarray, name: str, other: np.ndarray, horizon: int) -> list[dict]:
    """Diebold–Mariano of the attention-LSTM (ref) against one model, at the first and last step."""
    rows = []
    for step in (0, horizon - 1):
        stat, p = diebold_mariano(y_test[:, step] - ref[:, step], y_test[:, step] - other[:, step], horizon=step + 1)
        rows.append(dict(satellite=sat, vs=name, step=step + 1, dm_stat=stat, p_value=p))
    return rows


def bootstrap_ci_rows(sat: int, y_test: np.ndarray, ens: dict[str, np.ndarray], horizon: int,
                      reference: str = CI_REFERENCE, n_boot: int = CI_N_BOOT) -> list[dict]:
    """Moving-block bootstrap 95% CIs at the first and last forecast step.

    For every model: the MAE of its (seed-ensemble) forecast. Against `reference`: the mean
    absolute-error difference (model minus reference; positive means the reference is better).
    Overlapping multi-step forecasts are autocorrelated over about `step` windows, so the block
    length is max(12, step).
    """
    rows = []
    for step in sorted({1, horizon}):
        block = max(12, step)
        ref_err = np.abs(y_test[:, step - 1] - ens[reference][:, step - 1]) if reference in ens else None
        for name, pred in ens.items():
            err = np.abs(y_test[:, step - 1] - pred[:, step - 1])
            lo, hi = block_bootstrap_ci(err, block=block, n_boot=n_boot)
            rows.append(dict(satellite=sat, model=name, step=step, quantity="mae", value=float(err.mean()),
                             ci_lo=lo, ci_hi=hi, block=block))
            if ref_err is not None and name != reference:
                d = err - ref_err
                lo, hi = block_bootstrap_ci(d, block=block, n_boot=n_boot)
                rows.append(dict(satellite=sat, model=name, step=step, quantity=f"diff_vs_{reference}",
                                 value=float(d.mean()), ci_lo=lo, ci_hi=hi, block=block))
    return rows


def write_ci_summary(ci: pd.DataFrame, out_dir: str, reference: str = CI_REFERENCE) -> None:
    lines = ["# Moving-block bootstrap 95% confidence intervals (seed-ensemble forecasts)\n",
             f"MAE in ns. Δ = MAE(model) − MAE({reference}); a positive Δ means {reference} is better. "
             "A Δ interval that excludes 0 is marked *.\n",
             "Note: these use the seed-ensemble (mean) forecast, so MAE values can differ slightly from "
             "summary.md, which averages per-seed MAEs.\n"]
    for sat, g in ci.groupby("satellite"):
        for step, gs in g.groupby("step"):
            lines += ["", f"## I{int(sat):02d}, step {int(step)} (block {int(gs['block'].iloc[0])})\n",
                      f"| model | MAE [95% CI] | Δ vs {reference} [95% CI] |", "|---|---|---|"]
            mae = gs[gs["quantity"] == "mae"].set_index("model")
            diff = gs[gs["quantity"] == f"diff_vs_{reference}"].set_index("model")
            for model, r in mae.sort_values("value").iterrows():
                cell = "—"
                if model in diff.index:
                    d = diff.loc[model]
                    star = " *" if (d.ci_lo > 0 or d.ci_hi < 0) else ""
                    cell = f"{d.value:+.2f} [{d.ci_lo:+.2f}, {d.ci_hi:+.2f}]{star}"
                lines.append(f"| {model} | {r.value:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}] | {cell} |")
    with open(os.path.join(out_dir, "ci.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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
    rows, dm_rows, ci_rows = [], [], []
    done: set[int] = set()
    if resume and os.path.exists(os.path.join(out_dir, "results.csv")):
        prev = pd.read_csv(os.path.join(out_dir, "results.csv"))
        rows = prev.to_dict("records")
        done = set(prev["satellite"].unique())
        dm_path = os.path.join(out_dir, "dm_tests.csv")
        dm_rows = pd.read_csv(dm_path).to_dict("records") if os.path.exists(dm_path) else []
        ci_path = os.path.join(out_dir, "ci.csv")
        ci_rows = pd.read_csv(ci_path).to_dict("records") if os.path.exists(ci_path) else []
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
                    rows += _error_rows(sat, k, name, seed, w.y[f.test] - p)
                per_seed.append(np.concatenate(fold_preds))
            preds[name] = per_seed
            if verbose:
                mae1 = np.mean([np.abs(y_test[:, 0] - p[:, 0]).mean() for p in per_seed])
                print(f"[bench]   {name:<16} step-1 MAE {mae1:8.2f} ns   ({time.time() - t_start:.0f}s elapsed)")

        # Diebold–Mariano: attention-LSTM vs each other model (neural = seed-ensemble mean).
        if "attn_lstm" in preds:
            ref = np.mean(preds["attn_lstm"], axis=0)
            for name, ps in preds.items():
                if name != "attn_lstm":
                    dm_rows += _dm_rows(sat, y_test, ref, name, np.mean(ps, axis=0), horizon)

        # Per-window test predictions (seed-ensemble mean) + bootstrap confidence intervals.
        ens = {name: np.mean(ps, axis=0) for name, ps in preds.items()}
        np.savez_compressed(os.path.join(out_dir, f"predictions_I{sat:02d}.npz"), y=y_test.astype(np.float32),
                            **{name: p.astype(np.float32) for name, p in ens.items()})
        ci_rows += bootstrap_ci_rows(sat, y_test, ens, horizon)

        # Save after every satellite so an interrupted run keeps what it finished.
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "results.csv"), index=False)
        if dm_rows:
            pd.DataFrame(dm_rows).to_csv(os.path.join(out_dir, "dm_tests.csv"), index=False)
        pd.DataFrame(ci_rows).to_csv(os.path.join(out_dir, "ci.csv"), index=False)

    if not rows:
        raise ValueError("no satellite had enough windows; fetch more days (naviguard fetch --days N)")
    res = pd.DataFrame(rows)
    write_summary(res, out_dir, horizon)
    if ci_rows:
        write_ci_summary(pd.DataFrame(ci_rows), out_dir)
    return res


def extend_benchmark(
    csv_path: str,
    out_dir: str,
    models: tuple[str, ...],
    seq_len: int = 20,
    horizon: int = 6,
    n_folds: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """Add classical models to a finished benchmark in out_dir without retraining the others.

    Each satellite's windows and folds are rebuilt (deterministic) and checked against the saved
    test targets, so the data and --seq-len/--horizon/--folds must match the original run. Only
    `models` are fitted; their rows, Diebold–Mariano tests, predictions and bootstrap intervals
    are merged into the existing outputs (re-running replaces earlier rows of the same models).
    """
    if any(m in NEURAL for m in models):
        raise ValueError(f"extend adds classical models only; neural models need a full run: {NEURAL}")
    path = os.path.join(out_dir, "results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no benchmark results in {out_dir}; run `naviguard benchmark` first")
    res = pd.read_csv(path)
    res = res[~res["model"].isin(models)]
    dm_path, ci_path = os.path.join(out_dir, "dm_tests.csv"), os.path.join(out_dir, "ci.csv")
    dm = pd.read_csv(dm_path) if os.path.exists(dm_path) else pd.DataFrame(columns=["vs"])
    ci = pd.read_csv(ci_path) if os.path.exists(ci_path) else pd.DataFrame(columns=["model"])
    dm, ci = dm[~dm["vs"].isin(models)], ci[~ci["model"].isin(models)]
    series = load_series(csv_path)
    rows, dm_rows, ci_rows = [], [], []

    for sat in sorted(res["satellite"].unique()):
        npz_path = os.path.join(out_dir, f"predictions_I{sat:02d}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"{npz_path} missing: the run predates saved predictions; rerun it in full")
        with np.load(npz_path) as z:
            saved = {k: z[k] for k in z.files}
        w = make_windows(series[sat], seq_len, horizon)
        folds = rolling_origin_folds(w, horizon, n_folds)
        y_test = w.y[np.concatenate([f.test for f in folds])]
        if y_test.shape != saved["y"].shape or not np.allclose(y_test, saved["y"], rtol=1e-4, atol=1e-2):
            raise ValueError(f"I{sat:02d}: rebuilt windows do not match {npz_path}; use the original data and "
                             "--seq-len/--horizon/--folds")
        for name in models:
            fold_preds = []
            for k, f in enumerate(folds):
                p = make_model(name, horizon).fit(w.X[f.train], w.y[f.train], w.X[f.val], w.y[f.val]).predict(
                    w.X[f.test])
                fold_preds.append(p)
                rows += _error_rows(sat, k, name, 0, w.y[f.test] - p)
            saved[name] = np.concatenate(fold_preds).astype(np.float32)
            if "attn_lstm" in saved:
                dm_rows += _dm_rows(sat, y_test, saved["attn_lstm"], name, saved[name], horizon)
            if verbose:
                print(f"[extend] I{sat:02d} {name:<14} step-1 MAE {np.abs(y_test[:, 0] - saved[name][:, 0]).mean():8.2f} ns")
        np.savez_compressed(npz_path, **saved)
        ens = {name: saved[name] for name in (*models, CI_REFERENCE) if name in saved}
        ci_rows += [r for r in bootstrap_ci_rows(sat, y_test, ens, horizon) if r["model"] in models]

    res = pd.concat([res, pd.DataFrame(rows)], ignore_index=True)
    res.to_csv(path, index=False)
    if dm_rows:
        pd.concat([dm, pd.DataFrame(dm_rows)], ignore_index=True).to_csv(dm_path, index=False)
    ci = pd.concat([ci, pd.DataFrame(ci_rows)], ignore_index=True)
    ci.to_csv(ci_path, index=False)
    write_summary(res, out_dir, horizon)
    write_ci_summary(ci, out_dir)
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
                    rows += _error_rows(sat, k, f"{name}_pooled", seed, w.y[f.test] - m.predict(w.X[f.test]))
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
