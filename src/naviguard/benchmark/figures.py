"""naviguard.benchmark.figures — paper figures and LaTeX tables from benchmark outputs.

Everything is regenerated from the CSVs written by `naviguard benchmark` and
`naviguard anomaly-eval`, so figures never drift from the numbers in the tables.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from naviguard.benchmark.anomaly_eval import DETECTORS, KINDS
from naviguard.benchmark.run import pooled
from naviguard.benchmark.windows import load_series, make_windows

_COLORS = {
    "persistence": "#9e9e9e", "broadcast_drift": "#2e7d32", "linear_extrap": "#8d6e63", "arima_p10": "#ef6c00",
    "ridge": "#1565c0", "lstm": "#c62828", "gru": "#ad1457", "attn_lstm": "#6a1b9a",
}


def _color(model: str) -> str:
    return _COLORS.get(model.replace("_pooled", ""), "#444444")


def _save(fig, out_dir: str, name: str) -> list[str]:
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def plot_mae_vs_step(results_csv: str, out_dir: str, name: str, title: str) -> list[str]:
    """MAE vs forecast step, one panel per satellite (log y); shaded band = seed std."""
    p = pooled(pd.read_csv(results_csv))
    sats = sorted(p["satellite"].unique())
    fig, axes = plt.subplots(1, len(sats), figsize=(4.6 * len(sats), 3.8), squeeze=False, sharey=False)
    for ax, sat in zip(axes[0], sats):
        for model, g in p[p["satellite"] == sat].groupby("model"):
            g = g.sort_values("step")
            ls = "--" if model.endswith("_pooled") else "-"
            ax.plot(g["step"], g["mae"], ls, color=_color(model), label=model, lw=1.6)
            if (g["mae_std"] > 0).any():
                ax.fill_between(g["step"], g["mae"] - g["mae_std"], g["mae"] + g["mae_std"],
                                color=_color(model), alpha=0.12, lw=0)
        ax.set_yscale("log")
        ax.set_xlabel("forecast step")
        ax.set_title(f"I{sat:02d}")
        ax.grid(alpha=0.25, which="both")
    axes[0][0].set_ylabel("MAE (ns)")
    axes[0][-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(title, fontsize=11)
    return _save(fig, out_dir, name)


def plot_residual_tails(csv_path: str, out_dir: str, seq_len: int = 20) -> list[str]:
    """Survival function of the |z| of one-step physics residuals vs. a Gaussian: the heavy-tail evidence."""
    series = load_series(csv_path)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ks = np.logspace(np.log10(0.5), np.log10(60), 60)
    for sat, s in sorted(series.items()):
        try:
            w = make_windows(s, seq_len, 1)
        except ValueError:
            continue
        e = w.y[:, 0] - w.X[:, -1, 1]
        c = np.median(e)
        sig = 1.4826 * np.median(np.abs(e - c))
        z = np.abs((e - c) / sig)
        ax.loglog(ks, [(z > k).mean() for k in ks], label=f"I{sat:02d} (n={len(z)})", lw=1.6)
    ax.loglog(ks, [2 * (1 - stats.norm.cdf(k)) for k in ks], "k--", label="Gaussian", lw=1.2)
    ax.set_ylim(1e-5, 1.5)
    ax.set_xlabel("|z| (robust, MAD-scaled)")
    ax.set_ylabel("P(|z| > k)")
    ax.set_title("Broadcast-clock one-step residuals are heavy-tailed", fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "fig_residual_tails")


def _trial_table(events: pd.DataFrame) -> pd.DataFrame:
    trial = (events.groupby(["satellite", "detector", "kind", "magnitude_sigma", "trial"])
             .agg(tp=("detected", "sum"), n=("detected", "size"), fp=("fp_in_trial", "first")).reset_index())
    tot = trial.groupby(["kind", "magnitude_sigma", "detector"])[["tp", "n", "fp"]].sum().reset_index()
    tot["recall"] = tot["tp"] / tot["n"]
    tot["precision"] = tot["tp"] / (tot["tp"] + tot["fp"]).clip(lower=1)
    tot["f1"] = 2 * tot["precision"] * tot["recall"] / (tot["precision"] + tot["recall"]).clip(lower=1e-9)
    return tot


def plot_anomaly_detection(events_csv: str, out_dir: str) -> list[str]:
    """Recall (top) and F1 (bottom) vs injected-fault magnitude, one column per fault kind."""
    tot = _trial_table(pd.read_csv(events_csv))
    fig, axes = plt.subplots(2, len(KINDS), figsize=(4.0 * len(KINDS), 5.6), sharex=True, sharey="row")
    for j, kind in enumerate(KINDS):
        for d in DETECTORS:
            g = tot[(tot["kind"] == kind) & (tot["detector"] == d)].sort_values("magnitude_sigma")
            for i, metric in enumerate(("recall", "f1")):
                axes[i][j].plot(g["magnitude_sigma"], g[metric], "o-", label=d, lw=1.5, ms=4)
        axes[0][j].set_title(kind)
        axes[1][j].set_xlabel("fault size (× nominal residual σ)")
        for i in range(2):
            axes[i][j].set_ylim(0, 1.03)
            axes[i][j].grid(alpha=0.25)
    axes[0][0].set_ylabel("event recall")
    axes[1][0].set_ylabel("F1")
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Anomaly detectors on real NavIC series with injected faults", fontsize=11)
    return _save(fig, out_dir, "fig_anomaly_detection")


def write_anomaly_tex(events_csv: str, out_dir: str) -> str:
    tot = _trial_table(pd.read_csv(events_csv))
    tab = tot[tot["magnitude_sigma"].isin([5, 10])].pivot_table(
        index=["kind", "magnitude_sigma"], columns="detector", values="f1")[list(DETECTORS)]
    path = os.path.join(out_dir, "tab_anomaly_f1.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(tab.round(2).to_latex(caption="Event-level F1 of anomaly detectors on injected faults "
                                              "(5$\\sigma$ and 10$\\sigma$).", label="tab:anomaly-f1"))
    return path


def build_report(bench_dirs: dict[str, str], anomaly_dir: str | None, telemetry_csv: str | None,
                 out_dir: str) -> list[str]:
    """bench_dirs maps a label (e.g. 'horizon 6') to a benchmark output directory."""
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for label, d in bench_dirs.items():
        csv = os.path.join(d, "results.csv")
        if os.path.exists(csv):
            base = os.path.basename(os.path.normpath(d))
            made += plot_mae_vs_step(csv, out_dir, f"fig_mae_{base}", f"MAE vs forecast step ({label})")
    if telemetry_csv and os.path.exists(telemetry_csv):
        made += plot_residual_tails(telemetry_csv, out_dir)
    ev = os.path.join(anomaly_dir, "injection_events.csv") if anomaly_dir else None
    if ev and os.path.exists(ev):
        made += plot_anomaly_detection(ev, out_dir)
        made.append(write_anomaly_tex(ev, out_dir))
    return made
