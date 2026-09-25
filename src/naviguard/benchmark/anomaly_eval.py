"""naviguard.benchmark.anomaly_eval — evaluating clock-anomaly detectors on real NavIC series.

(A) Semi-synthetic: labelled faults of known size are injected into the *held-out* part of a
    real series, and detectors are scored on event recall, detection delay and false alarms.
    Fault magnitudes are expressed in multiples of the satellite's own nominal residual sigma.
        spike  - one-sample outlier that reverts            (label span 2)
        step   - persistent bias jump                       (label span 1, at onset)
        ramp   - 10-sample drift excursion that leaves an offset (label span 10)
(B) Real events: with no injection, flagged samples are checked against independent broadcast
    metadata (URA-index changes, health flags) versus the chance rate.

Detectors (all calibrated on the nominal first 60% of each series, evaluated on the last 30%):
    physics_z   robust-z of the one-step error of broadcast-drift extrapolation (no learning)
    physics_adaptive  same residual, scored against a causal rolling median/MAD (last 200 samples)
    ridge_z     robust-z of a ridge one-step forecaster's error (learned)
    cusum       two-sided CUSUM on the standardised physics residual
    iforest     Isolation Forest on the recent relative-window shape
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from naviguard.benchmark.models import RidgeDirect
from naviguard.benchmark.windows import SatSeries, load_series, make_windows

DETECTORS = ("physics_z", "physics_adaptive", "ridge_z", "cusum", "iforest")
KINDS = {"spike": 2, "step": 1, "ramp": 10}          # label span (samples)
MAGNITUDES = (3, 5, 10, 20)                          # in nominal-residual sigmas
Z_THRESHOLD = 4.0
CUSUM_K, CUSUM_H = 1.0, 10.0
_MAD = 1.4826


def _robust_scale(x: np.ndarray) -> tuple[float, float]:
    c = float(np.median(x))
    return c, max(_MAD * float(np.median(np.abs(x - c))), 1e-9)


def _cusum_alarms(z: np.ndarray, k: float = CUSUM_K, h: float = CUSUM_H) -> np.ndarray:
    s, out = 0.0, np.zeros(len(z), dtype=bool)
    for i, v in enumerate(np.abs(z)):
        s = max(0.0, s + v - k)
        if s > h:
            out[i] = True
            s = 0.0                                   # restart after an alarm
    return out


def inject(s: SatSeries, kind: str, magnitude_ns: float, starts: np.ndarray, rng: np.random.Generator) -> SatSeries:
    """Return a copy of s with one fault of `kind` at every series index in `starts`."""
    bias = s.bias_ns.copy()
    for t0 in starts:
        sign = rng.choice([-1.0, 1.0])
        if kind == "spike":
            bias[t0] += sign * magnitude_ns
        elif kind == "step":
            bias[t0:] += sign * magnitude_ns
        elif kind == "ramp":
            n = KINDS["ramp"]
            ramp = np.cumsum(np.full(n, sign * magnitude_ns))          # extra slope of `magnitude` ns/sample
            bias[t0:t0 + n] += ramp
            bias[t0 + n:] += ramp[-1]
        else:
            raise ValueError(kind)
    return SatSeries(satellite=s.satellite, t=s.t, bias_ns=bias, drift=s.drift)


class _Scorer:
    """Fit on nominal windows once; score any series' one-step windows."""

    def __init__(self, w_train, seq_len):
        self.ridge = RidgeDirect(1).fit(w_train.X, w_train.y[:, :1], None, None)
        phys = w_train.y[:, 0] - w_train.X[:, -1, 1]
        self.phys_c, self.phys_s = _robust_scale(phys)
        ridge_err = w_train.y[:, 0] - self.ridge.predict(w_train.X)[:, 0]
        self.ridge_c, self.ridge_s = _robust_scale(ridge_err)
        self.iso = IsolationForest(n_estimators=100, random_state=0).fit(self._feat(w_train.X))
        self.iso_thr = float(np.quantile(-self.iso.score_samples(self._feat(w_train.X)), 0.995))

    @staticmethod
    def _adaptive(w, phys_z, window: int = 200, min_periods: int = 50):
        """Re-score the physics residual against a causal rolling median/MAD of the previous
        `window` residuals (current sample excluded); falls back to the static z-score early on."""
        e = pd.Series(phys_z)
        med = e.rolling(window, min_periods=min_periods).median().shift(1)
        mad = e.rolling(window, min_periods=min_periods).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True).shift(1)
        z = (e - med) / (_MAD * mad.clip(lower=1e-9))
        return z.fillna(e).to_numpy()

    @staticmethod
    def _feat(X):
        return X[:, -10:, [0, 2]].reshape(len(X), -1)

    def scores(self, w):
        phys = (w.y[:, 0] - w.X[:, -1, 1] - self.phys_c) / self.phys_s
        ridge = (w.y[:, 0] - self.ridge.predict(w.X)[:, 0] - self.ridge_c) / self.ridge_s
        iso = -self.iso.score_samples(self._feat(w.X))
        return {"physics_z": np.abs(phys), "physics_adaptive": np.abs(self._adaptive(w, phys)),
                "ridge_z": np.abs(ridge), "cusum": phys, "iforest": iso}

    def alarms(self, sc):
        return {"physics_z": sc["physics_z"] > Z_THRESHOLD, "physics_adaptive": sc["physics_adaptive"] > Z_THRESHOLD,
                "ridge_z": sc["ridge_z"] > Z_THRESHOLD,
                "cusum": _cusum_alarms(sc["cusum"]), "iforest": sc["iforest"] > self.iso_thr}


def _event_starts(w, test_lo: int, n_events: int, span: int, rng: np.random.Generator, spacing: int = 40):
    """Series indices (targets) usable as fault onsets: contiguous windows, well spaced."""
    ok = [j for j in range(test_lo, len(w.i) - span - 2)
          if w.i[j + span + 1] - w.i[j] == span + 1]
    if len(ok) < n_events * spacing:
        n_events = max(len(ok) // spacing, 1)
    picks, pool = [], sorted(ok)
    while pool and len(picks) < n_events:
        j = int(rng.choice(pool))
        picks.append(j)
        pool = [p for p in pool if abs(p - j) > spacing]
    return np.array(sorted(w.i[picks]))


def evaluate_injection(s: SatSeries, seq_len: int = 20, n_events: int = 8, trials: int = 5, seed: int = 0):
    """Return (per-event rows, clean-test false-alarm rows) for one satellite."""
    w = make_windows(s, seq_len, 1)
    n = len(w.i)
    tr_hi, te_lo = int(n * 0.6), int(n * 0.7)
    train = type(w)(X=w.X[:tr_hi], y=w.y[:tr_hi], i=w.i[:tr_hi], last_bias_ns=w.last_bias_ns[:tr_hi],
                    t_target=w.t_target[:tr_hi])
    scorer = _Scorer(train, seq_len)
    sigma_ns = scorer.phys_s
    rng = np.random.default_rng(seed)

    clean_sc = scorer.scores(w)
    clean_al = scorer.alarms(clean_sc)
    fa_rows = [dict(satellite=s.satellite, detector=d, fp_per_1000=1000 * float(clean_al[d][te_lo:].mean()),
                    n_windows=n - te_lo) for d in DETECTORS]

    rows = []
    for kind, span in KINDS.items():
        for mag in MAGNITUDES:
            for trial in range(trials):
                starts = _event_starts(w, te_lo, n_events, span, rng)
                s_inj = inject(s, kind, mag * sigma_ns, starts, rng)
                wi = make_windows(s_inj, seq_len, 1)
                assert np.array_equal(wi.i, w.i)                        # same gap structure
                al = scorer.alarms(scorer.scores(wi))
                for d in DETECTORS:
                    flagged = al[d]
                    pos = {int(i): j for j, i in enumerate(wi.i)}
                    in_event = np.zeros(len(wi.i), dtype=bool)
                    per_start = []
                    for t0 in starts:
                        idx = [pos[t] for t in range(t0, t0 + span + 1) if t in pos]
                        in_event[idx] = True
                        per_start.append((t0, [j for j in idx if flagged[j]]))
                    # Alarms in the held-out region that fall outside every injected event.
                    fp = int((flagged & ~in_event)[te_lo:].sum())
                    for t0, hit in per_start:
                        rows.append(dict(satellite=s.satellite, detector=d, kind=kind, magnitude_sigma=mag,
                                         trial=trial, detected=bool(hit), fp_in_trial=fp,
                                         delay=(wi.i[hit[0]] - t0) if hit else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(fa_rows)


def real_event_agreement(csv_path: str, seq_len: int = 20, margin: int = 2) -> pd.DataFrame:
    """(B) Flag events on the real series and test agreement with URA-index / health changes."""
    df = pd.read_csv(csv_path)
    series = load_series(csv_path)
    out = []
    for sat, s in sorted(series.items()):
        try:
            w = make_windows(s, seq_len, 1)
        except ValueError:
            continue
        n = len(w.i)
        scorer = _Scorer(type(w)(X=w.X[:int(n * .6)], y=w.y[:int(n * .6)], i=w.i[:int(n * .6)],
                                 last_bias_ns=w.last_bias_ns[:int(n * .6)], t_target=w.t_target[:int(n * .6)]),
                         seq_len)
        sc = scorer.scores(w)
        flagged = sc["physics_z"] > Z_THRESHOLD
        g = df[df["satellite_id"] == sat].sort_values("timestamp_s").reset_index(drop=True)
        ura_changed = (g["ura_index"].diff().fillna(0) != 0).to_numpy()
        unhealthy = (g["health"].fillna(0) != 0).to_numpy()
        meta = ura_changed | unhealthy
        near = np.zeros(len(g), dtype=bool)
        for d in range(-margin, margin + 1):
            near |= np.roll(meta, d)
        at = near[w.i]                                               # metadata event near each window
        test = slice(int(n * 0.7), n)
        f_test, a_test = flagged[test], at[test]
        out.append(dict(
            satellite=sat, n_test=int(f_test.size), n_flagged=int(f_test.sum()),
            flagged_share_near_meta=float(a_test[f_test].mean()) if f_test.any() else float("nan"),
            chance_share_near_meta=float(a_test.mean()),
            lift=(float(a_test[f_test].mean() / a_test.mean()) if f_test.any() and a_test.mean() > 0 else float("nan")),
            max_abs_z=float(sc["physics_z"][test].max())))
    return pd.DataFrame(out)


def run_anomaly_eval(csv_path: str, out_dir: str, seq_len: int = 20, n_events: int = 8, trials: int = 5,
                     min_windows: int = 400, verbose: bool = True) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    series = load_series(csv_path)
    ev, fa = [], []
    for sat, s in sorted(series.items()):
        try:
            e, f = evaluate_injection(s, seq_len, n_events, trials)
        except ValueError as err:
            if verbose:
                print(f"[anomaly] I{sat:02d}: skipped — {err}")
            continue
        if len(make_windows(s, seq_len, 1).i) < min_windows:
            continue
        ev.append(e)
        fa.append(f)
        if verbose:
            print(f"[anomaly] I{sat:02d}: {len(e)} injected-event scores")
    if not ev:
        raise ValueError("no satellite had enough windows for the anomaly evaluation")
    events, fas = pd.concat(ev), pd.concat(fa)
    events.to_csv(os.path.join(out_dir, "injection_events.csv"), index=False)
    fas.to_csv(os.path.join(out_dir, "false_alarms.csv"), index=False)
    real = real_event_agreement(csv_path, seq_len)
    real.to_csv(os.path.join(out_dir, "real_events.csv"), index=False)
    write_anomaly_summary(events, fas, real, out_dir)
    return events


def write_anomaly_summary(events: pd.DataFrame, fas: pd.DataFrame, real: pd.DataFrame, out_dir: str) -> None:
    rec = (events.groupby(["kind", "magnitude_sigma", "detector"])["detected"].mean()
           .unstack("detector")[list(DETECTORS)])
    delay = events.groupby(["detector"])["delay"].median()
    fa = fas.groupby("detector").apply(
        lambda g: np.average(g["fp_per_1000"], weights=g["n_windows"]), include_groups=False)
    trial = (events.groupby(["satellite", "detector", "kind", "magnitude_sigma", "trial"])
             .agg(tp=("detected", "sum"), n=("detected", "size"), fp=("fp_in_trial", "first")).reset_index())
    tot = trial.groupby(["kind", "magnitude_sigma", "detector"])[["tp", "n", "fp"]].sum()
    tot["precision"] = tot["tp"] / (tot["tp"] + tot["fp"]).clip(lower=1)
    tot["recall"] = tot["tp"] / tot["n"]
    tot["f1"] = 2 * tot["precision"] * tot["recall"] / (tot["precision"] + tot["recall"]).clip(lower=1e-9)
    f1 = tot["f1"].unstack("detector")[list(DETECTORS)]
    prec = tot["precision"].unstack("detector")[list(DETECTORS)]
    lines = ["# Anomaly-detector evaluation (real NavIC series, injected faults)\n",
             "Event recall by fault kind and magnitude (multiples of nominal residual sigma).\n",
             "| kind | magnitude (σ) | " + " | ".join(DETECTORS) + " |", "|---|---|" + "---|" * len(DETECTORS)]
    for (kind, mag), r in rec.iterrows():
        best = r.max()
        lines.append(f"| {kind} | {mag} | " + " | ".join(
            f"**{v:.2f}**" if np.isclose(v, best) else f"{v:.2f}" for v in r) + " |")
    lines += ["", "Precision and F1 (alarms outside injected events count as false positives; note the held-out data "
              "itself contains natural discontinuities, so these are lower bounds on true precision).\n",
              "| kind | magnitude (σ) | " + " | ".join(f"{d} P / F1" for d in DETECTORS) + " |",
              "|---|---|" + "---|" * len(DETECTORS)]
    for (kind, mag), r in f1.iterrows():
        lines.append(f"| {kind} | {mag} | " + " | ".join(
            f"{prec.loc[(kind, mag), d]:.2f} / {r[d]:.2f}" for d in DETECTORS) + " |")
    lines += ["", "False alarms on clean held-out data (per 1000 windows) and median detection delay (samples):\n",
              "| detector | false alarms / 1000 | median delay |", "|---|---|---|"]
    for d in DETECTORS:
        lines.append(f"| {d} | {fa[d]:.1f} | {delay.get(d, float('nan')):.1f} |")
    lines += ["", "## Real events vs. independent broadcast metadata (URA-index change / health flag, ±2 records)\n",
              "| satellite | flagged | share near metadata event | chance share | lift |", "|---|---|---|---|---|"]
    for _, r in real.iterrows():
        lines.append(f"| I{int(r.satellite):02d} | {int(r.n_flagged)}/{int(r.n_test)} | "
                     f"{r.flagged_share_near_meta:.2f} | {r.chance_share_near_meta:.2f} | {r.lift:.1f}× |")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
