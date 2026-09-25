"""naviguard.benchmark.windows — level-free windows and rolling-origin folds for real clock series.

Real broadcast clock biases sit at hundreds of microseconds with nanosecond-scale
dynamics, so a raw MinMax scaling wastes float precision and lets the level drift
dominate. Every window is therefore expressed *relative to its last observed bias*:

    X[..., 0] = bias[j] - bias[last]                 (ns)
    X[..., 1] = broadcast drift (af1) x step          (ns per step)
    X[..., 2] = first difference of bias              (ns)
    y[h]      = bias[i+h] - bias[last]                (ns), h = 0..horizon-1

Windows never span a data gap (consecutive samples further apart than
`max_gap_factor` x the median cadence).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_CHANNELS = 3


@dataclass
class SatSeries:
    satellite: int
    t: np.ndarray            # seconds
    bias_ns: np.ndarray      # float64, absolute
    drift: np.ndarray        # s/s (broadcast af1)

    @property
    def step_s(self) -> float:
        return float(np.median(np.diff(self.t)))


@dataclass
class Windows:
    X: np.ndarray            # (n, seq_len, N_CHANNELS) float64, ns
    y: np.ndarray            # (n, horizon) float64, ns relative to last observed bias
    i: np.ndarray            # index of the first target sample in the series
    last_bias_ns: np.ndarray  # (n,) absolute bias of the last input sample
    t_target: np.ndarray     # (n,) time of the first target


@dataclass
class Fold:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def load_series(path: str) -> dict[int, SatSeries]:
    df = pd.read_csv(path)
    need = {"satellite_id", "timestamp_s", "clock_bias_s", "clock_drift_s_per_s"}
    if not need <= set(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(need)}")
    out = {}
    for sat, g in df.groupby("satellite_id"):
        g = g.sort_values("timestamp_s")
        out[int(sat)] = SatSeries(
            satellite=int(sat),
            t=g["timestamp_s"].to_numpy(dtype=np.float64),
            bias_ns=g["clock_bias_s"].to_numpy(dtype=np.float64) * 1e9,
            drift=g["clock_drift_s_per_s"].to_numpy(dtype=np.float64),
        )
    return out


def make_windows(s: SatSeries, seq_len: int, horizon: int, max_gap_factor: float = 1.5) -> Windows:
    n = len(s.t)
    step = s.step_s
    gap_bad = np.r_[0, (np.diff(s.t) > max_gap_factor * step).astype(int)]   # gap_bad[k]: gap into sample k
    bad_cum = np.concatenate([[0], np.cumsum(gap_bad)])                       # bad_cum[k] = sum(gap_bad[:k])

    starts = np.arange(seq_len, n - horizon + 1)
    # A window uses samples i-seq_len .. i+horizon-1; gaps *into* i-seq_len+1 .. i+horizon-1 must be clean.
    lo, hi = starts - seq_len + 1, starts + horizon                            # gap_bad[lo:hi]
    starts = starts[(bad_cum[hi] - bad_cum[lo]) == 0]
    if starts.size == 0:
        raise ValueError(f"satellite {s.satellite}: no gap-free windows for seq_len={seq_len}, horizon={horizon}")

    diff = np.r_[0.0, np.diff(s.bias_ns)]
    drift_step = s.drift * step * 1e9
    idx_in = starts[:, None] + np.arange(-seq_len, 0)[None, :]
    idx_out = starts[:, None] + np.arange(horizon)[None, :]
    last = s.bias_ns[starts - 1]
    X = np.stack([s.bias_ns[idx_in] - last[:, None], drift_step[idx_in], diff[idx_in]], axis=-1)
    y = s.bias_ns[idx_out] - last[:, None]
    return Windows(X=X, y=y, i=starts, last_bias_ns=last, t_target=s.t[starts])


def rolling_origin_folds(w: Windows, horizon: int, n_folds: int = 3, test_frac: float = 0.4,
                         val_frac: float = 0.15) -> list[Fold]:
    """Expanding-window, purged folds: the last `test_frac` of windows is split into
    `n_folds` consecutive test blocks; each fold trains on windows whose targets end
    before the block's first target, with the newest `val_frac` held out for early stopping."""
    n = len(w.i)
    edges = np.linspace(int(n * (1 - test_frac)), n, n_folds + 1, dtype=int)
    folds = []
    for f in range(n_folds):
        test = np.arange(edges[f], edges[f + 1])
        if len(test) == 0:
            raise ValueError("not enough data for the requested number of folds; use fewer folds or more days")
        pool =np.flatnonzero(w.i + horizon - 1 < w.i[test[0]])
        n_val = max(int(len(pool) * val_frac), 1)
        val = pool[-n_val:]
        train = pool[:-n_val]
        train = train[w.i[train] + horizon - 1 < w.i[val[0]]]
        if len(train) < 10 or len(test) == 0:
            raise ValueError("not enough data for the requested number of folds; use fewer folds or more days")
        folds.append(Fold(train=train, val=val, test=test))
    return folds


def relative_window(bias_ns: np.ndarray, drift: np.ndarray, step_s: float) -> np.ndarray:
    """Build one level-free input window (1, L, 3) from raw chronological rows.

    Same channels as make_windows; the first first-difference is 0 because the sample before
    the window is not available at serving time.
    """
    bias_ns = np.asarray(bias_ns, dtype=np.float64)
    drift = np.asarray(drift, dtype=np.float64)
    diff = np.r_[0.0, np.diff(bias_ns)]
    return np.stack([bias_ns - bias_ns[-1], drift * step_s * 1e9, diff], axis=-1)[None]


def split_windows(w: Windows, horizon: int, train: float = 0.70, val: float = 0.15) -> dict[str, np.ndarray]:
    """Chronological, purged train/val/test index sets over one satellite's windows: a window
    belongs to a split only if its targets end before the next split's first target."""
    n = len(w.i)
    tr_hi, va_hi = int(n * train), int(n * (train + val))
    ends = w.i + horizon - 1
    tr = np.arange(0, tr_hi)
    va = np.arange(tr_hi, va_hi)
    te = np.arange(va_hi, n)
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError("not enough windows to split into train/val/test")
    return {"train": tr[ends[tr] < w.i[va[0]]], "val": va[ends[va] < w.i[te[0]]], "test": te}
