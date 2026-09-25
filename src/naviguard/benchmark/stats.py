"""naviguard.benchmark.stats — significance testing for forecast comparisons."""

import numpy as np
from scipy import stats


def diebold_mariano(err_a: np.ndarray, err_b: np.ndarray, horizon: int = 1, power: int = 2) -> tuple[float, float]:
    """Diebold–Mariano test on paired per-window errors (Harvey small-sample correction).

    Returns (statistic, two-sided p). Negative statistic => model A has lower loss.
    Loss differential d = |e_a|^power - |e_b|^power, long-run variance via Newey–West
    with lag horizon-1 (overlapping multi-step forecasts are autocorrelated).
    """
    d = np.abs(err_a) ** power - np.abs(err_b) ** power
    n = len(d)
    if n < 8 or np.allclose(d, 0):
        return float("nan"), float("nan")
    dm = d - d.mean()
    lrv = float(np.dot(dm, dm) / n)
    for lag in range(1, horizon):
        w = 1 - lag / horizon
        lrv += 2 * w * float(np.dot(dm[lag:], dm[:-lag]) / n)
    if lrv <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / np.sqrt(lrv / n)
    k = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)   # Harvey et al. (1997)
    stat *= k
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


def block_bootstrap_ci(values: np.ndarray, block: int = 12, n_boot: int = 1000, alpha: float = 0.05,
                       seed: int = 0) -> tuple[float, float]:
    """Moving-block bootstrap CI for the mean of an autocorrelated error series."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n < 2:
        return float("nan"), float("nan")
    block = min(block, n)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = v[idx].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
