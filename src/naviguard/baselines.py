"""naviguard.baselines — classical forecasters the attention-LSTM must beat.

All operate on the scaled windows/targets produced by preprocessing (the
scaling is affine on the target column, so error ratios are unit-independent)
and return (n, horizon) predictions in scaled space.
"""

import numpy as np
from sklearn.linear_model import Ridge

from naviguard.config import TARGET_IDX


def persistence(X: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed clock bias for every horizon step."""
    return np.repeat(X[:, -1, TARGET_IDX][:, None], horizon, axis=1)


def linear_extrapolation(X: np.ndarray, horizon: int, lookback: int = 8) -> np.ndarray:
    """Least-squares line through the last `lookback` clock-bias points, extended forward."""
    k = min(lookback, X.shape[1])
    y = X[:, -k:, TARGET_IDX].astype(np.float64)
    t = np.arange(k, dtype=np.float64)
    t_mean = t.mean()
    slope = ((t - t_mean) * (y - y.mean(axis=1, keepdims=True))).sum(axis=1) / ((t - t_mean) ** 2).sum()
    steps = np.arange(1, horizon + 1, dtype=np.float64)
    future_t = (k - 1) + steps                                   # (horizon,)
    return y.mean(axis=1, keepdims=True) + slope[:, None] * (future_t - t_mean)[None, :]


def fit_ridge(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1e-3) -> Ridge:
    """Ridge regression on the flattened window — a strong linear multi-step baseline."""
    return Ridge(alpha=alpha).fit(X_train.reshape(len(X_train), -1), y_train)


def predict_ridge(model: Ridge, X: np.ndarray) -> np.ndarray:
    return model.predict(X.reshape(len(X), -1))
