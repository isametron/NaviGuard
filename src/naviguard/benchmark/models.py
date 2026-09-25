"""naviguard.benchmark.models — forecasters compared in the benchmark.

Every forecaster maps relative windows X (n, L, 3) to relative forecasts (n, horizon)
in nanoseconds. Classical models are closed-form / OLS; neural models (LSTM, GRU,
attention-LSTM) are Keras and standardise internally.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.linear_model import Ridge

CLASSICAL = ("persistence", "broadcast_drift", "linear_extrap", "arima_p10", "ridge")
NEURAL = ("lstm", "gru", "attn_lstm")
ALL_MODELS = CLASSICAL + NEURAL


class Forecaster:
    name = "base"

    def fit(self, X, y, Xv, yv):
        return self

    def predict(self, X):
        raise NotImplementedError


class Persistence(Forecaster):
    name = "persistence"

    def __init__(self, horizon):
        self.h = horizon

    def predict(self, X):
        return np.zeros((len(X), self.h))


class BroadcastDrift(Forecaster):
    """Extrapolate with the satellite's own broadcast drift (af1): the physical baseline."""
    name = "broadcast_drift"

    def __init__(self, horizon):
        self.h = horizon

    def predict(self, X):
        steps = np.arange(1, self.h + 1)[None, :]
        return X[:, -1, 1][:, None] * steps


class LinearExtrap(Forecaster):
    """Least-squares line through the last k bias samples, extended forward."""
    name = "linear_extrap"

    def __init__(self, horizon, k=8):
        self.h, self.k = horizon, k

    def predict(self, X):
        k = min(self.k, X.shape[1])
        y = X[:, -k:, 0]
        t = np.arange(k, dtype=np.float64)
        tc = t - t.mean()
        slope = (tc * (y - y.mean(1, keepdims=True))).sum(1) / (tc ** 2).sum()
        future = (k - 1) + np.arange(1, self.h + 1) - t.mean()
        return y.mean(1, keepdims=True) + slope[:, None] * future[None, :] - X[:, -1, 0][:, None]


class ARIMA(Forecaster):
    """ARIMA(p,1,0) fitted by OLS on first differences, forecast recursively."""
    name = "arima_p10"

    def __init__(self, horizon, p=10):
        self.h, self.p = horizon, p

    def _lagmat(self, d):
        # d: (n, m) diffs -> rows of p lags predicting the next diff, stacked across windows
        m = d.shape[1]
        rows = [d[:, j - self.p:j] for j in range(self.p, m)]
        tgt = [d[:, j] for j in range(self.p, m)]
        return np.concatenate(rows, 0), np.concatenate(tgt, 0)

    def fit(self, X, y, Xv, yv):
        self.p = min(self.p, X.shape[1] - 1)          # need at least one diff left to predict
        A, b = self._lagmat(X[:, :, 2])
        A1 = np.c_[A, np.ones(len(A))]
        self.coef = np.linalg.lstsq(A1, b, rcond=None)[0]
        return self

    def predict(self, X):
        hist = X[:, -self.p:, 2].copy()
        out = np.zeros((len(X), self.h))
        cum = np.zeros(len(X))
        for k in range(self.h):
            nxt = np.c_[hist, np.ones(len(hist))] @ self.coef
            cum = cum + nxt
            out[:, k] = cum
            hist = np.c_[hist[:, 1:], nxt]
        return out


class RidgeDirect(Forecaster):
    name = "ridge"

    def __init__(self, horizon, alpha=1.0):
        self.h, self.alpha = horizon, alpha

    def fit(self, X, y, Xv, yv):
        Xf = X.reshape(len(X), -1)
        self.mu, self.sd = Xf.mean(0), Xf.std(0) + 1e-9
        self.ysd = y.std(0) + 1e-9
        self.m = Ridge(alpha=self.alpha).fit((Xf - self.mu) / self.sd, y / self.ysd)
        return self

    def predict(self, X):
        Xf = X.reshape(len(X), -1)
        pred = self.m.predict((Xf - self.mu) / self.sd)
        return pred.reshape(len(X), -1) * self.ysd          # sklearn ravels single-target output


class Neural(Forecaster):
    def __init__(self, kind, horizon, seed=0, epochs=60, patience=8, batch=64):
        self.kind, self.h, self.seed = kind, horizon, seed
        self.epochs, self.patience, self.batch = epochs, patience, batch
        self.name = kind

    def _build(self, L, F):
        import tensorflow as tf

        from naviguard.models.attention import AttentionPooling
        K = tf.keras.layers
        inp = tf.keras.Input((L, F))
        if self.kind == "gru":
            x = K.GRU(64, return_sequences=True)(inp)
            x = K.Dropout(0.2)(x)
            x = K.GRU(32)(x)
        else:
            x = K.LSTM(64, return_sequences=True)(inp)
            x = K.Dropout(0.2)(x)
            if self.kind == "attn_lstm":
                x = K.LSTM(32, return_sequences=True)(x)
                x = K.Dropout(0.2)(x)
                x = AttentionPooling(32)(x)
            else:
                x = K.LSTM(32)(x)
        x = K.Dense(16, activation="relu")(x)
        out = K.Dense(self.h)(x)
        m = tf.keras.Model(inp, out)
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        return m

    def fit(self, X, y, Xv, yv):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.seed)
        self.mu = X.mean((0, 1))
        self.sd = X.std((0, 1)) + 1e-9
        self.ysd = y.std(0) + 1e-9
        self.model = self._build(X.shape[1], X.shape[2])
        cb = [tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True),
              tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=max(2, self.patience // 2), min_lr=1e-6)]
        self.model.fit(self._x(X), y / self.ysd, validation_data=(self._x(Xv), yv / self.ysd),
                       epochs=self.epochs, batch_size=self.batch, callbacks=cb, verbose=0)
        return self

    def _x(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)

    def predict(self, X):
        return self.model.predict(self._x(X), verbose=0) * self.ysd


def make_model(name: str, horizon: int, seed: int = 0, **kw) -> Forecaster:
    if name == "persistence":
        return Persistence(horizon)
    if name == "broadcast_drift":
        return BroadcastDrift(horizon)
    if name == "linear_extrap":
        return LinearExtrap(horizon)
    if name == "arima_p10":
        return ARIMA(horizon)
    if name == "ridge":
        return RidgeDirect(horizon)
    if name in NEURAL:
        return Neural(name, horizon, seed=seed, **kw)
    raise ValueError(f"unknown model {name!r}; choose from {ALL_MODELS}")
