"""naviguard.benchmark.models — forecasters compared in the benchmark.

Every forecaster maps relative windows X (n, L, 3) to relative forecasts (n, horizon)
in nanoseconds. Classical models are closed-form / OLS, plus a two-state clock Kalman filter;
neural models (LSTM, GRU, attention-LSTM) are Keras and standardise internally.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

CLASSICAL = ("persistence", "broadcast_drift", "linear_extrap", "arima_p10", "kalman", "ridge")
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


class Kalman(Forecaster):
    """Two-state clock Kalman filter (bias + drift), the classical GNSS clock model.

    State [bias, drift] in ns and ns/step, propagated as bias += drift with white-FM (q1) and
    random-walk-FM (q2) process noise. The filter starts from the window's first two records and
    measures the bias at every record (noise r, floored at the broadcast af0 resolution). The
    broadcast drift enters once, as a measurement of the drift at the forecast origin (noise
    r_af1): af1 only changes at uploads, so treating each repeated value as a fresh measurement
    would let the fit claim it is exact. The four variances are fitted by maximum likelihood on
    the training windows; with fixed variances every window has the same gain sequence, so the
    filter runs vectorised over all windows at once.
    """
    name = "kalman"
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    R_FLOOR = (2.0 ** -31 * 1e9) ** 2 / 12               # ns^2: quantisation of the 2^-31 s af0 field

    def __init__(self, horizon, max_fit_windows=2000):
        self.h, self.max_fit = horizon, max_fit_windows
        self.log_var = None

    def _filter(self, X, log_var):
        """Filter every window; return the final states (n, 2) and the innovation negative log-likelihood."""
        q1, q2, r, r_af1 = np.exp(np.clip(log_var, -30.0, 30.0))
        r += self.R_FLOOR
        Q = np.array([[q1 + q2 / 3, q2 / 2], [q2 / 2, q2]])
        z = X[:, :, 0]                                      # measured bias, relative to the last record
        x = np.stack([z[:, 1], z[:, 1] - z[:, 0]], axis=1)  # two-point start
        P = np.array([[r, r], [r, 2 * r]])
        nll = 0.0
        for k in range(2, z.shape[1]):
            x = x @ self.F.T
            P = self.F @ P @ self.F.T + Q
            nll, x, P = self._update(nll, x, P, z[:, k], s=P[0, 0] + r, col=0)
        nll, x, _ = self._update(nll, x, P, X[:, -1, 1], s=P[1, 1] + r_af1, col=1)   # broadcast drift at the origin
        return x, nll

    @staticmethod
    def _update(nll, x, P, meas, s, col):
        """Scalar measurement of state `col` with innovation variance s; accumulates the Gaussian NLL."""
        e = meas - x[:, col]
        K = P[:, col] / s
        return (nll + 0.5 * (len(e) * np.log(s) + (e ** 2).sum() / s),
                x + e[:, None] * K[None, :], P - np.outer(K, P[col]))

    def fit(self, X, y, Xv, yv):
        sub = X[:: max(1, len(X) // self.max_fit)]
        v = float(np.median(np.diff(sub[:, :, 0], n=2, axis=1) ** 2)) + 1e-6           # bias noise scale
        v_af1 = float(np.median((sub[:, -1, 1] - sub[:, -1, 2]) ** 2)) + v            # af1 vs the last step
        x0 = np.log([v, v / 100, v, v_af1])
        simplex = np.vstack([x0, x0 + 2.0 * np.eye(4)])     # e^2 steps: the default 5% simplex barely moves in log space
        res = minimize(lambda th: self._filter(sub, th)[1], x0, method="Nelder-Mead",
                       options={"initial_simplex": simplex, "maxiter": 2000, "xatol": 1e-3, "fatol": 1e-3})
        self.log_var = res.x
        return self

    def predict(self, X):
        x, _ = self._filter(X, self.log_var)
        return x[:, :1] + x[:, 1:] * np.arange(1, self.h + 1)[None, :]


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
    """Two-layer LSTM / GRU, optionally with attention pooling. The defaults are the untuned
    configuration used in the published benchmark; `naviguard tune` searches the others.

    lookback: use only the last `lookback` records of each window (None = all), so the input length
    can be tuned without changing the windows every model is evaluated on.
    """

    def __init__(self, kind, horizon, seed=0, epochs=60, patience=8, batch=64, units=(64, 32), dropout=0.2,
                 lr=1e-3, dense=16, lookback=None):
        self.kind, self.h, self.seed = kind, horizon, seed
        self.epochs, self.patience, self.batch = epochs, patience, batch
        self.units, self.dropout, self.lr, self.dense = tuple(units), dropout, lr, dense
        self.lookback = lookback
        self.name = kind

    def _build(self, L, F):
        import tensorflow as tf

        from naviguard.models.attention import AttentionPooling
        K = tf.keras.layers
        u1, u2 = self.units
        inp = tf.keras.Input((L, F))
        if self.kind == "gru":
            x = K.GRU(u1, return_sequences=True)(inp)
            x = K.Dropout(self.dropout)(x)
            x = K.GRU(u2)(x)
        else:
            x = K.LSTM(u1, return_sequences=True)(inp)
            x = K.Dropout(self.dropout)(x)
            if self.kind == "attn_lstm":
                x = K.LSTM(u2, return_sequences=True)(x)
                x = K.Dropout(self.dropout)(x)
                x = AttentionPooling(u2)(x)
            else:
                x = K.LSTM(u2)(x)
        x = K.Dense(self.dense, activation="relu")(x)
        out = K.Dense(self.h)(x)
        m = tf.keras.Model(inp, out)
        m.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss="mse")
        return m

    def _cut(self, X):
        return X[:, -self.lookback:, :] if self.lookback else X

    def fit(self, X, y, Xv, yv):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.seed)
        X, Xv = self._cut(X), self._cut(Xv)
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
        return self.model.predict(self._x(self._cut(X)), verbose=0) * self.ysd


TUNABLE = ("ridge", *NEURAL)     # models with settings for `naviguard tune`; the others have none


def make_model(name: str, horizon: int, seed: int = 0, **kw) -> Forecaster:
    if kw and name not in TUNABLE:
        raise ValueError(f"{name} takes no settings (got {sorted(kw)}); tunable models: {TUNABLE}")
    if name == "persistence":
        return Persistence(horizon)
    if name == "broadcast_drift":
        return BroadcastDrift(horizon)
    if name == "linear_extrap":
        return LinearExtrap(horizon)
    if name == "arima_p10":
        return ARIMA(horizon)
    if name == "kalman":
        return Kalman(horizon)
    if name == "ridge":
        return RidgeDirect(horizon, **kw)
    if name in NEURAL:
        return Neural(name, horizon, seed=seed, **kw)
    raise ValueError(f"unknown model {name!r}; choose from {ALL_MODELS}")
