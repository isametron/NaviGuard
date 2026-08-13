"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """A small, deterministic telemetry-shaped DataFrame for preprocessing tests."""
    n = 60
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "sample_id": range(n),
        "timestamp_s": [i * 900 for i in range(n)],
        "clock_bias_s": rng.normal(0, 1e-6, n),
        "clock_drift_s_per_s": rng.normal(0, 1e-10, n),
        "ephemeris_error_m": rng.normal(0, 0.3, n),
    })


class FakeScaler:
    """Identity-transform stand-in for a fitted MinMaxScaler in API tests."""

    def transform(self, values):
        return np.asarray(values, dtype=np.float32)

    def inverse_transform(self, values):
        return np.asarray(values, dtype=np.float32)


@pytest.fixture
def fake_scaler() -> FakeScaler:
    return FakeScaler()
