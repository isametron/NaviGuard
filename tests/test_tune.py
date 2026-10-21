"""Tests for leak-free hyperparameter tuning and tuned benchmark runs (hermetic, no network)."""

import json

import numpy as np
import pandas as pd
import pytest

from naviguard.benchmark import tune
from naviguard.benchmark.models import Neural, make_model
from naviguard.benchmark.run import run_benchmark
from naviguard.benchmark.windows import SatSeries, make_windows, rolling_origin_folds

STEP = 912.0
SEQ, HOR, FOLDS = 10, 4, 2


def _series(sat, n=500, slope=30.0, seed=0):
    rng = np.random.default_rng(seed)
    bias = 5.6e5 + slope * np.arange(n) + np.cumsum(rng.normal(0, 0.3, n)) + rng.normal(0, 0.5, n)
    return SatSeries(satellite=sat, t=np.arange(n) * STEP, bias_ns=bias, drift=np.full(n, slope / STEP * 1e-9))


def _csv(path, poison_test=False):
    frames = []
    for k, sat in enumerate((2, 10)):
        s = _series(sat, seed=k, slope=30.0 + 10 * k)
        if poison_test:                  # NaN from the benchmark's first test target onward
            w = make_windows(s, SEQ, HOR)
            first = w.i[rolling_origin_folds(w, HOR, FOLDS)[0].test[0]]
            s.bias_ns[first:] = np.nan
        frames.append(pd.DataFrame({"satellite_id": sat, "timestamp_s": s.t, "clock_bias_s": s.bias_ns * 1e-9,
                                    "clock_drift_s_per_s": s.drift}))
    pd.concat(frames).to_csv(path, index=False)
    return str(path)


def test_tuning_never_reads_benchmark_test_data(tmp_path):
    csv = _csv(tmp_path / "t.csv", poison_test=True)
    out = tmp_path / "tune"
    result = tune.run_tuning(csv, str(out), models=("ridge",), seq_len=SEQ, horizon=HOR, n_folds=FOLDS,
                             min_windows=100, verbose=False)
    trials = pd.read_csv(out / "trials.csv")
    assert len(trials) == len(tune.RIDGE_ALPHAS) and np.isfinite(trials["score"]).all()   # NaN test data untouched
    assert result["models"]["ridge"]["alpha"] in tune.RIDGE_ALPHAS
    best = json.loads((out / "best.json").read_text())
    assert (best["horizon"], best["seq_len"]) == (HOR, SEQ)
    assert "ridge" in (out / "summary.md").read_text()


def test_settings_are_chosen_per_satellite_so_one_satellite_cannot_decide_for_all():
    p = [{"units": [8, 8]}, {"units": [16, 16]}]
    trials = pd.DataFrame([      # satellite 10's huge errors make setting 1 the best on average
        {"model": "lstm", "trial": 0, "score": 0.5, "params": json.dumps(p[0]),
         "per_satellite": json.dumps({"2": {"val_rel": 0.03}, "10": {"val_rel": 0.97}})},
        {"model": "lstm", "trial": 1, "score": 0.3, "params": json.dumps(p[1]),
         "per_satellite": json.dumps({"2": {"val_rel": 0.05}, "10": {"val_rel": 0.55}})},
    ])
    overall, per_sat = tune.select_best(trials, ("lstm",))
    assert overall == {"lstm": p[1]}
    assert per_sat == {"2": {"lstm": p[0]}, "10": {"lstm": p[1]}}


def test_neural_candidates_start_from_the_default_and_are_reproducible():
    c = tune.candidates("attn_lstm", 6, seq_len=20, seed=1)
    assert c[0] == tune.NEURAL_DEFAULT and len(c) == 6
    assert len({json.dumps(x, sort_keys=True) for x in c}) == 6
    assert all(x["lookback"] in (None, 10) for x in c)                # "half" resolved for seq_len 20
    assert c == tune.candidates("attn_lstm", 6, seq_len=20, seed=1)
    with pytest.raises(ValueError, match="no settings"):
        tune.candidates("persistence", 5, seq_len=20)


def test_resume_skips_finished_trials(tmp_path, monkeypatch):
    csv = _csv(tmp_path / "t.csv")
    out = str(tmp_path / "tune")
    kw = dict(models=("ridge",), seq_len=SEQ, horizon=HOR, n_folds=FOLDS, min_windows=100, verbose=False)
    first = tune.run_tuning(csv, out, **kw)

    def no_fits(*a, **k):
        raise AssertionError("a finished trial was refitted")
    monkeypatch.setattr(tune, "make_model", no_fits)
    assert tune.run_tuning(csv, out, resume=True, **kw) == first


def test_benchmark_applies_tuned_settings_and_checks_they_match(tmp_path):
    csv = _csv(tmp_path / "t.csv")
    kw = dict(models=("ridge",), seq_len=SEQ, horizon=HOR, n_folds=FOLDS, seeds=1, min_windows=100, verbose=False)
    default = run_benchmark(csv, str(tmp_path / "a"), **kw)
    tuned = run_benchmark(csv, str(tmp_path / "b"), params={"ridge": {"alpha": 1e4}}, **kw)
    assert not np.allclose(default["mae"], tuned["mae"])
    assert json.loads((tmp_path / "b" / "params.json").read_text()) == {"models": {"ridge": {"alpha": 1e4}},
                                                                         "per_satellite": {}}

    by_sat = run_benchmark(csv, str(tmp_path / "c"), params={"ridge": {"alpha": 1e4}},
                           params_by_sat={2: {"ridge": {"alpha": 1.0}}}, **kw)
    same_as_default = by_sat[by_sat.satellite == 2]["mae"].to_numpy()
    assert np.allclose(same_as_default, default[default.satellite == 2]["mae"])        # per-satellite entry wins
    assert np.allclose(by_sat[by_sat.satellite == 10]["mae"], tuned[tuned.satellite == 10]["mae"])

    (tmp_path / "best.json").write_text(json.dumps({"horizon": HOR, "seq_len": SEQ, "models": {"ridge": {}},
                                                    "per_satellite": {"2": {"ridge": {"alpha": 10.0}}}}))
    assert tune.load_params(str(tmp_path / "best.json"), HOR, SEQ) == ({"ridge": {}}, {"2": {"ridge": {"alpha": 10.0}}})
    with pytest.raises(ValueError, match="tuned for horizon"):
        tune.load_params(str(tmp_path / "best.json"), 24, 48)
    with pytest.raises(ValueError, match="takes no settings"):
        make_model("broadcast_drift", HOR, alpha=1.0)


def test_neural_settings_change_the_network_and_defaults_match_the_published_one():
    d = Neural("attn_lstm", 3)
    assert (d.units, d.dropout, d.lr, d.dense, d.batch, d.lookback) == ((64, 32), 0.2, 1e-3, 16, 64, None)

    w = make_windows(_series(2, n=200), 12, 3)
    m = Neural("gru", 3, epochs=1, units=[8, 4], dropout=0.0, lr=3e-3, dense=8, batch=32, lookback=6)
    m.fit(w.X[:120], w.y[:120], w.X[120:150], w.y[120:150])
    assert m.model.input_shape == (None, 6, 3)                      # only the last 6 records are seen
    assert m.predict(w.X[150:]).shape == (len(w.X[150:]), 3)
