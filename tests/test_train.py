"""End-to-end (tiny) train -> persist -> evaluate checks."""

import json

import numpy as np
import pytest

from naviguard.config import ModelHParams
from naviguard.data.generate import generate_dataset
from naviguard.inference import predict as P
from naviguard.inference.artifacts import Artifacts
from naviguard.models.lstm_attention import build_model
from naviguard.models.train import train
from naviguard.preprocessing.sequences import build_sequences, fit_scaler, save_sequences

TINY = ModelHParams(lstm_units=(8, 4), dense_units=4)


def test_train_records_split_sizes_and_meta(tmp_path):
    csv = str(tmp_path / "t.csv")
    df = generate_dataset(n_samples=300, out_path=csv)
    scaler_path = str(tmp_path / "scaler.pkl")
    scaler = fit_scaler(df, scaler_path=scaler_path)
    seqs = build_sequences(df, scaler, seq_len=8, horizon=3)
    seq_path = str(tmp_path / "seq.npz")
    save_sequences(seqs, seq_path)

    meta = train(
        seq_len=8, horizon=3, epochs=2, batch_size=32, patience=2, hparams=TINY,
        sequences_path=seq_path, scaler_path=scaler_path,
        model_path=str(tmp_path / "m.keras"), meta_path=str(tmp_path / "meta.json"),
        telemetry_path=csv, verbose=0,
    )

    assert meta["n_train"] == seqs.count("train")
    assert meta["n_val"] == seqs.count("val")
    assert meta["n_test"] == seqs.count("test")
    assert len(meta["test_mae_ns"]) == 3 and len(meta["persistence_mae_ns"]) == 3
    assert meta["telemetry_sha256"]
    saved = json.loads((tmp_path / "meta.json").read_text())
    assert saved["hparams"]["lstm_units"] == [8, 4]


def test_train_rejects_mismatched_window_config(tmp_path):
    df = generate_dataset(n_samples=200, out_path=str(tmp_path / "t.csv"))
    scaler = fit_scaler(df, scaler_path=None)
    save_sequences(build_sequences(df, scaler, 8, 3), str(tmp_path / "seq.npz"))
    with pytest.raises(ValueError, match="naviguard preprocess"):
        train(seq_len=10, horizon=3, epochs=1, sequences_path=str(tmp_path / "seq.npz"), verbose=0)


def test_evaluate_split_reports_baselines_and_caches(tmp_path, monkeypatch):
    csv = str(tmp_path / "t.csv")
    df = generate_dataset(n_samples=300, out_path=csv)
    scaler = fit_scaler(df, scaler_path=None)
    monkeypatch.setattr(P, "TELEMETRY_CSV", csv)
    arts = Artifacts(model=build_model(8, 3, 3, TINY), scaler=scaler,
                     meta={"seq_len": 8, "horizon": 3})

    res = P.evaluate_split(arts, "test")
    assert set(res["baselines"]) == {"persistence", "linear_extrapolation", "ridge"}
    assert res["split"] == "test" and np.isfinite(res["mae_ns"]).all()
    assert P.evaluate_split(arts, "test") is res            # cached
    det = P.detect_anomalies(arts)
    assert det["n_scored"] == res["n_test_samples"]
