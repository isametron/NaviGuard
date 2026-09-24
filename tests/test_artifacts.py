"""Tests for artifact validation and CLI error handling."""

import pytest

from naviguard.cli import main
from naviguard.errors import ArtifactsInvalidError
from naviguard.inference.artifacts import _validate


class _Model:
    input_shape = (None, 20, 3)
    output_shape = (None, 6)


def test_validate_accepts_consistent_meta():
    _validate(_Model(), {"seq_len": 20, "horizon": 6})
    _validate(_Model(), {})


def test_validate_rejects_seq_len_mismatch():
    with pytest.raises(ArtifactsInvalidError, match="seq_len"):
        _validate(_Model(), {"seq_len": 10, "horizon": 6})


def test_validate_rejects_horizon_mismatch():
    with pytest.raises(ArtifactsInvalidError, match="horizon"):
        _validate(_Model(), {"seq_len": 20, "horizon": 3})


def test_cli_reports_missing_telemetry_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("naviguard.config.TELEMETRY_CSV", str(tmp_path / "missing.csv"))
    rc = main(["preprocess", "--telemetry", str(tmp_path / "missing.csv")])
    assert rc == 1
    assert "Telemetry CSV not found" in capsys.readouterr().err
