"""Tests for the RINEX IRNSS broadcast-clock parser and mirror switching (hermetic — no network)."""

import gzip
import urllib.error
from datetime import date

import numpy as np
import pytest

from naviguard.data import broadcast
from naviguard.data.broadcast import BroadcastFetchError, fetch_range, parse_irnss_records, to_telemetry

HEADER = "     3.04           NAVIGATION DATA     M                   RINEX VERSION / TYPE\n" \
         "                                                            END OF HEADER\n"

# Two real IRNSS records (I02 @ 00:05:36, 00:20:48) and one GPS record that must be ignored.
I02_A = """I02 2026 01 01 00 05 36-3.376686945558e-04 4.183675628155e-11 0.000000000000e+00
     1.610000000000e+02 2.146250000000e+02 2.937265206029e-09 1.741935593489e+00
     6.787478923798e-06 1.915221335366e-03 2.107396721840e-05 6.493439237595e+03
     3.459360000000e+05 1.974403858185e-07 2.560269994002e+00-7.823109626770e-08
     5.099907901349e-01-5.565625000000e+02 3.036225532741e+00-2.387242295367e-09
    -1.035400271434e-09                    2.399000000000e+03
     2.000000000000e+00 0.000000000000e+00-1.862645149231e-09
     3.460920000000e+05
"""
I02_B = I02_A.replace("00 05 36-3.376686945558e-04", "00 20 48-3.376314416528e-04")
GPS = """G01 2026 01 01 00 00 00 1.000000000000e-04 2.000000000000e-12 0.000000000000e+00
     1.000000000000e+01 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00
     0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00
     0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00
     0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00
     0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00
     0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00
     0.000000000000e+00 0.000000000000e+00
"""


def _parse(text):
    return parse_irnss_records((HEADER + text).splitlines())


def test_parses_clock_fields_and_ignores_other_systems():
    df = _parse(GPS + I02_A + I02_B)
    assert list(df["satellite_id"]) == [2, 2]
    first = df.iloc[0]
    assert first["clock_bias_s"] == np.float64(-3.376686945558e-04)
    assert first["clock_drift_s_per_s"] == np.float64(4.183675628155e-11)
    assert first["ura_index"] == 2.0 and first["health"] == 0.0
    assert first["tgd_s"] == np.float64(-1.862645149231e-09)
    assert first["week"] == 2399.0
    assert str(first["toc"]) == "2026-01-01 00:05:36"


def test_deduplicates_and_sorts():
    df = _parse(I02_B + I02_A + I02_A)
    assert len(df) == 2
    assert df["toc"].is_monotonic_increasing


def test_malformed_record_is_skipped_not_guessed():
    bad = I02_A.replace("2026 01 01", "2026 XX 01")
    df = _parse(bad + I02_B)
    assert len(df) == 1


def test_handles_fortran_d_exponents():
    df = _parse(I02_A.replace("e-04", "D-04"))
    assert df.iloc[0]["clock_bias_s"] == np.float64(-3.376686945558e-04)


def test_to_telemetry_drops_sparse_satellites_and_indexes_samples():
    df = _parse(I02_A + I02_B)
    sparse = df.copy()
    sparse["satellite_id"] = 9
    both = __import__("pandas").concat([df, sparse.iloc[:1]], ignore_index=True)
    tel = to_telemetry(both, min_records=2)
    assert set(tel["satellite_id"]) == {2}
    assert list(tel["sample_id"]) == [0, 1]
    assert tel["timestamp_s"].iloc[1] == 912


# ── mirror switching ──────────────────────────────────────────────────────────
DAYS = (date(2026, 8, 1), date(2026, 8, 3))


def _fake_mirrors(monkeypatch, down=(), missing=()):
    """Replace downloads: mirrors in `down` time out, (mirror, day-of-year) pairs in `missing` 404."""
    calls = []

    def fake_download(url, dest, timeout_s, deadline_s):
        name = "bkg" if "bkg.bund.de" in url else "esa"
        doy = int(url.rsplit("_S_", 1)[1][4:7])
        calls.append((name, doy))
        if name in down:
            raise urllib.error.URLError(TimeoutError("timed out"))
        if (name, doy) in missing:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        with gzip.open(dest, "wt", encoding="ascii") as f:
            f.write(HEADER + I02_A.replace("2026 01 01", f"2026 08 {doy - 212:02d}"))

    monkeypatch.setattr(broadcast, "_download", fake_download)
    return calls


def test_auto_switches_to_esa_once_bkg_is_down_and_stays_there(monkeypatch, tmp_path, capsys):
    calls = _fake_mirrors(monkeypatch, down={"bkg"})
    records, failed = fetch_range(*DAYS, cache_dir=str(tmp_path))
    assert failed == [] and len(records) == 3
    assert [c for c in calls if c[0] == "bkg"] == [("bkg", 213), ("bkg", 213)]   # one day's retries, then never again
    assert "esa 3" in capsys.readouterr().out


def test_fixed_source_never_falls_back(monkeypatch, tmp_path):
    calls = _fake_mirrors(monkeypatch, down={"bkg"})
    with pytest.raises(BroadcastFetchError, match="no data fetched"):
        fetch_range(*DAYS, cache_dir=str(tmp_path), source="bkg")
    assert {c[0] for c in calls} == {"bkg"}
    calls.clear()
    records, failed = fetch_range(*DAYS, cache_dir=str(tmp_path), source="esa")
    assert failed == [] and {c[0] for c in calls} == {"esa"}


def test_a_missing_file_borrows_from_the_other_mirror_without_switching(monkeypatch, tmp_path, capsys):
    calls = _fake_mirrors(monkeypatch, missing={("bkg", 213)})
    fetch_range(*DAYS, cache_dir=str(tmp_path))
    assert calls == [("bkg", 213), ("esa", 213), ("bkg", 214), ("bkg", 215)]
    assert "bkg 2, esa 1" in capsys.readouterr().out


def test_cached_days_need_no_mirror_and_unknown_source_is_rejected(monkeypatch, tmp_path):
    _fake_mirrors(monkeypatch)
    fetch_range(*DAYS, cache_dir=str(tmp_path))
    calls = _fake_mirrors(monkeypatch, down={"bkg", "esa"})
    records, failed = fetch_range(*DAYS, cache_dir=str(tmp_path))
    assert failed == [] and calls == []
    with pytest.raises(ValueError, match="unknown source"):
        fetch_range(*DAYS, cache_dir=str(tmp_path), source="cddis")
