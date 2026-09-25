"""Tests for the RINEX IRNSS broadcast-clock parser (hermetic — no network)."""

import numpy as np

from naviguard.data.broadcast import parse_irnss_records, to_telemetry

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
