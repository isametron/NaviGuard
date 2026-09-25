"""naviguard.data.broadcast — real NavIC (IRNSS) broadcast clock data.

Public source: the multi-GNSS broadcast-ephemeris product BRDM00DLR (DLR/GSOC,
RINEX 3.04, ~1.4 MB gzipped per day) mirrored at BKG. Each IRNSS navigation
record carries the satellite's own clock polynomial (af0, af1, af2) at its
reference epoch `toc`; for the continuously tracked satellites a new record
arrives every ~15 minutes, so af0 at each `toc` is directly a per-satellite
clock-bias series at the pipeline's native cadence.

This is the *broadcast* clock (what the satellite tells receivers), not a
precise post-processed clock — there is no public precise NavIC clock product.

Only the extracted IRNSS rows are cached (one small CSV per day); the raw
download is deleted unless keep_raw=True.
"""

import gzip
import os
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from naviguard.config import DATA_DIR

RAW_DIR = os.path.join(DATA_DIR, "raw")
BRDM_URL = ("https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy:03d}/"
            "BRDM00DLR_S_{year}{doy:03d}0000_01D_MN.rnx.gz")
_USER_AGENT = "naviguard/0.3 (research; contact via repo)"
_RECORD_LINES = 8     # IRNSS record = epoch/clock line + 7 broadcast-orbit lines

COLUMNS = ["satellite_id", "toc", "clock_bias_s", "clock_drift_s_per_s", "clock_drift_rate",
           "ura_index", "health", "tgd_s", "week"]


class BroadcastFetchError(RuntimeError):
    """Raised when a day's broadcast file cannot be downloaded."""


def _num(field: str) -> float:
    field = field.strip()
    return float(field.replace("D", "E")) if field else float("nan")


def _fields(line: str, start: int) -> list[float]:
    return [_num(line[i:i + 19]) for i in range(start, start + 76, 19)]


def parse_irnss_records(lines) -> pd.DataFrame:
    """Extract IRNSS records from RINEX 3 navigation text (iterable of lines).

    Record layout (RINEX 3.04): line 0 = `Iss yyyy mm dd hh mm ss af0 af1 af2`,
    then 7 orbit lines of 4 x D19.12 starting at column 4. Orbit-5 carries the
    IRN week, orbit-6 carries URA index / health / TGD.
    """
    it = iter(lines)
    for line in it:                                   # skip header
        if "END OF HEADER" in line:
            break
    rows = []
    for line in it:
        if not (len(line) > 23 and line[0] == "I" and line[1:3].isdigit()):
            continue
        block = [line] + [next(it, "") for _ in range(_RECORD_LINES - 1)]
        try:
            toc = datetime.strptime(line[4:23], "%Y %m %d %H %M %S")
            af0, af1, af2 = _fields(line, 23)[:3]
            orbit5 = _fields(block[5], 4)
            orbit6 = _fields(block[6], 4)
        except ValueError:
            continue                                   # malformed record — skip, never guess
        rows.append((int(line[1:3]), toc, af0, af1, af2, orbit6[0], orbit6[1], orbit6[2], orbit5[2]))
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.drop_duplicates(["satellite_id", "toc"]).sort_values(["satellite_id", "toc"]).reset_index(drop=True)


def _day_csv(day: date, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"irnss_{day:%Y%m%d}.csv")


def _download(url: str, dest: str, timeout_s: float, deadline_s: float) -> None:
    """Stream to dest.part, aborting if the whole transfer exceeds deadline_s
    (socket timeouts alone don't stop a server that trickles bytes)."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp, open(dest + ".part", "wb") as out:
        while chunk := resp.read(1 << 16):
            out.write(chunk)
            if time.monotonic() - start > deadline_s:
                raise TimeoutError(f"transfer exceeded {deadline_s:.0f}s")
    os.replace(dest + ".part", dest)


def fetch_day(day: date, cache_dir: str = RAW_DIR, keep_raw: bool = False, force: bool = False,
              timeout_s: float = 30.0, deadline_s: float = 90.0, retries: int = 2) -> pd.DataFrame:
    """Return the IRNSS records for one UTC day, downloading + caching if needed."""
    os.makedirs(cache_dir, exist_ok=True)
    csv_path = _day_csv(day, cache_dir)
    if os.path.exists(csv_path) and not force:
        return pd.read_csv(csv_path, parse_dates=["toc"])

    url = BRDM_URL.format(year=day.year, doy=day.timetuple().tm_yday)
    gz_path = os.path.join(cache_dir, os.path.basename(url))
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            _download(url, gz_path, timeout_s, deadline_s)
            break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            if os.path.exists(gz_path + ".part"):
                os.remove(gz_path + ".part")
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                break                                  # file not published (yet) — retrying won't help
    else:
        last_err = last_err or RuntimeError("unknown")
    if not os.path.exists(gz_path):
        raise BroadcastFetchError(f"could not fetch {url}: {last_err}") from last_err

    with gzip.open(gz_path, "rt", encoding="ascii", errors="replace") as f:
        df = parse_irnss_records(f)
    df.to_csv(csv_path, index=False)
    if not keep_raw:
        os.remove(gz_path)
    return df


def fetch_range(start: date, end: date, cache_dir: str = RAW_DIR, keep_raw: bool = False,
                force: bool = False) -> tuple[pd.DataFrame, list[date]]:
    """Fetch [start, end] inclusive. Days that fail are returned (not fatal)."""
    frames, failed = [], []
    day = start
    while day <= end:
        try:
            frames.append(fetch_day(day, cache_dir, keep_raw, force))
        except BroadcastFetchError as e:
            print(f"[fetch] {day}: {e}")
            failed.append(day)
        day += timedelta(days=1)
    if not frames:
        raise BroadcastFetchError(f"no data fetched for {start}..{end}")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["satellite_id", "toc"]).sort_values(["satellite_id", "toc"])
    return df.reset_index(drop=True), failed


def to_telemetry(records: pd.DataFrame, min_records: int = 48) -> pd.DataFrame:
    """Shape broadcast records into the pipeline's per-satellite telemetry schema.

    Adds sample_id / timestamp_s (seconds since the first record overall), drops
    satellites with fewer than `min_records` records (sparsely tracked PRNs).
    """
    counts = records.groupby("satellite_id").size()
    keep = counts[counts >= min_records].index
    df = records[records["satellite_id"].isin(keep)].copy()
    t0 = df["toc"].min()
    df["timestamp_s"] = (df["toc"] - t0).dt.total_seconds().astype(np.int64)
    df["sample_id"] = df.groupby("satellite_id").cumcount()
    cols = ["satellite_id", "sample_id", "timestamp_s", "clock_bias_s", "clock_drift_s_per_s",
            "ura_index", "health", "tgd_s"]
    return df[cols].reset_index(drop=True)
