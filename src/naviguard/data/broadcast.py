"""naviguard.data.broadcast — real NavIC (IRNSS) broadcast clock data.

Public source: the multi-GNSS broadcast-ephemeris product BRDM00DLR (DLR/GSOC,
RINEX 3.04, ~1.4 MB gzipped per day) mirrored at BKG and ESA. Each IRNSS navigation
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
from collections import Counter
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from naviguard.config import DATA_DIR

RAW_DIR = os.path.join(DATA_DIR, "raw")
_BRDM_FILE = "BRDM00DLR_S_{year}{doy:03d}0000_01D_MN.rnx.gz"
# The same DLR product at two IGS data centres. Their IRNSS records are identical (checked on
# 2026-08-01), so switching mirrors never changes results.
MIRRORS = {
    "bkg": "https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy:03d}/" + _BRDM_FILE,
    "esa": "ftp://gssc.esa.int/gnss/data/daily/{year}/brdc/" + _BRDM_FILE,
}
SOURCES = ("auto", *MIRRORS)    # auto: BKG first, ESA first for the rest of a run once BKG is down
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


def _missing(err: Exception) -> bool:
    """The server answered but has no file for that day: another mirror may, but this one isn't down."""
    if isinstance(err, urllib.error.HTTPError):
        return err.code == 404
    return "550" in str(getattr(err, "reason", err))          # FTP "file unavailable"


def _download_with_retries(url: str, dest: str, timeout_s: float, deadline_s: float, retries: int) -> None:
    last_err: Exception = RuntimeError("no download attempted")
    for _ in range(max(1, retries)):
        try:
            _download(url, dest, timeout_s, deadline_s)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            if os.path.exists(dest + ".part"):
                os.remove(dest + ".part")
            if _missing(e):
                break                                  # not published (yet): retrying won't help
    raise last_err


def fetch_day(day: date, cache_dir: str = RAW_DIR, keep_raw: bool = False, force: bool = False,
              timeout_s: float = 30.0, deadline_s: float = 90.0, retries: int = 2,
              mirrors: tuple[str, ...] = tuple(MIRRORS), on_down=None) -> pd.DataFrame:
    """Return the IRNSS records for one UTC day, downloading + caching if needed.

    Mirrors are tried in order. `on_down(name, error)` is called when a mirror is unreachable
    (as opposed to merely missing that day's file). The returned frame's attrs["source"] names
    the mirror that served it, or "cache".
    """
    os.makedirs(cache_dir, exist_ok=True)
    csv_path = _day_csv(day, cache_dir)
    if os.path.exists(csv_path) and not force:
        df = pd.read_csv(csv_path, parse_dates=["toc"])
        df.attrs["source"] = "cache"
        return df

    errors = []
    for name in mirrors:
        url = MIRRORS[name].format(year=day.year, doy=day.timetuple().tm_yday)
        gz_path = os.path.join(cache_dir, os.path.basename(url))
        try:
            _download_with_retries(url, gz_path, timeout_s, deadline_s, retries)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            errors.append(f"{name}: {e}")
            if on_down is not None and not _missing(e):
                on_down(name, e)
            continue
        with gzip.open(gz_path, "rt", encoding="ascii", errors="replace") as f:
            df = parse_irnss_records(f)
        df.to_csv(csv_path, index=False)
        if not keep_raw:
            os.remove(gz_path)
        df.attrs["source"] = name
        return df
    raise BroadcastFetchError(f"could not fetch {day} ({'; '.join(errors)})")


def fetch_range(start: date, end: date, cache_dir: str = RAW_DIR, keep_raw: bool = False,
                force: bool = False, source: str = "auto") -> tuple[pd.DataFrame, list[date]]:
    """Fetch [start, end] inclusive. Days that fail are returned (not fatal).

    source: "bkg" or "esa" downloads only from that mirror; "auto" tries BKG first and, as soon
    as a mirror turns out to be unreachable, tries the other one first for the rest of the range,
    so a dead mirror costs one timeout rather than one per day.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; choose from {', '.join(SOURCES)}")
    order = list(MIRRORS) if source == "auto" else [source]

    def demote(name, err):
        if len(order) > 1 and order[0] == name:
            order.append(order.pop(0))
            print(f"[fetch] {name.upper()} unreachable ({err}); trying {order[0].upper()} first from now on")

    frames, failed, used = [], [], Counter()
    day = start
    while day <= end:
        try:
            df = fetch_day(day, cache_dir, keep_raw, force, mirrors=tuple(order), on_down=demote)
            frames.append(df)
            used[df.attrs.get("source", "?")] += 1
        except BroadcastFetchError as e:
            print(f"[fetch] {e}")
            failed.append(day)
        day += timedelta(days=1)
    if used:
        print("[fetch] days by source: " + ", ".join(f"{k} {n}" for k, n in sorted(used.items())))
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
