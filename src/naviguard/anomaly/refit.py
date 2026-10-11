"""naviguard.anomaly.refit — refit-aware anomaly detection for broadcast clocks.

Broadcast clock residuals contain frequent large jumps caused by routine ground-segment refits
of the clock model. On real NavIC data most of these are *isolated level shifts*: the bias jumps
once and the series then continues smoothly. A plain |z| threshold flags every one of them.

This detector looks a few records past each large residual and classifies its shape:

    spike      the next residual reverts the jump almost fully (a single bad sample)  -> alarm
    sustained  several following residuals stay large with the same sign (a drift
               excursion)                                                              -> alarm
    refit      anything else: an isolated level shift or a partial reversion           -> no alarm
    edge       a data gap or the end of the series inside the look-ahead               -> no alarm

Alarms are raised at the record where the shape is confirmed, so the detection delay is real
(1 record for a spike, up to `look` records for a sustained excursion). A persistent step fault
is indistinguishable from a refit by shape alone and is deliberately not flagged.
"""

import numpy as np

from naviguard.anomaly.detect import classify_severity

SPIKE, SUSTAINED, REFIT, EDGE = "spike", "sustained", "refit", "edge"


def classify_jumps(z: np.ndarray, contiguous: np.ndarray, thr: float = 4.0, low: float = 2.0,
                   look: int = 3, revert_band: tuple[float, float] = (0.7, 1.3)):
    """Classify every |z| > thr jump in a robust-z residual series.

    z           signed robust z-scores of one-step residuals, in time order
    contiguous  contiguous[j] is True when sample j+1 directly follows sample j (no gap)

    Returns (alarms, events): a boolean alarm array aligned with z, and a list of
    (onset_index, label, confirm_index) tuples, one per jump.
    """
    z = np.asarray(z, dtype=np.float64)
    contiguous = np.asarray(contiguous, dtype=bool)
    n = len(z)
    alarms = np.zeros(n, dtype=bool)
    events = []
    lo, hi = revert_band
    consumed_until = -1                          # samples already explained by an earlier spike/excursion
    for j in np.flatnonzero(np.abs(z) > thr):
        if j <= consumed_until:
            continue
        if j + look >= n or not contiguous[j:j + look].all():
            events.append((int(j), EDGE, -1))
            continue
        nxt = z[j + 1:j + 1 + look]
        sign = np.sign(z[j])
        ratio = -nxt[0] / z[j]
        if abs(nxt[0]) > low and lo <= ratio <= hi:
            label, confirm = SPIKE, j + 1
        else:
            same = np.flatnonzero((np.abs(nxt) > low) & (np.sign(nxt) == sign))
            if len(same) >= 2:
                label, confirm = SUSTAINED, j + 1 + int(same[1])
            else:
                label, confirm = REFIT, -1
        if confirm >= 0:
            alarms[confirm] = True
            consumed_until = confirm             # the reversion / excursion samples are part of this event
        events.append((int(j), label, int(confirm)))
    return alarms, events


def shape_counts(events) -> dict[str, int]:
    out = {SPIKE: 0, SUSTAINED: 0, REFIT: 0, EDGE: 0}
    for _, label, _ in events:
        out[label] += 1
    return out


def summarize(z: np.ndarray, events, z_threshold: float, satellite_ids=None) -> dict:
    """API-ready summary of classify_jumps output.

    `events` index into `z`, which may concatenate several satellites (classify each satellite
    separately and offset its indices); satellite_ids[k] names the satellite of sample k.
    Severity uses the same rules as the plain detector, counting only alarmed jumps.
    """
    z = np.asarray(z, dtype=np.float64)
    alarmed = [(j, c) for j, _, c in events if c >= 0]
    max_z = max((abs(float(z[j])) for j, _ in alarmed), default=0.0)
    return {
        "z_threshold": float(z_threshold),
        "n_jumps": len(events),
        "n_alarms": len(alarmed),
        "alarm_rate": len(alarmed) / len(z) if len(z) else 0.0,
        "max_alarm_abs_z": max_z,
        "severity": classify_severity(len(alarmed), len(z), max_z, z_threshold),
        "shape_counts": shape_counts(events),
        "alarm_indices": sorted(c for _, c in alarmed),
        "z_scores": z.tolist(),
        "events": [{"index": j, "label": label, "alarm_index": c if c >= 0 else None, "z": float(z[j]),
                    "satellite_id": int(satellite_ids[j]) if satellite_ids is not None else None}
                   for j, label, c in events],
    }
