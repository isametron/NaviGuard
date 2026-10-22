"""UI building blocks for the NaviGuard dashboard.

Everything here is a plain function that returns an HTML or SVG string. Nothing imports Streamlit,
so each piece can be tested on its own. dashboard.py hands the strings to
st.markdown(..., unsafe_allow_html=True). The look itself lives in style.css.
"""
from __future__ import annotations

import html
import math
import re

import numpy as np
import pandas as pd

TEAL = "#0F5B5C"
INK = "#0E1B2A"
SLATE = "#5B6B78"
GREY = "#8A97A3"
ALARM = "#C2412D"
PEACH = "#D98B4F"
LINE_COLORS = [TEAL, ALARM, "#B7791F", "#3B6EA5"]


# ── small helpers ─────────────────────────────────────────────────────────────
def esc(x) -> str:
    return html.escape(str(x))


def compact(s: str) -> str:
    """Remove whitespace between tags so Markdown never sees blank lines or indented code blocks."""
    return re.sub(r">\s+<", "><", s.replace("\n", " ")).strip()


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def num(text) -> str:
    """Inline figure in the mono face (used inside hero sentences)."""
    return f'<span class="ng-num">{esc(text)}</span>'


def broadcast_deviation_ns(df: pd.DataFrame) -> tuple[pd.Series, list[int]]:
    """Real clock bias minus the line its own broadcast drift (af1) predicts, in ns from the first row.

    It would stay at 0 if af1 explained every step; routine ground-segment refits show up as steps.
    Also returns the rows whose single-step miss exceeds 4 robust sigmas (a local display rule; the
    anomaly report does the actual refit/fault classification).
    """
    bias_ns = df["clock_bias_s"].to_numpy(dtype=float) * 1e9
    predicted = df["clock_drift_s_per_s"].to_numpy(dtype=float)[:-1] * np.diff(df["timestamp_s"].to_numpy(dtype=float))
    miss = np.diff(bias_ns) - predicted * 1e9
    deviation = pd.Series(np.r_[0.0, np.cumsum(miss)])
    if len(miss) < 3:
        return deviation, []
    center = np.median(miss)
    sigma = 1.4826 * np.median(np.abs(miss - center))
    jumps = (np.flatnonzero(np.abs(miss - center) > 4 * sigma) + 1).tolist() if sigma > 0 else []
    return deviation, jumps


def time_axis(df: pd.DataFrame, n_ticks: int = 5):
    """x tick labels (hours before the latest sample) and per-point tooltip names for a telemetry frame."""
    n = len(df)
    if n < 2:
        return [], None
    if "timestamp_s" in df.columns:
        t = (df["timestamp_s"].to_numpy(dtype=float) - float(df["timestamp_s"].iloc[-1])) / 3600.0
        names = ["now" if abs(v) < 1e-9 else f"{v:.1f} h" for v in t]
        idx = np.linspace(0, n - 1, n_ticks).round().astype(int)
        ticks = [(int(i), "now" if i == n - 1 else f"\u2212{abs(t[i]):.0f} h") for i in idx]
        return ticks, names
    idx = np.linspace(0, n - 1, n_ticks).round().astype(int)
    return [(int(i), f"#{i}") for i in idx], [f"sample {i}" for i in range(n)]


# ── charts (inline SVG) ───────────────────────────────────────────────────────
def _nice_ticks(lo: float, hi: float, n: int = 5):
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return [0.0, 1.0], 1.0
    if hi - lo < 1e-12:
        pad = abs(hi) * 0.1 or 1.0
        lo, hi = lo - pad, hi + pad
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = mag * 10
    for m in (1, 2, 5, 10):
        if m * mag >= raw:
            step = m * mag
            break
    v = math.floor(lo / step) * step
    ticks = []
    while len(ticks) < 14:
        ticks.append(round(v, 10))
        if v >= hi - 1e-12:
            break
        v += step
    return ticks, step


def _fmt_tick(v: float, decimals: int) -> str:
    s = f"{v:,.{decimals}f}"
    return "0" if s.lstrip("-").strip("0.,") == "" else s


def chart(series, *, height=300, x_ticks=(), x_names=None, y_unit="", hlines=(), dots=(),
          fill=True, uid="c", aria="chart", width=960, font=11.5) -> str:
    """Line/area chart as inline SVG.

    series: [{"name", "values", "color", "dash": bool}]; the first series gets the gradient area.
    x_ticks: [(index, label)]; x_names: per-point tooltip names.
    hlines: [{"y", "label", "color"}]; dots: [{"series": 0, "idx": [...], "color", "label"}].
    """
    series = [s for s in series if len(s["values"]) >= 2]
    if not series:
        return ""
    L, R, T, B = (48, 12, 26, 30) if width < 500 else (60, 18, 26, 34)
    W, H = width, height
    n = max(len(s["values"]) for s in series)
    vals = [float(v) for s in series for v in s["values"]] + [float(h["y"]) for h in hlines]
    ticks, step = _nice_ticks(min(vals), max(vals))
    t0, t1 = ticks[0], ticks[-1]
    dec = max(0, -math.floor(math.log10(step))) if step > 0 else 0

    def px(i):
        return L + (W - L - R) * i / (n - 1)

    def py(v):
        return T + (H - T - B) * (1 - (float(v) - t0) / (t1 - t0))

    el = []
    first_color = series[0].get("color", TEAL)
    el.append(f'<defs><linearGradient id="{uid}-g" x1="0" y1="0" x2="0" y2="1">'
              f'<stop offset="0" stop-color="{first_color}" stop-opacity="0.26"/>'
              f'<stop offset="1" stop-color="{first_color}" stop-opacity="0.02"/></linearGradient></defs>')
    for tk in ticks:
        y = py(tk)
        el.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="rgba(14,27,42,0.08)" stroke-width="1"/>')
        el.append(f'<text x="{L - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="{font}" fill="{SLATE}">{_fmt_tick(tk, dec)}</text>')
    for idx, label in x_ticks:
        anchor = "start" if idx == 0 else ("end" if idx == n - 1 else "middle")
        el.append(f'<text x="{px(idx):.1f}" y="{H - 10}" text-anchor="{anchor}" font-size="{font}" fill="{SLATE}">{esc(label)}</text>')
    if y_unit:
        el.append(f'<text x="2" y="{font}" text-anchor="start" font-size="{font - 0.5}" fill="{GREY}">{esc(y_unit)}</text>')

    for hl in hlines:
        y = py(hl["y"])
        c = hl.get("color", ALARM)
        el.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="1.3" stroke-dasharray="5 5"/>')
        if hl.get("label"):
            el.append(f'<text x="{L + 8}" y="{y - 6:.1f}" text-anchor="start" font-size="{font - 0.5}" fill="{c}">{esc(hl["label"])}</text>')

    for k, s in enumerate(series):
        v = [float(x) for x in s["values"]]
        xs = [px(i) for i in range(len(v))]
        ys = [py(x) for x in v]
        line = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        if k == 0 and fill and not s.get("dash"):
            el.append(f'<path d="{line} L{xs[-1]:.1f},{H - B} L{xs[0]:.1f},{H - B} Z" fill="url(#{uid}-g)"/>')
        dash = ' stroke-dasharray="6 5"' if s.get("dash") else ""
        el.append(f'<path d="{line}" fill="none" stroke="{s.get("color", TEAL)}" stroke-width="{s.get("w", 1.8)}" '
                  f'stroke-linejoin="round" stroke-linecap="round"{dash}/>')

    for d in dots:
        sv = series[d.get("series", 0)]["values"]
        for i in d["idx"]:
            if 0 <= i < len(sv):
                el.append(f'<circle cx="{px(i):.1f}" cy="{py(sv[i]):.1f}" r="{d.get("r", 4)}" fill="{d["color"]}" '
                          f'stroke="#fff" stroke-width="1.5"/>')

    # invisible hover targets with native browser tooltips (first series only)
    sv = series[0]["values"]
    stride = max(1, len(sv) // 160)
    for i in range(0, len(sv), stride):
        name = x_names[i] if x_names and i < len(x_names) else f"point {i}"
        el.append(f'<circle class="ng-hit" cx="{px(i):.1f}" cy="{py(sv[i]):.1f}" r="7" fill="transparent">'
                  f'<title>{esc(name)}: {float(sv[i]):,.2f} {esc(y_unit)}</title></circle>')

    legend = []
    if len(series) > 1 or dots:
        for s in series:
            if s.get("name"):
                swatch = "ng-sw dash" if s.get("dash") else "ng-sw"
                legend.append(f'<span><i class="{swatch}" style="--c:{s.get("color", TEAL)}"></i>{esc(s["name"])}</span>')
        for d in dots:
            if d.get("label"):
                legend.append(f'<span><i class="ng-sw dot" style="--c:{d["color"]}"></i>{esc(d["label"])}</span>')
        for hl in hlines:
            if hl.get("label"):
                legend.append(f'<span><i class="ng-sw dash" style="--c:{hl.get("color", ALARM)}"></i>{esc(hl["label"])}</span>')
    legend_html = f'<div class="ng-legend">{"".join(legend)}</div>' if legend else ""
    return compact(f'<div class="ng-chart"><svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(aria)}" '
                   f'style="width:100%;height:auto;display:block">{"".join(el)}</svg>{legend_html}</div>')


def _decimate(values, n: int, how: str = "stride") -> np.ndarray:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) <= n:
        return v
    if how == "max":
        edges = np.linspace(0, len(v), n + 1).astype(int)
        return np.array([v[a:b].max() if b > a else v[min(a, len(v) - 1)] for a, b in zip(edges[:-1], edges[1:])])
    return v[np.linspace(0, len(v) - 1, n).round().astype(int)]


def sparkline(values, color=TEAL, uid="s", how="stride") -> str:
    v = _decimate(values, 120, how)
    if len(v) < 2:
        return ""
    W, H = 300, 56
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        hi = lo + 1.0
    xs = np.linspace(2, W - 2, len(v))
    ys = 6 + (H - 12) * (1 - (v - lo) / (hi - lo))
    line = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return compact(
        f'<svg class="ng-spark" viewBox="0 0 {W} {H}" preserveAspectRatio="none" aria-hidden="true">'
        f'<defs><linearGradient id="{uid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{color}" stop-opacity="0.30"/>'
        f'<stop offset="1" stop-color="{color}" stop-opacity="0.06"/></linearGradient></defs>'
        f'<path d="{line} L{xs[-1]:.1f},{H} L{xs[0]:.1f},{H} Z" fill="url(#{uid})"/>'
        f'<path d="{line}" fill="none" stroke="{color}" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>'
    )


# ── page furniture ────────────────────────────────────────────────────────────
def brand() -> str:
    return compact(
        '<div class="ng-brand"><div class="ng-mark"><svg viewBox="0 0 24 24" aria-hidden="true">'
        '<ellipse cx="12" cy="12" rx="9.5" ry="4" transform="rotate(-28 12 12)" fill="none" stroke="#fff" stroke-width="1.6"/>'
        '<circle cx="12" cy="12" r="2.6" fill="#fff"/></svg></div>'
        '<div><div class="ng-brand-name">naviguard</div><div class="ng-brand-sub">NavIC clock intelligence</div></div></div>'
    )


def side_label(text: str) -> str:
    return f'<div class="ng-side-label">{esc(text)}</div>'


def nav_header(text: str) -> str:
    """Heading above the view tabs on the main page (one icon, so the tabs themselves stay plain)."""
    icon = ('<svg viewBox="0 0 18 18" fill="none" stroke="#0E1B2A" stroke-width="1.5" stroke-linejoin="round" aria-hidden="true">'
            '<rect x="2" y="2" width="5.5" height="5.5" rx="1.2"/><rect x="10.5" y="2" width="5.5" height="5.5" rx="1.2"/>'
            '<rect x="2" y="10.5" width="5.5" height="5.5" rx="1.2"/><rect x="10.5" y="10.5" width="5.5" height="5.5" rx="1.2"/></svg>')
    return compact(f'<div class="ng-nav-head">{icon}<span>{esc(text)}</span></div>')


def sat_note(text: str) -> str:
    return f'<div class="ng-sat-note">{esc(text)}</div>'


def model_block(rows, notes, target_ns: float = 50.0) -> str:
    body = "".join(f'<div class="ng-model-row"><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in rows)
    note = "".join(f'<p>{esc(n)}</p>' for n in notes)
    return compact(f'<div class="ng-model">{body}{note}<div class="ng-target">Target: step-1 MAE \u2264 {target_ns:.0f} ns</div></div>')


def topbar(crumbs, status: str, ok: bool = True) -> str:
    parts = []
    for i, c in enumerate(crumbs):
        last = i == len(crumbs) - 1
        parts.append(f'<span class="{"cur" if last else ""}">{esc(c)}</span>')
        if not last:
            parts.append('<span class="sep">/</span>')
    return compact(f'<div class="ng-topbar"><div class="ng-crumbs">{"".join(parts)}</div>'
                   f'<div class="ng-status {"ok" if ok else "bad"}"><i></i>{esc(status)}</div></div>')


def hero(eyebrow: str, headline: str, sub_html: str) -> str:
    """sub_html is trusted HTML: build it with esc() and num()."""
    return compact(f'<div class="ng-hero"><div class="ng-eyebrow">{esc(eyebrow)}</div>'
                   f'<h1>{esc(headline)}</h1><p>{sub_html}</p></div>')


def pill(text: str, tone: str = "neutral") -> str:
    return f'<span class="ng-pill {tone}">{esc(text)}</span>'


def kpi_strip(cells, uid: str = "k") -> str:
    out = []
    for i, c in enumerate(cells):
        spark = sparkline(c.get("spark") or [], c.get("color", TEAL), uid=f"{uid}{i}", how=c.get("how", "stride"))
        status = ""
        if c.get("status"):
            status = f'<div class="ng-kpi-status {c.get("tone", "ok")}"><i></i>{esc(c["status"])}</div>'
        out.append(
            f'<div class="ng-kpi"><div class="ng-label">{esc(c["label"])}</div>'
            f'<div class="ng-kpi-value">{esc(c["value"])}<small>{esc(c.get("unit", ""))}</small></div>{status}'
            f'<div class="ng-kpi-spark">{spark}</div><div class="ng-kpi-foot">{esc(c.get("foot", ""))}</div></div>'
        )
    return compact(f'<div class="ng-kpis">{"".join(out)}</div>')


def stats(items) -> str:
    """items: [(label, value, tone)] where tone is '', 'ok', 'warn' or 'bad'."""
    cells = "".join(f'<div><span class="ng-label">{esc(l)}</span><b class="{t}">{esc(v)}</b></div>' for l, v, t in items)
    return compact(f'<div class="ng-stats">{cells}</div>')


def panel(title: str, sub: str = "", body: str = "", aside: str = "") -> str:
    sub_html = f"<p>{esc(sub)}</p>" if sub else ""
    return compact(f'<section class="ng-panel"><header><div><h2>{esc(title)}</h2>{sub_html}</div>{aside}</header>{body}</section>')


def notice(kind: str, title: str, body_html: str = "") -> str:
    """kind: info | warn | error. body_html is trusted HTML."""
    return compact(f'<div class="ng-notice {kind}" role="status"><b>{esc(title)}</b><p>{body_html}</p></div>')


def footer(detail: str) -> str:
    return f'<div class="ng-footer">{esc(detail)}</div>'


# ── tables ────────────────────────────────────────────────────────────────────
def baseline_table(rows, last_step: int) -> str:
    """rows: [{"name", "m1", "m_last", "served"}] with MAE in ns."""
    top = max(r["m1"] for r in rows) or 1.0
    best = min(r["m1"] for r in rows)
    body = []
    for r in sorted(rows, key=lambda r: r["m1"]):
        w = max(3.0, 100.0 * r["m1"] / top)
        tag = '<span class="ng-tag">served</span>' if r["served"] else ""
        bcls = " best" if abs(r["m1"] - best) < 1e-12 else ""
        body.append(f'<tr class="{"served" if r["served"] else ""}"><td>{esc(r["name"])}{tag}</td>'
                    f'<td class="bar"><i style="width:{w:.0f}%"></i></td>'
                    f'<td class="n{bcls}">{r["m1"]:.2f}</td><td class="n">{r["m_last"]:.2f}</td></tr>')
    head = f'<tr><th>model</th><th></th><th>step 1 (ns)</th><th>step {last_step} (ns)</th></tr>'
    return compact(f'<div class="ng-scroll"><table class="ng-table"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>')


def sat_table(rows) -> str:
    def f(v, d=2):
        return "\u2014" if v is None else (f"{v:,}" if isinstance(v, int) else f"{v:.{d}f}")
    head = ("<tr><th>satellite</th><th>samples</th><th>MAE step 1 (ns)</th><th>MAE last step (ns)</th>"
            "<th>RMSE step 1 (ns)</th><th>skill vs persistence</th></tr>")
    body = "".join(
        f'<tr><td><b>{esc(r["satellite"])}</b></td><td class="n">{f(r["samples"])}</td><td class="n">{f(r["mae1"])}</td>'
        f'<td class="n">{f(r["mae_last"])}</td><td class="n">{f(r["rmse1"])}</td><td class="n">{f(r["skill"], 3)}</td></tr>'
        for r in rows
    )
    return compact(f'<div class="ng-scroll"><table class="ng-table plain"><thead>{head}</thead><tbody>{body}</tbody></table></div>')


def data_table(df: pd.DataFrame, fmt: dict) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            try:
                txt = fmt[c].format(v) if c in fmt else (f"{v:,.0f}" if float(v).is_integer() else f"{v:.4g}")
            except (ValueError, TypeError):
                txt = str(v)
            cells.append(f'<td class="n">{esc(txt)}</td>')
        body.append(f'<tr>{"".join(cells)}</tr>')
    return compact(f'<div class="ng-scroll"><table class="ng-table plain mono"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def brief(text: str, second_opinion: dict | None = None) -> str:
    """Operator report: one row per 'Label: text' line, all text escaped."""
    rows = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        label, sep, rest = raw.partition(":")
        if sep and 0 < len(label.strip()) <= 20:
            rows.append((label.strip(), rest.strip()))
        else:
            rows.append(("", raw))
    if second_opinion:
        rows.append(("Second opinion", f'{second_opinion.get("severity", "unknown")}. {second_opinion.get("reasoning", "")}'.strip()))
    items = "".join(f'<div class="ng-brief-row"><dt>{esc(l)}</dt><dd>{esc(t)}</dd></div>' for l, t in rows)
    return f'<dl class="ng-brief">{compact(items)}</dl>'