"""
Report Generator
----------------
Renders a structured PT report with:
  - Test summary KPIs
  - LR transaction results table
  - Per-server utilization (App servers with IIS + PerfMon, DB server with SQL metrics)
  - Anomaly / trend findings table
  - Correlation timeline
  - RCA section (executive summary + ranked hypotheses)
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from Utils.logger import get_logger

log = get_logger(__name__)

_TEMPLATE_DIR  = Path(__file__).parent / "templates"
_TEMPLATE_FILE = "report.html.j2"


# ── Server stats builder ─────────────────────────────────────────────────────

def _metric_stat(df: pd.DataFrame, patterns: list[str], stat: str = "max") -> float | None:
    """Return a single stat (max/mean/last) for metric_names matching any pattern."""
    mask = pd.Series([False] * len(df), index=df.index)
    for p in patterns:
        mask |= df["metric_name"].str.contains(p, case=False, na=False)
    sub = df.loc[mask, "value"].dropna()
    if sub.empty:
        return None
    return round(float(getattr(sub, stat)()), 2)


def _sev(val, warn_thresh, crit_thresh, higher_is_worse=True) -> str:
    if val is None:
        return "ok"
    if higher_is_worse:
        if val >= crit_thresh:
            return "crit"
        if val >= warn_thresh:
            return "warn"
    else:
        if val <= crit_thresh:
            return "crit"
        if val <= warn_thresh:
            return "warn"
    return "ok"


def _build_server_stats(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """
    Build a list of server stat dicts from the normalised dataframe.
    Groups by _server column (set in app.py from filename) and source type.
    """
    if df is None or df.empty:
        return []

    sla = cfg.get("sla", {})
    sql_sla = cfg.get("sql_sla", {})

    # Ensure _server column exists
    if "_server" not in df.columns:
        df = df.copy()
        df["_server"] = df["source"].str.upper()

    servers = []
    grouped = df.groupby(["_server", "source"], sort=False)

    for (server_name, source), grp in grouped:
        if source == "loadrunner":
            continue   # LR handled separately in transaction table

        sev_counts = {"crit": 0, "warn": 0}
        metrics = []
        bars = []

        if source == "iis":
            total_req = len(grp)
            err5xx = int(grp[grp["metric_name"].str.contains("5\d\d|sc-status.*5|error", case=False, na=False)]["value"].count())
            err4xx = int(grp[grp["metric_name"].str.contains("4\d\d|sc-status.*4", case=False, na=False)]["value"].count())
            avg_ms  = _metric_stat(grp, ["time.taken", "time_taken", "timetaken", "response.*time"], "mean")
            rps     = round(total_req / max((grp["timestamp"].max() - grp["timestamp"].min()).total_seconds(), 1), 1) if total_req > 0 else 0

            ok_pct  = round((total_req - err5xx - err4xx) / max(total_req, 1) * 100, 1)
            err5_pct = round(err5xx / max(total_req, 1) * 100, 2)
            err4_pct = round(err4xx / max(total_req, 1) * 100, 2)

            if err5xx > 0:   sev_counts["crit"] += 1
            if err4xx > 100: sev_counts["warn"] += 1

            metrics = [
                {"label": "Total requests", "value": f"{total_req:,}", "sev": "ok"},
                {"label": "HTTP 500s",       "value": str(err5xx),      "sev": "crit" if err5xx > 0 else "ok"},
                {"label": "HTTP 4xxs",       "value": str(err4xx),      "sev": "warn" if err4xx > 100 else "ok"},
                {"label": "Req/sec",         "value": str(rps),         "sev": "ok"},
            ]
            bars = [
                {"label": "HTTP 200 (ok)",  "pct": min(ok_pct, 100),   "val": f"{ok_pct}%",   "sev": "ok"},
                {"label": "HTTP 4xx",       "pct": min(err4_pct*10,100),"val": f"{err4_pct}%", "sev": "warn" if err4xx > 0 else "ok"},
                {"label": "HTTP 5xx",       "pct": min(err5_pct*10,100),"val": str(err5xx),    "sev": "crit" if err5xx > 0 else "ok"},
            ]
            findings = []
            if err5xx > 0:
                findings.append({"sev": "crit", "text": f"{err5xx:,} HTTP 500 errors detected"})
            if err4xx > 100:
                findings.append({"sev": "warn", "text": f"{err4xx:,} HTTP 4xx client errors"})
            if not findings:
                findings.append({"sev": "ok", "text": "All IIS metrics within normal range"})

            servers.append({
                "name":   server_name,
                "source": "IIS",
                "tags":   ["IIS"],
                "sev_counts": sev_counts,
                "sections": [{"title": "IIS — web traffic", "metrics": metrics, "bars": bars}],
                "findings": findings,
            })

        elif source in ("perfmon", "blg"):
            cpu  = _metric_stat(grp, ["processor.*%", "% processor", "cpu"], "max")
            mem  = _metric_stat(grp, ["memory.*%", "% committed", "available mbytes"], "max")
            dq   = _metric_stat(grp, ["disk queue", "current disk queue"], "max")
            net  = _metric_stat(grp, ["bytes total/sec", "network.*bytes", "bytes.*sec"], "max")

            cpu_warn = sla.get("cpu_pct", 85)
            mem_warn = sla.get("memory_pct", 90)

            cpu_sev = _sev(cpu, cpu_warn * 0.85, cpu_warn)
            mem_sev = _sev(mem, mem_warn * 0.85, mem_warn)
            dq_sev  = _sev(dq, 1.5, 2.0) if dq is not None else "ok"

            if cpu_sev  == "crit": sev_counts["crit"] += 1
            elif cpu_sev == "warn": sev_counts["warn"] += 1
            if mem_sev  == "crit": sev_counts["crit"] += 1
            elif mem_sev == "warn": sev_counts["warn"] += 1

            metrics = []
            if cpu is not None:
                metrics.append({"label": "CPU (peak)",  "value": f"{cpu}%",  "sev": cpu_sev})
            if mem is not None:
                metrics.append({"label": "Memory %",    "value": f"{mem}%",  "sev": mem_sev})
            if dq is not None:
                metrics.append({"label": "Disk queue",  "value": str(dq),    "sev": dq_sev})
            if net is not None:
                metrics.append({"label": "Network MB/s","value": str(round(net/1_000_000 if net > 1_000_000 else net, 1)), "sev": "ok"})

            bars = []
            if cpu is not None:
                bars.append({"label": "CPU %",      "pct": min(cpu, 100), "val": f"{cpu}%",  "sev": cpu_sev})
            if mem is not None:
                bars.append({"label": "Memory %",   "pct": min(mem, 100), "val": f"{mem}%",  "sev": mem_sev})
            if dq is not None:
                bars.append({"label": "Disk queue", "pct": min(dq*20, 100),"val": str(dq),   "sev": dq_sev})

            findings = []
            if cpu_sev == "crit":
                findings.append({"sev": "crit", "text": f"CPU peaked at {cpu}% — exceeded {cpu_warn}% threshold"})
            elif cpu_sev == "warn":
                findings.append({"sev": "warn", "text": f"CPU peaked at {cpu}% — approaching {cpu_warn}% threshold"})
            if mem_sev in ("warn", "crit"):
                findings.append({"sev": mem_sev, "text": f"Memory at {mem}% — near threshold"})
            if not findings:
                findings.append({"sev": "ok", "text": "All system metrics within thresholds"})

            servers.append({
                "name":   server_name,
                "source": "PerfMon",
                "tags":   ["PerfMon"],
                "sev_counts": sev_counts,
                "sections": [{"title": "PerfMon — system", "metrics": metrics, "bars": bars}],
                "findings": findings,
            })

        elif source == "sql":
            cpu     = _metric_stat(grp, ["processor.*%", "% processor", "cpu"], "max")
            mem_pct = _metric_stat(grp, ["memory.*%", "% committed"], "max")
            bch     = _metric_stat(grp, ["buffer cache hit", "buffer.*hit.*ratio"], "min")
            ple     = _metric_stat(grp, ["page life expectancy", "ple"], "min")
            dlocks  = _metric_stat(grp, ["deadlock"], "max")
            locks   = _metric_stat(grp, ["lock waits/sec", "lock.*wait"], "max")
            blocked = _metric_stat(grp, ["processes blocked", "blocked proc"], "max")
            recomp  = _metric_stat(grp, ["recompilations/sec", "sql.*recompil"], "max")

            cpu_thresh   = sla.get("cpu_pct", 85)
            bch_thresh   = sql_sla.get("min_buffer_cache_hit_pct", 95)
            ple_thresh   = sql_sla.get("min_page_life_expectancy_s", 300)
            blk_thresh   = sql_sla.get("max_processes_blocked", 5)

            cpu_sev  = _sev(cpu,     cpu_thresh * 0.85, cpu_thresh)
            bch_sev  = _sev(bch,     bch_thresh + 1,    bch_thresh, higher_is_worse=False)
            ple_sev  = _sev(ple,     ple_thresh + 60,   ple_thresh, higher_is_worse=False)
            blk_sev  = _sev(blocked, blk_thresh - 1,    blk_thresh)

            for s in [cpu_sev, bch_sev, ple_sev, blk_sev]:
                if s == "crit":  sev_counts["crit"] += 1
                elif s == "warn": sev_counts["warn"] += 1

            sys_metrics, sql_metrics = [], []
            sys_bars, sql_bars = [], []

            if cpu is not None:
                sys_metrics.append({"label": "CPU (peak)",   "value": f"{cpu}%",   "sev": cpu_sev})
                sys_bars.append({"label": "CPU %", "pct": min(cpu, 100), "val": f"{cpu}%", "sev": cpu_sev})
            if mem_pct is not None:
                sys_metrics.append({"label": "Memory %",    "value": f"{mem_pct}%","sev": _sev(mem_pct, 85, 90)})
                sys_bars.append({"label": "Memory %", "pct": min(mem_pct,100), "val": f"{mem_pct}%", "sev": _sev(mem_pct,85,90)})

            if bch is not None:
                sql_metrics.append({"label": "Buffer cache hit", "value": f"{bch}%",  "sev": bch_sev})
                sql_bars.append({"label": "Buffer cache hit", "pct": min(bch,100), "val": f"{bch}%", "sev": bch_sev})
            if ple is not None:
                sql_metrics.append({"label": "Page life exp.", "value": f"{ple}s",   "sev": ple_sev})
                sql_bars.append({"label": "Page life exp.", "pct": min(ple/10,100), "val": f"{ple}s", "sev": ple_sev})
            if blocked is not None:
                sql_metrics.append({"label": "Blocked procs",  "value": str(int(blocked)), "sev": blk_sev})
                sql_bars.append({"label": "Blocked procs", "pct": min(blocked*10,100), "val": str(int(blocked)), "sev": blk_sev})
            if dlocks is not None:
                sql_metrics.append({"label": "Deadlocks/s",   "value": str(dlocks),     "sev": "crit" if dlocks > 0.1 else "ok"})
            if locks is not None:
                sql_metrics.append({"label": "Lock waits/s",  "value": str(locks),      "sev": "warn" if locks > 5 else "ok"})
            if recomp is not None:
                sql_metrics.append({"label": "Recompiles/s",  "value": str(recomp),     "sev": "warn" if recomp > 10 else "ok"})

            findings = []
            if cpu_sev in ("crit", "warn"):
                findings.append({"sev": cpu_sev, "text": f"SQL CPU at {cpu}% — exceeded {cpu_thresh}% threshold"})
            if bch_sev in ("crit", "warn"):
                findings.append({"sev": bch_sev, "text": f"Buffer cache hit {bch}% — below {bch_thresh}% minimum"})
            if ple_sev in ("crit", "warn"):
                findings.append({"sev": ple_sev, "text": f"Page life expectancy {ple}s — near {ple_thresh}s floor"})
            if blk_sev in ("crit", "warn"):
                findings.append({"sev": blk_sev, "text": f"Blocked processes: {int(blocked)} — exceeded {blk_thresh} threshold"})
            if not findings:
                findings.append({"sev": "ok", "text": "All SQL Server metrics within thresholds"})

            sections = []
            if sys_metrics:
                sections.append({"title": "PerfMon — system", "metrics": sys_metrics, "bars": sys_bars})
            if sql_metrics:
                sections.append({"title": "SQL Server counters", "metrics": sql_metrics, "bars": sql_bars})

            servers.append({
                "name":   server_name,
                "source": "SQL",
                "tags":   ["PerfMon", "SQL"],
                "sev_counts": sev_counts,
                "sections": sections,
                "findings": findings,
            })

    # ── Merge IIS + PerfMon cards for same server ────────────────────────────
    merged: dict[str, dict] = {}
    for s in servers:
        key = s["name"]
        if key not in merged:
            merged[key] = s
        else:
            existing = merged[key]
            existing["sections"].extend(s["sections"])
            existing["findings"].extend(s["findings"])
            existing["tags"] = list(set(existing["tags"] + s["tags"]))
            for sev in ("crit", "warn"):
                existing["sev_counts"][sev] += s["sev_counts"][sev]
            # Promote source label
            if existing["source"] != s["source"]:
                existing["source"] = "IIS + PerfMon"

    return list(merged.values())


def _build_lr_stats(df: pd.DataFrame) -> list[dict]:
    """Build LR transaction rows from the normalised dataframe."""
    if df is None or df.empty:
        return []
    lr = df[df["source"] == "loadrunner"].copy()
    if lr.empty:
        return []

    rows = []
    for tx, grp in lr.groupby("transaction_name", sort=False):
        if not tx or tx == "unknown":
            continue
        def v(patterns):
            for p in patterns:
                m = grp[grp["metric_name"] == p]["value"]
                if not m.empty:
                    return round(float(m.iloc[0]), 3)
            return None

        avg   = v(["avg_response_sec"])
        mn    = v(["min_response_sec"])
        p90   = v(["p90_response_sec"])
        p95   = v(["p95_response_sec"])
        p99   = v(["p99_response_sec"])
        mx    = v(["max_response_sec"])
        tps   = v(["tps"])
        fails = v(["fail_count"])
        passed= v(["pass_count"])
        fr    = v(["fail_rate"])

        # SLA badge from severity column
        sev = str(grp["severity"].mode().iloc[0]) if not grp.empty else "info"

        rows.append({
            "transaction": tx,
            "avg":   avg,  "min": mn,  "p90": p90,
            "p95":   p95,  "p99": p99, "max": mx,
            "tps":   tps,  "passed": int(passed) if passed is not None else None,
            "failed": int(fails) if fails is not None else None,
            "fail_rate": round(fr * 100, 2) if fr is not None else None,
            "sev": sev,
        })
    return rows


# ── Public entry point ───────────────────────────────────────────────────────

def generate(
    rca_result: dict,
    output_dir: str | Path,
    run_id: str = "unknown",
    cfg: dict | None = None,
    export_pdf: bool = True,
    combined_df=None,
) -> dict[str, Path]:
    cfg = cfg or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    server_stats = _build_server_stats(combined_df, cfg)
    lr_rows      = _build_lr_stats(combined_df)

    # Test duration from combined_df
    duration_min = None
    if combined_df is not None and not combined_df.empty and "timestamp" in combined_df.columns:
        ts = combined_df["timestamp"].dropna()
        if not ts.empty:
            duration_min = round((ts.max() - ts.min()).total_seconds() / 60, 1)

    context = {
        "company_name":       cfg.get("report", {}).get("company_name", "Performance Engineering"),
        "generated_at":       generated_at,
        "run_id":             run_id,
        "verdict":            rca_result.get("verdict", "UNKNOWN"),
        "summary":            rca_result.get("summary", ""),
        "leading_indicator":  rca_result.get("leading_indicator"),
        "cause_effect_chain": rca_result.get("cause_effect_chain", []),
        "scored_hypotheses":  rca_result.get("scored_hypotheses", []),
        "finding_counts":     rca_result.get("finding_counts", {}),
        "raw_findings":       rca_result.get("raw_findings", {}),
        "timeline":           rca_result.get("timeline", {}),
        "server_stats":       server_stats,
        "lr_rows":            lr_rows,
        "duration_min":       duration_min,
        "lr_percentile":      cfg.get("sla", {}).get("lr_percentile", 90),
    }

    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=True)
    env.filters["min"] = min
    template = env.get_template(_TEMPLATE_FILE)
    html_content = template.render(**context)

    html_path = output_dir / f"pt_report_{run_id}.html"
    html_path.write_text(html_content, encoding="utf-8")
    log.info(f"[report] HTML written: {html_path}")

    pdf_path = None
    if export_pdf:
        pdf_path = output_dir / f"pt_report_{run_id}.pdf"
        try:
            from xhtml2pdf import pisa
            pdf_html = _build_pdf_html(context)
            with open(pdf_path, "wb") as pdf_fh:
                pisa.CreatePDF(pdf_html, dest=pdf_fh, encoding="utf-8")
            log.info(f"[report] PDF written: {pdf_path}")
        except ImportError:
            log.warning("[report] PDF skipped. Run: pip install xhtml2pdf")
            pdf_path = None
        except Exception as exc:
            log.error(f"[report] PDF export failed: {exc}")
            pdf_path = None

    return {"html": html_path, "pdf": pdf_path}


# ── PDF builder ──────────────────────────────────────────────────────────────

def _build_pdf_html(ctx: dict) -> str:
    verdict       = ctx.get("verdict", "UNKNOWN")
    verdict_color = "#b91c1c" if verdict == "FAIL" else "#166534"
    counts        = ctx.get("finding_counts", {})
    summary       = ctx.get("summary", "No summary available.")
    chain         = ctx.get("cause_effect_chain", [])
    hypotheses    = ctx.get("scored_hypotheses", [])
    raw           = ctx.get("raw_findings", {})
    timeline      = ctx.get("timeline", {})
    server_stats  = ctx.get("server_stats", [])
    lr_rows       = ctx.get("lr_rows", [])
    lr_pct        = ctx.get("lr_percentile", 90)
    events        = (timeline.get("events") or [])[:25]

    def _badge(sev):
        colors = {"crit": "#b91c1c", "warn": "#92400e", "ok": "#166534", "info": "#1e40af"}
        c = colors.get(sev, "#374151")
        return f'<span style="background:{c};color:#fff;padding:1px 7px;border-radius:3px;font-size:8pt">{sev.upper()}</span>'

    # LR table
    lr_header = f"<tr><th>Transaction</th><th>Avg(s)</th><th>Min</th><th>P{lr_pct}</th><th>Max</th><th>Passed</th><th>Failed</th><th>TPS</th><th>SLA</th></tr>"
    lr_body = ""
    for r in lr_rows:
        lr_body += f"<tr><td>{r['transaction']}</td><td>{r.get('avg','—')}</td><td>{r.get('min','—')}</td>"
        pct_val = r.get(f"p{lr_pct}") or r.get("p90") or r.get("p95") or "—"
        lr_body += f"<td>{pct_val}</td><td>{r.get('max','—')}</td><td>{r.get('passed','—')}</td>"
        lr_body += f"<td>{r.get('failed','—')}</td><td>{r.get('tps','—')}</td><td>{_badge(r.get('sev','info'))}</td></tr>"

    # Server blocks
    server_html = ""
    for s in server_stats:
        server_html += f"<h3>{s['name']} ({s['source']})</h3>"
        for section in s.get("sections", []):
            server_html += f"<p><em>{section['title']}</em></p><table>"
            for m in section.get("metrics", []):
                server_html += f"<tr><td>{m['label']}</td><td><strong>{m['value']}</strong></td></tr>"
            server_html += "</table>"
        for f in s.get("findings", []):
            server_html += f"<p>{'[!]' if f['sev']=='crit' else '[~]' if f['sev']=='warn' else '[ok]'} {f['text']}</p>"

    hyp_rows = ""
    for h in hypotheses:
        hyp_rows += f"<tr><td>{h.get('hypothesis','')}</td><td>{h.get('confidence','')}</td><td>{h.get('evidence','')}</td></tr>"

    chain_html = "".join(f"<li>{s}</li>" for s in chain) if chain else "<li>N/A</li>"
    tl_rows = ""
    for e in events:
        tl_rows += f"<tr><td>{str(e.get('timestamp',''))[:19]}</td><td>{e.get('source','')}</td><td>{e.get('metric_name','')}</td><td>{e.get('severity','')}</td></tr>"

    thresh_rows = ""
    for t in raw.get("threshold", []):
        thresh_rows += f"<tr><td>{t.get('metric_name','')}</td><td>{t.get('actual_value','')}</td><td>{t.get('threshold_value','')}</td><td>{_badge(t.get('severity','info'))}</td></tr>"

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<style>
  body{{font-family:Arial,sans-serif;font-size:10pt;color:#111;margin:24px;background:#fff}}
  h1{{font-size:17pt;color:#1e3a5f;margin-bottom:4px}}
  h2{{font-size:13pt;color:#1e3a5f;border-bottom:2px solid #1e3a5f;padding-bottom:3px;margin:18px 0 8px}}
  h3{{font-size:11pt;color:#374151;margin:12px 0 4px}}
  .meta{{color:#6b7280;font-size:9pt}}
  .verdict{{font-size:20pt;font-weight:bold;color:{verdict_color}}}
  .kpi-row td{{padding:6px 14px;text-align:center;border:1px solid #d1d5db;background:#f9fafb;font-size:10pt}}
  .kpi-row th{{padding:5px 14px;background:#1e3a5f;color:#fff;font-size:9pt}}
  table{{width:100%;border-collapse:collapse;margin:6px 0;font-size:9pt}}
  th{{background:#1e3a5f;color:#fff;padding:5px 8px;text-align:left}}
  td{{padding:4px 8px;border-bottom:1px solid #e5e7eb}}
  tr:nth-child(even) td{{background:#f9fafb}}
  .summary-box{{background:#eff6ff;border-left:4px solid #1e3a5f;padding:10px 14px;margin:8px 0;font-size:10pt;line-height:1.6}}
  ol{{margin:6px 0;padding-left:20px}}
  li{{margin-bottom:4px;line-height:1.5}}
</style>
</head><body>
<h1>Performance Test Report — AI Log Mining</h1>
<p class="meta">Run ID: {ctx.get('run_id','?')} &nbsp;|&nbsp; Generated: {ctx.get('generated_at','?')} &nbsp;|&nbsp; {ctx.get('company_name','')}</p>
<p style="margin-top:8px">Verdict: <span class="verdict">{verdict}</span></p>

<h2>Test Summary</h2>
<table class="kpi-row">
  <tr><th>Total findings</th><th>SLA breaches</th><th>Anomalies</th><th>Trends</th><th>Patterns</th></tr>
  <tr>
    <td>{counts.get('total',0)}</td><td>{counts.get('threshold',0)}</td>
    <td>{counts.get('anomaly',0)}</td><td>{counts.get('trend',0)}</td><td>{counts.get('pattern',0)}</td>
  </tr>
</table>

<h2>LoadRunner Transaction Results</h2>
<table><thead>{lr_header}</thead><tbody>
{lr_body if lr_body else '<tr><td colspan="9">No LR transaction data available</td></tr>'}
</tbody></table>

<h2>Server Resource Utilization</h2>
{server_html if server_html else '<p>No server utilization data available</p>'}

<h2>Executive Summary</h2>
<div class="summary-box">{summary}</div>

<h2>Cause → Effect Chain</h2>
<ol>{chain_html}</ol>

<h2>Root Cause Hypotheses</h2>
<table><tr><th>Hypothesis</th><th>Confidence</th><th>Evidence</th></tr>
{hyp_rows if hyp_rows else '<tr><td colspan="3">No hypotheses (OpenAI key not configured)</td></tr>'}
</table>

<h2>SLA &amp; Threshold Breaches</h2>
<table><tr><th>Metric</th><th>Actual</th><th>Threshold</th><th>Severity</th></tr>
{thresh_rows if thresh_rows else '<tr><td colspan="4">None</td></tr>'}
</table>

<h2>Event Timeline (top 25)</h2>
<table><tr><th>Timestamp</th><th>Source</th><th>Metric</th><th>Severity</th></tr>
{tl_rows if tl_rows else '<tr><td colspan="4">No events</td></tr>'}
</table>

</body></html>"""
