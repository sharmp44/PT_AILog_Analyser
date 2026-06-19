"""
Report Generator
----------------
Renders the Jinja2 HTML template with RCA results and
optionally exports to PDF via WeasyPrint.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from Utils.logger import get_logger

log = get_logger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_FILE = "report.html.j2"


def generate(
    rca_result: dict,
    output_dir: str | Path,
    run_id: str = "unknown",
    cfg: dict | None = None,
    export_pdf: bool = True,
) -> dict[str, Path]:
    """
    Generate HTML (and optionally PDF) report.

    Returns:
        {"html": Path, "pdf": Path | None}
    """
    cfg = cfg or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    context = {
        "company_name":      cfg.get("report", {}).get("company_name", "Performance Engineering"),
        "generated_at":      generated_at,
        "run_id":            run_id,
        "verdict":           rca_result.get("verdict", "UNKNOWN"),
        "summary":           rca_result.get("summary", ""),
        "leading_indicator": rca_result.get("leading_indicator"),
        "cause_effect_chain":rca_result.get("cause_effect_chain", []),
        "scored_hypotheses": rca_result.get("scored_hypotheses", []),
        "finding_counts":    rca_result.get("finding_counts", {}),
        "raw_findings":      rca_result.get("raw_findings", {}),
        "timeline":          rca_result.get("timeline", {}),
    }

    # ── Render HTML ──────────────────────────────────────────────────────────
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=True,
    )
    # Add min filter for Jinja2
    env.filters["min"] = lambda a, b: min(a, b)

    template = env.get_template(_TEMPLATE_FILE)
    html_content = template.render(**context)

    html_path = output_dir / f"pt_report_{run_id}.html"
    html_path.write_text(html_content, encoding="utf-8")
    log.info(f"[report] HTML written: {html_path}")

    # ── Export PDF ───────────────────────────────────────────────────────────
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


def _build_pdf_html(ctx: dict) -> str:
    """Build a simple, xhtml2pdf-compatible PDF report (no CSS variables, no flexbox)."""
    verdict = ctx.get("verdict", "UNKNOWN")
    verdict_color = "#cc3333" if verdict == "FAIL" else "#2e7d32"
    counts = ctx.get("finding_counts", {})
    summary = ctx.get("summary", "No summary available.")
    li = ctx.get("leading_indicator") or {}
    chain = ctx.get("cause_effect_chain", [])
    hypotheses = ctx.get("scored_hypotheses", [])
    raw = ctx.get("raw_findings", {})
    timeline = ctx.get("timeline", {})
    events = (timeline.get("events") or [])[:30]

    def _rows(items, cols):
        rows = ""
        for item in items:
            rows += "<tr>" + "".join(f"<td>{item.get(c, '')}</td>" for c in cols) + "</tr>"
        return rows

    anomalies = raw.get("anomaly", [])[:20]
    thresholds = raw.get("threshold", [])
    trends = raw.get("trend", [])
    patterns = raw.get("pattern", [])

    chain_html = "".join(f"<li>{step}</li>" for step in chain) if chain else "<li>N/A</li>"
    hyp_rows = ""
    for h in hypotheses:
        hyp_rows += f"<tr><td>{h.get('hypothesis','')}</td><td>{h.get('confidence','')}</td><td>{h.get('sources_count','')}</td></tr>"

    tl_rows = ""
    for e in events:
        tl_rows += f"<tr><td>{e.get('timestamp','')}</td><td>{e.get('metric_name','')}</td><td>{e.get('severity','')}</td><td>{e.get('value','')}</td></tr>"

    anomaly_rows = ""
    for a in anomalies:
        anomaly_rows += f"<tr><td>{a.get('metric_name','')}</td><td>{a.get('method','')}</td><td>{a.get('severity','')}</td><td>{str(a.get('value',''))[:40]}</td></tr>"

    threshold_rows = ""
    for t in thresholds:
        threshold_rows += f"<tr><td>{t.get('metric_name','')}</td><td>{t.get('actual_value','')}</td><td>{t.get('threshold_value','')}</td><td>{t.get('severity','')}</td></tr>"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  body {{ font-family: Helvetica, Arial, sans-serif; font-size: 10pt; color: #222; margin: 20px; }}
  h1 {{ font-size: 18pt; color: #1a237e; }}
  h2 {{ font-size: 13pt; color: #1a237e; border-bottom: 1px solid #aaa; padding-bottom: 3px; margin-top: 18px; }}
  .verdict {{ font-size: 22pt; font-weight: bold; color: {verdict_color}; }}
  .meta {{ color: #555; font-size: 9pt; }}
  .kpi-row td {{ padding: 6px 12px; text-align: center; border: 1px solid #ccc; background: #f5f5f5; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 9pt; }}
  th {{ background: #1a237e; color: #fff; padding: 4px 6px; text-align: left; }}
  td {{ padding: 3px 6px; border-bottom: 1px solid #ddd; }}
  tr:nth-child(even) td {{ background: #f9f9f9; }}
  ul {{ margin: 4px 0; padding-left: 18px; }}
  li {{ margin-bottom: 2px; }}
  .summary-box {{ background: #f0f4ff; border-left: 4px solid #1a237e; padding: 8px 12px; margin: 8px 0; }}
</style>
</head>
<body>
<h1>PT AI Log Mining — RCA Report</h1>
<p class="meta">Run ID: {ctx.get('run_id','?')} &nbsp;|&nbsp; Generated: {ctx.get('generated_at','?')} &nbsp;|&nbsp; {ctx.get('company_name','')}</p>

<p>Verdict: <span class="verdict">{verdict}</span></p>

<h2>Finding Counts</h2>
<table class="kpi-row">
  <tr>
    <th>Anomalies</th><th>Trends</th><th>Threshold Breaches</th><th>Patterns</th><th>Total</th>
  </tr>
  <tr>
    <td>{counts.get('anomaly', 0)}</td>
    <td>{counts.get('trend', 0)}</td>
    <td>{counts.get('threshold', 0)}</td>
    <td>{counts.get('pattern', 0)}</td>
    <td>{counts.get('total', 0)}</td>
  </tr>
</table>

<h2>Executive Summary</h2>
<div class="summary-box">{summary}</div>

<h2>Leading Indicator</h2>
<p><b>{li.get('metric_name', 'N/A')}</b> — {li.get('severity', '')} at {li.get('timestamp', '')}<br/>
{li.get('description', '')}</p>

<h2>Cause-Effect Chain</h2>
<ol>{chain_html}</ol>

<h2>Scored Hypotheses</h2>
<table>
  <tr><th>Hypothesis</th><th>Confidence</th><th>Supporting Sources</th></tr>
  {hyp_rows if hyp_rows else '<tr><td colspan="3">No hypotheses (OpenAI key not set)</td></tr>'}
</table>

<h2>Threshold Breaches</h2>
<table>
  <tr><th>Metric</th><th>Actual</th><th>Threshold</th><th>Severity</th></tr>
  {threshold_rows if threshold_rows else '<tr><td colspan="4">None</td></tr>'}
</table>

<h2>Anomalies (top 20)</h2>
<table>
  <tr><th>Metric</th><th>Method</th><th>Severity</th><th>Value</th></tr>
  {anomaly_rows if anomaly_rows else '<tr><td colspan="4">None</td></tr>'}
</table>

<h2>Timeline (top 30 events)</h2>
<table>
  <tr><th>Timestamp</th><th>Metric</th><th>Severity</th><th>Value</th></tr>
  {tl_rows if tl_rows else '<tr><td colspan="4">No events</td></tr>'}
</table>

</body>
</html>"""
