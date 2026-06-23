"""
PT AI Log Analyser — Streamlit UI
===================================
Run with:  streamlit run app.py
"""
from __future__ import annotations

import os
import sys
import json
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
print(load_dotenv())

# ── Project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from Utils.config import load_config
from Utils.logger import get_logger
from ingest import blg_parser, iis_parser, lr_parser, sql_parser
from ingest.file_watcher import detect_type
from parse.normaliser import normalise
from parse.context_tagger import tag
from parse.store import Store
from agents import anomaly_agent, trend_agent, threshold_agent, pattern_agent, causal_agent
from correlation import timeline as timeline_mod, rca_engine
from report import generator

log = get_logger("streamlit_app")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PT AI Log Analyser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #c9d1d9;
}
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebarContent"] { padding-top: 1rem; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    font-size: 1rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 700; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 8px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e !important;
    border-radius: 6px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1f6feb !important;
    color: white !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #161b22;
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stFileUploader"]:hover { border-color: #1f6feb; }

/* ── Status boxes ── */
[data-testid="stStatusWidget"] { background: #161b22; border-radius: 8px; }

/* ── Text ── */
h1, h2, h3, h4 { color: #f0f6fc !important; }
p, li, td, th { color: #c9d1d9; }
.stMarkdown a { color: #58a6ff; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── Input fields ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: #0d1117 !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px;
}

/* ── Divider ── */
hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)

# ── Parsers registry ──────────────────────────────────────────────────────────
_PARSERS = {
    "blg":  blg_parser.parse,
    "iis":  iis_parser.parse,
    "lr":   lr_parser.parse,
    "sql":  sql_parser.parse,
}


def _apply_timezone(df: pd.DataFrame, tz_str: str) -> pd.DataFrame:
    """
    Re-interpret timestamps as being in tz_str, then convert to UTC.
    Handles both naive and already-tz-aware (UTC) timestamps from parsers.
    """
    if not tz_str or tz_str in ("UTC", "GMT"):
        return df
    df = df.copy()
    ts = df["timestamp"]
    try:
        if ts.dt.tz is None:
            # Naive → localize to source tz → UTC
            df["timestamp"] = ts.dt.tz_localize(tz_str, ambiguous="infer",
                                                  nonexistent="shift_forward"
                                                  ).dt.tz_convert("UTC")
        else:
            # Already tz-aware (parser set UTC by default) → strip → re-localize → UTC
            df["timestamp"] = (
                ts.dt.tz_localize(None)
                  .dt.tz_localize(tz_str, ambiguous="infer",
                                   nonexistent="shift_forward")
                  .dt.tz_convert("UTC")
            )
    except Exception as e:
        log.warning(f"Timezone conversion failed ({tz_str}): {e} — keeping as-is")
    return df


def _parse_file(file_path: Path, kind: str, cfg: dict) -> pd.DataFrame | None:
    parser = _PARSERS.get(kind)
    if parser is None:
        return None
    try:
        return parser(file_path, cfg)
    except Exception as exc:
        st.warning(f"⚠️ Could not parse `{file_path.name}`: {exc}")
        return None


def run_pipeline_ui(file_paths: list[Path], cfg: dict, output_dir: Path,
                    time_window: dict | None = None,
                    tz_cfg: dict | None = None) -> dict | None:
    """Run the full pipeline with Streamlit progress indicators."""

    # ── Phase 1: Ingest ───────────────────────────────────────────────────────
    st.markdown("**① Ingesting & parsing log files…**")
    prog = st.progress(0)
    dfs = []
    file_info = []

    for i, fp in enumerate(file_paths):
        kind = detect_type(fp)
        if kind is None:
            st.warning(f"⚠️ Cannot detect type for `{fp.name}` — skipping")
            continue
        df = _parse_file(fp, kind, cfg)
        if df is not None and not df.empty:
            # Apply per-source timezone correction before normalising
            if tz_cfg and kind in tz_cfg:
                df = _apply_timezone(df, tz_cfg[kind])
            dfs.append(df)
            tz_label = (tz_cfg or {}).get(kind, "UTC")
            file_info.append({"File": fp.name, "Type": kind.upper(), "Rows": len(df),
                              "Metrics": df["metric_name"].nunique(),
                              "Timezone": tz_label})
        prog.progress((i + 1) / len(file_paths))

    if not dfs:
        st.error("❌ No data could be parsed from the uploaded files.")
        return None

    # Show parse summary
    st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)

    # ── Phase 2: Normalise / Tag / Store ──────────────────────────────────────
    st.markdown("**② Normalising, tagging & storing events…**")
    combined = pd.concat(dfs, ignore_index=True)
    combined = normalise(combined)
    combined = tag(combined, cfg)
    run_id = combined["run_id"].iloc[0] if "run_id" in combined.columns else "unknown"

    db_path = cfg.get("pipeline", {}).get("db_path", "/tmp/pt_pipeline_ui.duckdb")
    with Store(db_path) as store:
        store.insert(combined)

    # ── Time Window Filter ────────────────────────────────────────────────────
    if time_window and time_window.get("enabled"):
        tw_start = pd.Timestamp(time_window["start"], tz="UTC")
        tw_end   = pd.Timestamp(time_window["end"],   tz="UTC")
        before   = len(combined)
        combined = combined[
            (combined["timestamp"] >= tw_start) &
            (combined["timestamp"] <= tw_end)
        ].copy()
        after = len(combined)
        if combined.empty:
            st.error(
                f"❌ Time window filter removed all {before} events. "
                f"No data between {tw_start.strftime('%H:%M')} and {tw_end.strftime('%H:%M')}. "
                "Try widening the window or check your log timestamps."
            )
            return None
        st.info(
            f"⏱️ Time window applied: **{tw_start.strftime('%Y-%m-%d %H:%M')}** → "
            f"**{tw_end.strftime('%H:%M')}**  |  "
            f"{after:,} events kept out of {before:,} total"
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Events", f"{len(combined):,}")
    col2.metric("Unique Metrics", combined["metric_name"].nunique())
    col3.metric("Run ID", run_id[:8])

    # ── Phase 3: AI Agents ────────────────────────────────────────────────────
    st.markdown("**③ Running AI agents in parallel…**")
    agent_bar = st.progress(0)

    with ThreadPoolExecutor(max_workers=4) as ex:
        f_anomaly   = ex.submit(anomaly_agent.run,   combined, cfg)
        f_trend     = ex.submit(trend_agent.run,     combined, cfg)
        f_threshold = ex.submit(threshold_agent.run, combined, cfg)
        f_pattern   = ex.submit(pattern_agent.run,   combined, cfg)

    agent_bar.progress(50)
    anomaly_f   = f_anomaly.result()
    trend_f     = f_trend.result()
    threshold_f = f_threshold.result()
    pattern_f   = f_pattern.result()
    agent_bar.progress(100)

    # ── Phase 4: Correlation & RCA ────────────────────────────────────────────
    st.markdown("**④ Correlating findings & building RCA…**")
    causal_result = causal_agent.run(anomaly_f, trend_f, threshold_f, pattern_f, cfg)
    tl = timeline_mod.build(anomaly_f, trend_f, threshold_f, pattern_f)
    rca = rca_engine.build(
        causal_result, tl,
        anomaly_f, trend_f, threshold_f, pattern_f,
        output_dir=str(output_dir),
    )

    # ── Phase 5: Report ───────────────────────────────────────────────────────
    st.markdown("**⑤ Generating HTML & PDF report…**")
    paths = generator.generate(rca, output_dir, run_id=run_id, cfg=cfg, export_pdf=True)

    return {**rca, "report_paths": paths, "run_id": run_id}


def _severity_badge(sev: str) -> str:
    colors = {"critical": "#da3633", "warn": "#d29922", "info": "#388bfd"}
    c = colors.get(sev.lower(), "#8b949e")
    return f'<span style="background:{c};color:white;padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:600">{sev.upper()}</span>'


def show_results(result: dict) -> None:
    verdict  = result.get("verdict", "UNKNOWN")
    counts   = result.get("finding_counts", {})
    run_id   = result.get("run_id", "?")
    paths    = result.get("report_paths", {})

    # ── Verdict hero ──────────────────────────────────────────────────────────
    v_color  = "#da3633" if verdict == "FAIL" else "#3fb950"
    v_bg     = "#2d1b1b" if verdict == "FAIL" else "#1b2d1b"
    v_icon   = "❌" if verdict == "FAIL" else "✅"
    st.markdown(f"""
    <div style="background:{v_bg};border:2px solid {v_color};border-radius:16px;
                padding:2rem;text-align:center;margin:1rem 0 2rem 0;">
        <div style="font-size:4rem;margin-bottom:0.5rem">{v_icon}</div>
        <div style="font-size:3rem;font-weight:800;color:{v_color};letter-spacing:0.1em">{verdict}</div>
        <div style="color:#8b949e;margin-top:0.5rem">Run ID: <code>{run_id}</code></div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔴 Anomalies",   counts.get("anomaly", 0))
    c2.metric("📈 Trends",      counts.get("trend", 0))
    c3.metric("⚡ Thresholds",  counts.get("threshold", 0))
    c4.metric("🔍 Patterns",    counts.get("pattern", 0))
    c5.metric("📊 Total",       counts.get("total", 0))

    st.markdown("---")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = result.get("summary", "")
    if summary:
        st.markdown("### 📝 Executive Summary")
        st.info(summary)

    # ── Leading indicator + chain ─────────────────────────────────────────────
    li    = result.get("leading_indicator") or {}
    chain = result.get("cause_effect_chain", [])
    if li or chain:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown("### 🎯 Leading Indicator")
            if li:
                st.markdown(f"""
                <div style="background:#161b22;border-left:4px solid #d29922;
                            border-radius:8px;padding:1rem;">
                    <strong style="color:#f0f6fc">{li.get('metric_name','N/A')}</strong><br>
                    <span style="color:#8b949e">{li.get('timestamp','')}</span><br>
                    <span style="color:#c9d1d9">{li.get('description','')}</span>
                </div>
                """, unsafe_allow_html=True)
        with col_b:
            if chain:
                st.markdown("### 🔗 Cause-Effect Chain")
                for i, step in enumerate(chain, 1):
                    st.markdown(f"""
                    <div style="display:flex;align-items:flex-start;margin-bottom:0.5rem">
                        <div style="background:#1f6feb;color:white;border-radius:50%;
                                    width:24px;height:24px;min-width:24px;display:flex;
                                    align-items:center;justify-content:center;
                                    font-weight:700;font-size:0.8rem;margin-right:0.75rem;
                                    margin-top:2px">{i}</div>
                        <div style="color:#c9d1d9;padding-top:2px">{step}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Tabs: findings detail ─────────────────────────────────────────────────
    st.markdown("### 🔎 Findings Detail")
    raw = result.get("raw_findings", {})
    tab_threshold, tab_anomaly, tab_trend, tab_pattern, tab_timeline = st.tabs([
        "⚡ Threshold Breaches",
        "🔴 Anomalies",
        "📈 Trends",
        "🔍 Patterns",
        "⏱️ Timeline",
    ])

    with tab_threshold:
        data = raw.get("threshold", [])
        if data:
            df = pd.DataFrame(data)[["metric_name","rule_name","actual","threshold","severity","business_kpi","phase"]]
            df.columns = ["Metric","Rule","Actual","Threshold","Severity","Business KPI","Phase"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.success("No threshold breaches detected.")

    with tab_anomaly:
        data = raw.get("anomaly", [])
        if data:
            df = pd.DataFrame(data)[["metric_name","method","severity","value","timestamp","source"]]
            df.columns = ["Metric","Method","Severity","Value","Timestamp","Source"]
            st.dataframe(df.head(100), use_container_width=True, hide_index=True)
            if len(data) > 100:
                st.caption(f"Showing top 100 of {len(data)} anomalies.")
        else:
            st.success("No anomalies detected.")

    with tab_trend:
        data = raw.get("trend", [])
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.success("No degrading trends detected.")

    with tab_pattern:
        data = raw.get("pattern", [])
        if data:
            df = pd.DataFrame(data)[["pattern_name","severity","count","description","method","source"]]
            df.columns = ["Pattern","Severity","Count","Description","Method","Source"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.success("No log patterns detected.")

    with tab_timeline:
        tl = result.get("timeline", {})
        events = (tl.get("events") or [])[:50]
        if events:
            df = pd.DataFrame(events)
            cols = [c for c in ["timestamp","metric_name","severity","value","source","phase"] if c in df.columns]
            st.dataframe(df[cols].head(50), use_container_width=True, hide_index=True)
            st.caption(f"Showing top 50 of {len(tl.get('events',[]))} timeline events.")
        else:
            st.info("No timeline events available.")

    # ── Hypotheses ────────────────────────────────────────────────────────────
    hyps = result.get("scored_hypotheses", [])
    if hyps:
        st.markdown("### 🧠 Ranked Hypotheses")
        for h in hyps:
            conf   = h.get("confidence", "LOW")
            c_col  = {"HIGH": "#3fb950", "MEDIUM": "#d29922", "LOW": "#da3633"}.get(conf, "#8b949e")
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;
                        padding:1rem;margin-bottom:0.75rem;border-left:4px solid {c_col}">
                <span style="background:{c_col};color:white;padding:2px 8px;border-radius:12px;
                             font-size:0.75rem;font-weight:700">{conf}</span>
                <span style="color:#f0f6fc;font-weight:600;margin-left:0.75rem">
                    {h.get('hypothesis','')}</span><br>
                <span style="color:#8b949e;font-size:0.85rem;margin-top:0.25rem;display:block">
                    Sources: {h.get('sources_count', 0)} &nbsp;|&nbsp; {h.get('explanation','')}
                </span>
            </div>
            """, unsafe_allow_html=True)

    # ── Downloads + Online View ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Reports")

    html_content = st.session_state.get("_html_content", "")
    pdf_bytes    = st.session_state.get("_pdf_bytes", None)
    run_id_str   = result.get("run_id", "report")

    btn1, btn2, btn3, _ = st.columns([1, 1, 1, 2])

    if html_content:
        btn1.download_button(
            "⬇️ Download HTML",
            data=html_content.encode("utf-8"),
            file_name=f"pt_report_{run_id_str}.html",
            mime="text/html",
            use_container_width=True,
        )

    if pdf_bytes:
        btn2.download_button(
            "⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"pt_report_{run_id_str}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        btn2.info("PDF: `pip install xhtml2pdf`")

    if html_content:
        if "show_report_online" not in st.session_state:
            st.session_state["show_report_online"] = False
        label = "🙈 Hide Report" if st.session_state["show_report_online"] else "👁️ View Online"
        if btn3.button(label, use_container_width=True):
            st.session_state["show_report_online"] = not st.session_state["show_report_online"]

    if st.session_state.get("show_report_online") and html_content:
        st.markdown("#### 📄 Full Report Preview")
        import streamlit.components.v1 as components
        components.html(html_content, height=950, scrolling=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

import datetime as _dt

# Extra CSS for coloured expander headers
st.markdown("""
<style>
/* Coloured left-border per expander using nth-child targeting via class trick */
.sla-expander    details { border-left: 4px solid #388bfd !important; border-radius: 8px; }
.sql-expander    details { border-left: 4px solid #3fb950 !important; border-radius: 8px; }
.tz-expander     details { border-left: 4px solid #d29922 !important; border-radius: 8px; }
.tw-expander     details { border-left: 4px solid #ab47bc !important; border-radius: 8px; }

/* Expander summary text colour */
.sla-expander summary span { color: #388bfd !important; font-weight: 600; }
.sql-expander summary span { color: #3fb950 !important; font-weight: 600; }
.tz-expander  summary span { color: #d29922 !important; font-weight: 600; }
.tw-expander  summary span { color: #ab47bc !important; font-weight: 600; }

/* General expander polish */
[data-testid="stExpander"] details {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    margin-bottom: 0.5rem;
}
[data-testid="stExpander"] summary {
    padding: 0.6rem 0.8rem !important;
    font-size: 0.9rem !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    # ── Logo ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1rem 0;
                border-bottom:1px solid #21262d;margin-bottom:1rem;">
        <div style="font-size:2rem">🔬</div>
        <div style="font-size:1rem;font-weight:700;color:#f0f6fc">PT AI Log Analyser</div>
        <div style="font-size:0.7rem;color:#8b949e;margin-top:2px">AI-Powered RCA Engine</div>
    </div>
    """, unsafe_allow_html=True)

    # ── 1. SLA Thresholds (BLUE) ─────────────────────────────────────────────
    st.markdown("""<div style="background:#1a2a4a;border-left:4px solid #388bfd;
        border-radius:6px;padding:4px 10px;margin-bottom:4px;">
        <span style="color:#388bfd;font-weight:700;font-size:0.85rem">📊 SLA Thresholds</span>
    </div>""", unsafe_allow_html=True)

    with st.expander("Expand to configure SLA thresholds", expanded=False):
        p95_ms    = st.number_input("P95 Latency (ms)",   value=2000, step=100)
        error_pct = st.number_input("Max Error Rate (%)", value=1.0,  step=0.1, format="%.1f")
        cpu_pct   = st.number_input("CPU Alert (%)",       value=85,   step=5)
        mem_pct   = st.number_input("Memory Alert (%)",    value=90,   step=5)
        disk_q    = st.number_input("Disk Queue Length",   value=2.0,  step=0.5, format="%.1f")
        tps_drop  = st.number_input("TPS Drop Alert (%)",  value=20.0, step=5.0, format="%.1f")
    # defaults if not expanded
    if "p95_ms" not in dir():
        p95_ms = 2000; error_pct = 1.0; cpu_pct = 85
        mem_pct = 90; disk_q = 2.0; tps_drop = 20.0

    # ── 2. SQL Server SLA (GREEN) ─────────────────────────────────────────────
    st.markdown("""<div style="background:#1a2d1a;border-left:4px solid #3fb950;
        border-radius:6px;padding:4px 10px;margin-bottom:4px;margin-top:8px;">
        <span style="color:#3fb950;font-weight:700;font-size:0.85rem">🗄️ SQL Server SLA</span>
    </div>""", unsafe_allow_html=True)

    with st.expander("Expand to configure SQL Server thresholds", expanded=False):
        min_cache  = st.number_input("Min Buffer Cache Hit (%)",      value=95.0, step=1.0,  format="%.1f")
        min_ple    = st.number_input("Min Page Life Expectancy (s)",  value=300,  step=60)
        max_deadlk = st.number_input("Max Deadlocks/sec",             value=0.1,  step=0.1,  format="%.1f")
        max_locks  = st.number_input("Max Lock Waits/sec",            value=5.0,  step=1.0,  format="%.1f")
        max_block  = st.number_input("Max Blocked Processes",         value=5,    step=1)
        max_recomp = st.number_input("Max Re-compilations/sec",       value=10.0, step=1.0,  format="%.1f")

    # ── 3. Log Timezones (AMBER) ─────────────────────────────────────────────
    st.markdown("""<div style="background:#2d2510;border-left:4px solid #d29922;
        border-radius:6px;padding:4px 10px;margin-bottom:4px;margin-top:8px;">
        <span style="color:#d29922;font-weight:700;font-size:0.85rem">🌍 Log Timezones</span>
    </div>""", unsafe_allow_html=True)

    _TZ_OPTIONS = [
        "UTC / GMT",
        "Europe/London (BST/GMT auto)",
        "UTC+1 (fixed)",
        "UTC+2 (fixed)",
        "UTC+3 (fixed)",
        "UTC-5 (EST)",
        "UTC-8 (PST)",
    ]
    _TZ_MAP = {
        "UTC / GMT":                   "UTC",
        "Europe/London (BST/GMT auto)": "Europe/London",
        "UTC+1 (fixed)":               "Etc/GMT-1",
        "UTC+2 (fixed)":               "Etc/GMT-2",
        "UTC+3 (fixed)":               "Etc/GMT-3",
        "UTC-5 (EST)":                 "Etc/GMT+5",
        "UTC-8 (PST)":                 "Etc/GMT+8",
    }

    with st.expander("Expand to set per-source timezones", expanded=False):
        st.caption("All sources will be aligned to UTC before analysis.")
        tz_iis = st.selectbox("🌐 IIS Log Timezone",       _TZ_OPTIONS, index=0)
        tz_blg = st.selectbox("🪟 BLG / PerfMon Timezone", _TZ_OPTIONS, index=1)
        tz_lr  = st.selectbox("🏃 LoadRunner Timezone",     _TZ_OPTIONS, index=1)
        tz_sql = st.selectbox("🗄️ SQL Server Timezone",     _TZ_OPTIONS, index=1)

    tz_cfg = {
        "iis": _TZ_MAP.get(tz_iis if "tz_iis" in dir() else "UTC / GMT", "UTC"),
        "blg": _TZ_MAP.get(tz_blg if "tz_blg" in dir() else "Europe/London (BST/GMT auto)", "Europe/London"),
        "lr":  _TZ_MAP.get(tz_lr  if "tz_lr"  in dir() else "Europe/London (BST/GMT auto)", "Europe/London"),
        "sql": _TZ_MAP.get(tz_sql if "tz_sql" in dir() else "Europe/London (BST/GMT auto)", "Europe/London"),
    }

    # ── 4. Time Window (PURPLE) ──────────────────────────────────────────────
    st.markdown("""<div style="background:#1e1228;border-left:4px solid #ab47bc;
        border-radius:6px;padding:4px 10px;margin-bottom:4px;margin-top:8px;">
        <span style="color:#ab47bc;font-weight:700;font-size:0.85rem">⏱️ Test Time Window</span>
    </div>""", unsafe_allow_html=True)

    use_time_window = st.toggle("Enable Time Window Filter", value=False)

    if use_time_window:
        with st.expander("Set time window", expanded=True):
            st.caption("Filter logs to your exact test period only.")
            tw_date       = st.date_input("Test Date",  value=_dt.date.today())
            tw_start_time = st.time_input("Start Time", value=_dt.time(9, 0, 0))
            tw_end_time   = st.time_input("End Time",   value=_dt.time(18, 0, 0))
            if tw_start_time >= tw_end_time:
                st.warning("⚠️ End time must be after start time.")
                use_time_window = False
    else:
        tw_date       = _dt.date.today()
        tw_start_time = _dt.time(9, 0, 0)
        tw_end_time   = _dt.time(18, 0, 0)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:#8b949e;text-align:center;line-height:1.8">
        🪟 BLG/PerfMon &nbsp;·&nbsp; 🌐 IIS W3C<br>
        🏃 LoadRunner &nbsp;·&nbsp; 🗄️ SQL Server
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style="background:linear-gradient(135deg,#0d1117 0%,#1a2332 40%,#0d2137 100%);
            border:1px solid #21262d;border-radius:16px;padding:2rem 2.5rem;margin-bottom:2rem;">
    <h1 style="margin:0;font-size:2rem;color:#f0f6fc">
        🔬 PT AI Log Analyser
    </h1>
    <p style="margin:0.4rem 0 0 0;color:#8b949e;font-size:1rem">
        Upload performance test logs → AI analyses anomalies, trends &amp; SLA breaches →
        Automated Root Cause Analysis report
    </p>
    <div style="margin-top:1rem;display:flex;gap:1rem;flex-wrap:wrap">
        <span style="background:#1f3a5f;color:#58a6ff;padding:3px 10px;border-radius:20px;font-size:0.8rem">BLG / PerfMon</span>
        <span style="background:#1f3a5f;color:#58a6ff;padding:3px 10px;border-radius:20px;font-size:0.8rem">IIS W3C Logs</span>
        <span style="background:#1f3a5f;color:#58a6ff;padding:3px 10px;border-radius:20px;font-size:0.8rem">LoadRunner</span>
        <span style="background:#1f3a5f;color:#58a6ff;padding:3px 10px;border-radius:20px;font-size:0.8rem">SQL Server Metrics</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── File Upload ───────────────────────────────────────────────────────────────
st.markdown("### 📂 Upload Log Files")
st.markdown(
    "Drag & drop files from **anywhere** on your computer, or click **Browse files**. "
    "Mix file types freely — the pipeline auto-detects each one."
)

uploaded_files = st.file_uploader(
    "Drop files here",
    type=["log", "csv", "blg", "txt", "tsv"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    with st.expander(f"📁 {len(uploaded_files)} file(s) selected", expanded=True):
        cols = st.columns(4)
        cols[0].markdown("**File**")
        cols[1].markdown("**Size**")
        cols[2].markdown("**Detected Type**")
        cols[3].markdown("**Status**")

        for uf in uploaded_files:
            size_kb = len(uf.getvalue()) / 1024
            # Sniff type from name (before saving)
            hint_path = Path(uf.name)
            sniffed   = detect_type(hint_path) or "auto"
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"`{uf.name}`")
            c2.markdown(f"{size_kb:.1f} KB")
            c3.markdown(f"**{sniffed.upper()}**")
            c4.markdown("✅ Ready")

    st.markdown("---")

    run_col, _ = st.columns([2, 5])
    run_btn = run_col.button("▶ Run AI Analysis", type="primary", use_container_width=True)

    if run_btn:
        # Build cfg from sidebar values
        base_cfg = load_config()
        cfg = {
            **base_cfg,
            "pipeline": {
                **base_cfg.get("pipeline", {}),
                "db_path": "/tmp/pt_pipeline_ui.duckdb",
            },
            "openai": {
                **base_cfg.get("openai", {}),
                "api_key":  os.environ.get("OPENAI_API_KEY", "").strip(),
                "base_url": os.environ.get("OPENAI_BASE_URL", base_cfg.get("openai", {}).get("base_url", "")),
                "model":    os.environ.get("OPENAI_MODEL",    base_cfg.get("openai", {}).get("model", "gpt-4o")),
            },
            "sla": {
                "p95_latency_ms":  p95_ms,
                "error_rate_pct":  error_pct,
                "cpu_pct":         cpu_pct,
                "memory_pct":      mem_pct,
                "disk_queue_length": disk_q,
                "tps_drop_pct":    tps_drop,
            },
            "sql_sla": {
                "min_buffer_cache_hit_pct":    min_cache,
                "min_page_life_expectancy_s":  min_ple,
                "max_deadlocks_sec":           max_deadlk,
                "max_lock_waits_sec":          max_locks,
                "max_processes_blocked":       max_block,
                "max_recompilations_sec":      max_recomp,
            },
        }

        # Save uploaded files to a temp directory
        tmp_dir = Path(tempfile.mkdtemp(prefix="pt_analyser_"))
        output_dir = tmp_dir / "output"
        output_dir.mkdir()

        file_paths: list[Path] = []
        for uf in uploaded_files:
            dest = tmp_dir / uf.name
            dest.write_bytes(uf.getvalue())
            file_paths.append(dest)

        # Run pipeline with UI progress
        st.markdown("---")
        st.markdown("### ⚙️ Pipeline Running")

        # tz_cfg is already built in sidebar scope — available here
        # Build time window dict if enabled
        time_window = None
        if use_time_window:
            import datetime as _dt
            tw_start_dt = _dt.datetime.combine(tw_date, tw_start_time)
            tw_end_dt   = _dt.datetime.combine(tw_date, tw_end_time)
            time_window = {
                "enabled": True,
                "start":   tw_start_dt.isoformat(),
                "end":     tw_end_dt.isoformat(),
            }

        with st.status("🔄 Analysing logs…", expanded=True) as status:
            result = run_pipeline_ui(file_paths, cfg, output_dir,
                                     time_window=time_window,
                                     tz_cfg=tz_cfg)
            if result:
                # Persist result and report bytes in session state so they
                # survive button-click reruns (e.g. "View Online" toggle)
                st.session_state["last_result"] = result
                paths = result.get("report_paths", {})
                html_p = paths.get("html")
                pdf_p  = paths.get("pdf")
                st.session_state["_html_content"] = (
                    Path(str(html_p)).read_text(encoding="utf-8")
                    if html_p and Path(str(html_p)).exists() else ""
                )
                st.session_state["_pdf_bytes"] = (
                    Path(str(pdf_p)).read_bytes()
                    if pdf_p and Path(str(pdf_p)).exists() else None
                )
                st.session_state["show_report_online"] = False  # reset on new run
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)

# Always render results from session state (survives reruns from button clicks)
if "last_result" in st.session_state:
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    show_results(st.session_state["last_result"])

else:
    # Empty state
    st.markdown("""
    <div style="background:#161b22;border:2px dashed #30363d;border-radius:16px;
                padding:3rem;text-align:center;margin-top:1rem">
        <div style="font-size:3rem;margin-bottom:1rem">📁</div>
        <div style="font-size:1.1rem;color:#f0f6fc;font-weight:600">No files uploaded yet</div>
        <div style="color:#8b949e;margin-top:0.5rem">
            Click the upload area above and select your log files.<br>
            You can pick files from any location on your computer.
        </div>
        <div style="margin-top:1.5rem;color:#8b949e;font-size:0.85rem">
            Supported: <code>.log</code> &nbsp; <code>.csv</code> &nbsp;
            <code>.blg</code> &nbsp; <code>.txt</code> &nbsp; <code>.tsv</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 💡 Quick Start")
    st.code("""# 1. Generate sample data first
python sample_data/generate_samples.py

# 2. Then upload the files from:
#    logs_drop/iis_u_ex240619.log
#    logs_drop/perfmon_server01.csv
#    logs_drop/lr_results.csv
#    logs_drop/vuser_0.log
#    logs_drop/sql_server01.csv  ← new SQL metrics file
""", language="bash")
