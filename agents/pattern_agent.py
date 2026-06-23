"""
Pattern Agent
-------------
Two-pass analysis of raw log text:

Pass 1 – Regex:  Fast, no API cost.
  - OOM events          (OutOfMemory, MemoryError)
  - Stack traces        (at *.* / Exception in thread)
  - Timeout clusters    (connection timed out, socket timeout)
  - 5xx spike bursts
  - Connection refused / reset
  - GC pauses           (GC overhead limit exceeded)
  - SLA breach tags     (from LR logs)

Pass 2 – OpenAI GPT-4o:  Semantic understanding on the top-N suspicious lines.
  - Groups related errors into named patterns
  - Extracts root cause hypotheses from stack traces
  - Returns structured JSON

Returns a list of PatternFinding dicts.
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, asdict

import pandas as pd

from Utils.logger import get_logger

log = get_logger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────
_PATTERNS: list[tuple[str, str, str, str]] = [
    # (name, regex, severity, description)
    ("oom",         r"(OutOfMemory|out of memory|MemoryError|java\.lang\.OutOfMemoryError)",
                    "critical", "Out-of-memory event"),
    ("stack_trace", r"(Exception in thread|Unhandled exception|at \w+\.\w+\(|Traceback \(most recent)",
                    "critical", "Stack trace / unhandled exception"),
    ("timeout",     r"(timed? ?out|connection timeout|socket timeout|read timeout|operation timed out)",
                    "warn",     "Timeout event"),
    ("conn_refused",r"(connection refused|connection reset|ECONNREFUSED|ECONNRESET)",
                    "warn",     "Connection refused/reset"),
    ("gc_pressure", r"(GC overhead limit exceeded|Full GC|Stop.the.world|System\.gc\(\))",
                    "warn",     "Garbage collection pressure"),
    ("5xx_spike",   r"HTTP[/ ]\S+\s+5\d{2}|status[=: ]+5\d{2}",
                    "critical", "HTTP 5xx error"),
    ("sla_breach",  r"(SLA breach|SLA failed|transaction failed|FAIL\s*$)",
                    "critical", "SLA / transaction failure"),
    ("disk_full",   r"(no space left|disk full|ENOSPC|disk quota exceeded)",
                    "critical", "Disk full / no space"),
    ("deadlock",    r"(deadlock|Deadlock|dead lock|locking conflict)",
                    "critical", "Deadlock detected"),
    ("thread_pool", r"(thread pool exhausted|no available thread|all worker threads busy)",
                    "warn",     "Thread pool saturation"),
]

_MAX_LLM_LINES = 80   # send at most this many suspicious lines to OpenAI


@dataclass
class PatternFinding:
    pattern_name: str
    source:       str
    severity:     str
    count:        int
    first_ts:     str
    last_ts:      str
    examples:     list[str]
    description:  str
    method:       str   # "regex" | "llm"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Pass 1: Regex ─────────────────────────────────────────────────────────────

def _regex_pass(df: pd.DataFrame) -> tuple[list[PatternFinding], list[str]]:
    """Returns (findings, suspicious_lines_for_llm)."""
    findings: list[PatternFinding] = []
    suspicious: list[str] = []

    for name, pattern, severity, desc in _PATTERNS:
        rx = re.compile(pattern, re.IGNORECASE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = df["raw_line"].str.contains(rx, na=False)
        hits = df[mask]
        if hits.empty:
            continue

        examples = hits["raw_line"].head(3).tolist()
        suspicious.extend(hits["raw_line"].head(10).tolist())

        findings.append(PatternFinding(
            pattern_name=name,
            source=str(hits["source"].mode().iloc[0]) if not hits.empty else "",
            severity=severity,
            count=len(hits),
            first_ts=str(hits["timestamp"].min()),
            last_ts=str(hits["timestamp"].max()),
            examples=examples,
            description=f"{desc}: {len(hits)} occurrences",
            method="regex",
        ))

    return findings, list(dict.fromkeys(suspicious))[:_MAX_LLM_LINES]


# ── Pass 2: LLM ──────────────────────────────────────────────────────────────

def _llm_pass(suspicious_lines: list[str], cfg: dict) -> list[PatternFinding]:
    """Send suspicious lines to OpenAI and parse structured findings."""
    if not suspicious_lines:
        return []

    llm_cfg  = cfg.get("openai", {})
    api_key  = llm_cfg.get("api_key", "")
    base_url = llm_cfg.get("base_url", "")   # e.g. https://api.groq.com/openai/v1

    if not api_key:
        log.warning("[pattern_agent] No API key – skipping LLM pass")
        return []

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            **({ "base_url": base_url } if base_url else {}),
        )
    except ImportError:
        log.warning("[pattern_agent] openai package not installed – skipping LLM pass")
        return []

    model = llm_cfg.get("model", "gpt-4o")
    log.info(f"[pattern_agent] Sending {len(suspicious_lines)} lines to {model} for semantic analysis …")

    prompt = f"""You are a performance engineering expert analysing log lines from a load test.
Below are suspicious log lines extracted from .BLG, IIS, and LoadRunner logs.

Analyse them and return a JSON array of findings. Each finding must have these fields:
- "pattern_name": short snake_case label (e.g. "db_connection_pool_exhaustion")
- "severity": "info" | "warn" | "critical"
- "count": estimated number of occurrences
- "description": 1-2 sentence explanation of what is happening and why it is a problem
- "examples": list of up to 2 representative log lines

Return ONLY valid JSON — no markdown, no prose.

Log lines:
{chr(10).join(suspicious_lines)}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=cfg.get("openai", {}).get("max_tokens", 2048),
            temperature=0.1,
        )
        raw_json = response.choices[0].message.content.strip()
        items = json.loads(raw_json)
    except Exception as exc:
        log.error(f"[pattern_agent] OpenAI call failed: {exc}")
        return []

    results: list[PatternFinding] = []
    for item in (items if isinstance(items, list) else []):
        results.append(PatternFinding(
            pattern_name=item.get("pattern_name", "unknown"),
            source="llm_analysis",
            severity=item.get("severity", "warn"),
            count=int(item.get("count", 0)),
            first_ts="",
            last_ts="",
            examples=item.get("examples", []),
            description=item.get("description", ""),
            method="llm",
        ))

    log.info(f"[pattern_agent] OpenAI returned {len(results)} semantic patterns")
    return results


# ── Public entry point ────────────────────────────────────────────────────────

def run(df: pd.DataFrame, cfg: dict | None = None) -> list[dict]:
    cfg = cfg or {}
    regex_findings, suspicious = _regex_pass(df)
    llm_findings = _llm_pass(suspicious, cfg)
    all_findings = regex_findings + llm_findings
    log.info(f"[pattern_agent] Total patterns: {len(all_findings)} "
             f"({len(regex_findings)} regex + {len(llm_findings)} llm)")
    return [f.to_dict() for f in all_findings]
