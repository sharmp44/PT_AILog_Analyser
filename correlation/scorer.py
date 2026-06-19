"""
Confidence Scorer
-----------------
Each root cause finding is scored High / Medium / Low based on
how many independent signal sources corroborate it.

A finding backed by .BLG + IIS + LR simultaneously = High confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from Utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ScoredFinding:
    hypothesis:       str
    evidence:         str
    sources_cited:    list[str]
    corroboration:    int      # number of distinct source types that support it
    confidence:       str      # "High" | "Medium" | "Low"
    original_rank:    int

    def to_dict(self) -> dict:
        return asdict(self)


_SOURCE_KEYWORDS = {
    "blg":        ["blg", "cpu", "mem", "disk", "network", "perfmon", "processor", "memory", "counter"],
    "iis":        ["iis", "http", "response_time", "latency", "bytes", "request", "uri", "status"],
    "loadrunner": ["lr", "loadrunner", "transaction", "vuser", "tps", "throughput", "sla"],
}


def _sources_from_text(text: str) -> list[str]:
    """Guess which log sources are referenced in an evidence/hypothesis string."""
    text_lower = text.lower()
    found = []
    for src, keywords in _SOURCE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(src)
    return list(set(found))


def score(hypotheses: list[dict]) -> list[dict]:
    """
    Assign confidence scores to LLM hypotheses.

    Args:
        hypotheses: list from causal_agent — [{rank, hypothesis, evidence, confidence}]

    Returns:
        List of ScoredFinding dicts, sorted by corroboration desc.
    """
    scored: list[ScoredFinding] = []

    for h in hypotheses:
        text = f"{h.get('hypothesis', '')} {h.get('evidence', '')}"
        sources = _sources_from_text(text)
        corroboration = len(sources)

        # Override LLM confidence with our rule
        if corroboration >= 3:
            confidence = "High"
        elif corroboration == 2:
            confidence = "Medium"
        else:
            # Fall back to LLM-provided confidence if only 1 source
            llm_conf = str(h.get("confidence", "Low"))
            confidence = llm_conf if llm_conf in ("High", "Medium", "Low") else "Low"

        scored.append(ScoredFinding(
            hypothesis=h.get("hypothesis", ""),
            evidence=h.get("evidence", ""),
            sources_cited=sources,
            corroboration=corroboration,
            confidence=confidence,
            original_rank=int(h.get("rank", 99)),
        ))

    scored.sort(key=lambda s: (-s.corroboration, s.original_rank))
    log.info(f"[scorer] {len(scored)} hypotheses scored")
    return [s.to_dict() for s in scored]
