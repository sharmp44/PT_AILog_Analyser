"""
PT AI Log Mining Pipeline – Main Orchestrator
=============================================

Usage:
    # Process specific files
    python main.py run path/to/file.blg path/to/iis.log path/to/results.csv

    # Watch drop folder continuously
    python main.py watch

    # Set your OpenAI key before running
    export OPENAI_API_KEY=sk-...
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel

# ── Ensure the project root is on sys.path ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from Utils.config import load_config
from Utils.logger import get_logger

from ingest import blg_parser, iis_parser, lr_parser
from ingest.file_watcher import detect_type, watch as watch_folder

from parse.normaliser import normalise
from parse.context_tagger import tag
from parse.store import Store

from agents import anomaly_agent, trend_agent, threshold_agent, pattern_agent, causal_agent
from correlation import timeline as timeline_mod, rca_engine
from report import generator

log     = get_logger("pipeline")
console = Console()
app     = typer.Typer(help="PT AI Log Mining Pipeline")


# ── Parsers registry ──────────────────────────────────────────────────────────
_PARSERS = {
    "blg":        blg_parser.parse,
    "iis":        iis_parser.parse,
    "loadrunner": lr_parser.parse,
    "lr":         lr_parser.parse,
}


def _parse_file(file_path: Path, kind: str, cfg: dict) -> pd.DataFrame | None:
    parser = _PARSERS.get(kind)
    if parser is None:
        log.warning(f"No parser for type '{kind}' — skipping {file_path.name}")
        return None
    try:
        return parser(file_path, cfg)
    except Exception as exc:
        log.error(f"Failed to parse {file_path.name}: {exc}", exc_info=True)
        return None


def run_pipeline(files: list[Path], cfg: dict) -> dict:
    """
    Full end-to-end pipeline.

    1. Parse all input files (parallel)
    2. Normalise + tag + store
    3. Run AI agents (parallel)
    4. Correlation + RCA
    5. Report generation
    """
    console.rule("[bold blue]① INGEST")

    # ── Parse files ───────────────────────────────────────────────────────────
    dfs: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {}
        for f in files:
            kind = detect_type(f)
            if kind is None:
                log.warning(f"Cannot detect type for {f.name}, skipping")
                continue
            futures[ex.submit(_parse_file, f, kind, cfg)] = f

        for future in as_completed(futures):
            df = future.result()
            if df is not None and not df.empty:
                dfs.append(df)

    if not dfs:
        console.print("[red]No data parsed from any input file. Aborting.[/red]")
        raise SystemExit(1)

    console.rule("[bold green]② PARSE / NORMALISE / STORE")

    # ── Merge all sources ─────────────────────────────────────────────────────
    combined = pd.concat(dfs, ignore_index=True)
    combined = normalise(combined)
    combined = tag(combined, cfg)

    run_id   = combined["run_id"].iloc[0] if "run_id" in combined.columns else "unknown"
    db_path  = cfg.get("pipeline", {}).get("db_path", "./output/pipeline.duckdb")

    with Store(db_path) as store:
        store.insert(combined)

    console.print(f"[green]✓[/green] {len(combined)} events stored, run_id=[bold]{run_id}[/bold]")

    console.rule("[bold purple]③ AI AGENTS")

    # ── Run agents in parallel ────────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_anomaly   = ex.submit(anomaly_agent.run,   combined, cfg)
        f_trend     = ex.submit(trend_agent.run,     combined, cfg)
        f_threshold = ex.submit(threshold_agent.run, combined, cfg)
        f_pattern   = ex.submit(pattern_agent.run,   combined, cfg)

    anomaly_f   = f_anomaly.result()
    trend_f     = f_trend.result()
    threshold_f = f_threshold.result()
    pattern_f   = f_pattern.result()

    console.print(
        f"[green]✓[/green] Agents: "
        f"anomaly={len(anomaly_f)}, trend={len(trend_f)}, "
        f"threshold={len(threshold_f)}, pattern={len(pattern_f)}"
    )

    # ── Causal agent (sequential – needs all agent outputs) ───────────────────
    console.rule("[bold orange3]④ CORRELATION")
    causal_result = causal_agent.run(anomaly_f, trend_f, threshold_f, pattern_f, cfg)
    timeline = timeline_mod.build(anomaly_f, trend_f, threshold_f, pattern_f)

    output_dir = cfg.get("pipeline", {}).get("output_folder", "./output")
    rca = rca_engine.build(
        causal_result, timeline,
        anomaly_f, trend_f, threshold_f, pattern_f,
        output_dir=output_dir,
    )

    console.rule("[bold cyan]⑤ REPORT")
    paths = generator.generate(rca, output_dir, run_id=run_id, cfg=cfg, export_pdf=True)

    console.print(Panel(
        f"[bold green]Pipeline complete![/bold green]\n\n"
        f"Verdict:  [bold]{'[red]FAIL' if rca['verdict']=='FAIL' else '[green]PASS'}[/bold]\n"
        f"Findings: {rca['finding_counts']['total']}\n"
        f"HTML:     {paths['html']}\n"
        f"PDF:      {paths['pdf'] or 'skipped (install weasyprint)'}",
        title="Results", border_style="green",
    ))

    return {**rca, "report_paths": {k: str(v) for k, v in paths.items() if v}}


# ── CLI commands ──────────────────────────────────────────────────────────────

@app.command()
def run(
    files: list[Path] = typer.Argument(..., help="Log files to process (.blg, .log, .csv)"),
    config: Path = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
):
    """Process one or more log files through the full pipeline."""
    cfg = load_config(config)
    run_pipeline(files, cfg)


@app.command()
def watch(
    drop_folder: str = typer.Option(None, "--folder", "-f", help="Folder to watch"),
    config: Path = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
):
    """Watch a folder and auto-process new log files."""
    cfg = load_config(config)
    folder = drop_folder or cfg.get("pipeline", {}).get("drop_folder", "./logs_drop")

    def callback(path: Path, kind: str):
        run_pipeline([path], cfg)

    watch_folder(folder, callback)


if __name__ == "__main__":
    app()
