"""
File Watcher Agent
------------------
Monitors the drop folder using `watchdog`.
On new file arrival → detects type → triggers the full pipeline.

Usage (standalone):
    python -m ingest.file_watcher --drop-folder ./logs_drop

Or via main.py watch command.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent
from watchdog.observers import Observer

from Utils.logger import get_logger

log = get_logger(__name__)

# Supported file extensions and the log type they map to
_EXT_MAP: dict[str, str] = {
    ".blg": "blg",
    ".csv": "auto",   # could be BLG pre-converted CSV or LR results CSV
    ".log": "auto",   # IIS or LR vuser log
    ".txt": "lr",
    ".tsv": "lr",
}

# File name hints that override extension-based detection
_NAME_HINTS: list[tuple[str, str]] = [
    ("vuser",    "lr"),
    ("output",   "lr"),
    ("results",  "lr"),
    ("sql",      "sql"),   # SQL Server PerfMon CSV
    ("sqlserver","sql"),
    ("mssql",    "sql"),
    ("perfmon",  "blg"),
    ("iis",      "iis"),
    ("access",   "iis"),
    ("w3svc",    "iis"),
    ("u_ex",     "iis"),   # IIS default log name pattern
    ("ex",       "iis"),
]


def detect_type(path: Path) -> str | None:
    """Return 'blg' | 'iis' | 'lr' | None."""
    suffix = path.suffix.lower()
    if suffix == ".blg":
        return "blg"

    name_lower = path.name.lower()
    for hint, kind in _NAME_HINTS:
        if hint in name_lower:
            return kind

    # Sniff first lines for IIS W3C / SQL PerfMon / LR format
    if suffix in (".log", ".txt", ".csv"):
        try:
            with open(path, errors="replace") as fh:
                first = fh.readline()
                second = fh.readline()
            content = (first + second).lower()
            if "#software: microsoft internet information" in content:
                return "iis"
            if "sqlserver:" in content:
                return "sql"
            if "notify:" in content or "action.c" in content:
                return "lr"
        except Exception:
            pass

    if suffix in _EXT_MAP:
        return _EXT_MAP[suffix] if _EXT_MAP[suffix] != "auto" else None

    return None


class DropFolderHandler(FileSystemEventHandler):
    """Handles file-created / file-moved events in the drop folder."""

    def __init__(self, callback: Callable[[Path, str], None]):
        super().__init__()
        self._callback = callback
        self._seen: set[str] = set()

    def _handle(self, path: Path) -> None:
        key = str(path)
        if key in self._seen:
            return
        self._seen.add(key)

        # Wait briefly for the file to finish writing
        time.sleep(1)

        kind = detect_type(path)
        if kind is None:
            log.warning(f"Unrecognised file type, skipping: {path.name}")
            return

        log.info(f"[watcher] New {kind.upper()} file detected: {path.name}")
        try:
            self._callback(path, kind)
        except Exception as exc:
            log.error(f"Pipeline error processing {path.name}: {exc}", exc_info=True)

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory:
            self._handle(Path(event.src_path))

    def on_moved(self, event: FileMovedEvent) -> None:
        if not event.is_directory:
            self._handle(Path(event.dest_path))


def watch(drop_folder: str | Path, callback: Callable[[Path, str], None]) -> None:
    """
    Block and watch `drop_folder` indefinitely.
    Calls callback(path, log_type) for each new file.
    """
    folder = Path(drop_folder)
    folder.mkdir(parents=True, exist_ok=True)

    handler  = DropFolderHandler(callback)
    observer = Observer()
    observer.schedule(handler, str(folder), recursive=False)
    observer.start()
    log.info(f"[watcher] Watching {folder.resolve()} for new log files …  (Ctrl+C to stop)")

    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        observer.stop()
        log.info("[watcher] Stopped.")
    observer.join()
