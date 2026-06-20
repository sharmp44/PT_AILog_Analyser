"""
Sample Data Generator
---------------------
Creates realistic synthetic test logs for all three sources.
Run from the project root:  python sample_data/generate_samples.py
"""
from __future__ import annotations

import csv
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

random.seed(42)

OUT = Path(__file__).parent
START = datetime(2024, 6, 19, 9, 0, 0, tzinfo=timezone.utc)


# ── IIS Log ──────────────────────────────────────────────────────────────────

def gen_iis(path: Path, n: int = 500) -> None:
    """Generate a W3C IIS log with realistic latency spikes."""
    uris  = ["/api/orders", "/api/products", "/health", "/api/users", "/api/search"]
    ips   = [f"10.0.0.{i}" for i in range(1, 11)]

    lines = [
        "#Software: Microsoft Internet Information Services 10.0",
        f"#Date: {START.strftime('%Y-%m-%d %H:%M:%S')}",
        "#Fields: date time c-ip cs-method cs-uri-stem sc-status sc-bytes cs-bytes time-taken",
    ]
    ts = START
    for i in range(n):
        ts += timedelta(seconds=random.uniform(0.5, 3))
        uri    = random.choice(uris)
        ip     = random.choice(ips)
        method = random.choice(["GET", "GET", "GET", "POST"])

        # Inject anomaly from minute 18-22 (simulates a spike mid-test)
        elapsed_min = (ts - START).total_seconds() / 60
        if 18 < elapsed_min < 22:
            status    = random.choices([200, 500, 503], weights=[60, 25, 15])[0]
            time_taken = random.randint(2500, 8000)   # SLA breach
        elif elapsed_min > 25:
            status    = random.choices([200, 200, 500], weights=[70, 20, 10])[0]
            time_taken = random.randint(800, 2100)
        else:
            status    = random.choices([200, 302, 404, 500], weights=[88, 5, 5, 2])[0]
            time_taken = random.randint(80, 600)

        sc_bytes = random.randint(1000, 50000)
        cs_bytes = random.randint(100, 2000)
        lines.append(
            f"{ts.strftime('%Y-%m-%d')} {ts.strftime('%H:%M:%S')} "
            f"{ip} {method} {uri} {status} {sc_bytes} {cs_bytes} {time_taken}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[IIS]  {path}  ({len(lines)-3} entries)")


# ── BLG (CSV export) ─────────────────────────────────────────────────────────

def gen_blg_csv(path: Path, n: int = 300) -> None:
    """
    Simulate a PerfMon CSV export (as produced by relog -f CSV).
    """
    headers = [
        r"(PDH-CSV 4.0) (UTC)(0)",
        r"\\SERVER01\Processor(_Total)\% Processor Time",
        r"\\SERVER01\Memory\Available MBytes",
        r"\\SERVER01\Memory\% Committed Bytes In Use",
        r"\\SERVER01\PhysicalDisk(_Total)\Avg. Disk Queue Length",
        r"\\SERVER01\Process(w3wp)\Working Set",
        r"\\SERVER01\Process(w3wp)\Handle Count",
        r"\\SERVER01\Network Interface(Intel PRO)\Bytes Total/sec",
    ]

    rows = [headers]
    ts = START
    mem_available = 4096.0   # start with 4 GB available
    handle_count  = 500.0

    for i in range(n):
        ts += timedelta(seconds=10)
        elapsed_min = (ts - START).total_seconds() / 60

        # CPU spikes during anomaly window
        if 18 < elapsed_min < 22:
            cpu = random.uniform(88, 98)
        else:
            cpu = random.uniform(30, 65)

        # Memory leak: available RAM decreases steadily
        mem_available = max(512, mem_available - random.uniform(5, 20))
        mem_committed  = min(98, 60 + (4096 - mem_available) / 40)

        disk_queue = random.uniform(0.1, 0.8) if elapsed_min < 18 else random.uniform(1.5, 4.5)

        # Handle leak: handle count rises monotonically
        handle_count += random.uniform(0.5, 2.5)
        working_set = 800_000_000 + i * 500_000   # steady growth

        net_bytes = random.uniform(5_000_000, 50_000_000)

        rows.append([
            ts.strftime("%m/%d/%Y %H:%M:%S.000"),
            f"{cpu:.4f}",
            f"{mem_available:.4f}",
            f"{mem_committed:.4f}",
            f"{disk_queue:.4f}",
            f"{working_set:.0f}",
            f"{handle_count:.0f}",
            f"{net_bytes:.0f}",
        ])

    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"[BLG]  {path}  ({n} rows)")


# ── LoadRunner Results CSV ────────────────────────────────────────────────────

def gen_lr_results(path: Path) -> None:
    """Simulate a LoadRunner Analysis CSV results export."""
    transactions = [
        ("Login",          0.45, 0.38, 1.80, 0.80, 0.92, 0,  200.0),
        ("Search Products",0.30, 0.25, 0.90, 0.62, 0.75, 0,  350.0),
        ("Add to Cart",    0.55, 0.44, 2.40, 0.98, 1.20, 3,  180.0),
        ("Checkout",       1.20, 0.95, 6.80, 2.10, 2.80, 12,  80.0),  # SLA breaches
        ("Payment",        0.95, 0.80, 5.20, 1.75, 2.10, 8,   60.0),
        ("Order Confirm",  0.35, 0.28, 1.50, 0.70, 0.85, 0,  120.0),
    ]

    headers = [
        "Transaction Name", "Average (sec)", "Minimum (sec)", "Maximum (sec)",
        "90th Percentile", "95th Percentile", "Fail", "Throughput (hits/sec)",
        "Start Time",
    ]
    rows = [headers]
    base_ts = START.strftime("%Y-%m-%d %H:%M:%S")

    for (name, avg, mn, mx, p90, p95, fails, tps) in transactions:
        rows.append([name, avg, mn, mx, p90, p95, fails, tps, base_ts])

    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"[LR]   {path}  ({len(rows)-1} transactions)")


# ── LoadRunner Vuser Log ──────────────────────────────────────────────────────

def gen_lr_vuser_log(path: Path, n_iter: int = 60) -> None:
    """Simulate a vuser_0.log file with transaction logs and errors."""
    lines = []
    ts = START

    for i in range(n_iter):
        ts += timedelta(seconds=random.uniform(3, 8))
        ts_str = ts.strftime("%m/%d/%Y %H:%M:%S")
        tx     = random.choice(["Checkout", "Payment", "Search Products", "Login"])
        dur    = round(random.uniform(0.3, 6.5), 3)
        status = "Pass" if random.random() > 0.12 else "Fail"

        lines.append(f"Action.c(45): {ts_str}: Notify: Transaction \"{tx}\" started")
        lines.append(
            f"Action.c(52): {ts_str}: Notify: Transaction \"{tx}\" ended "
            f"with a duration of {dur} secs status: {status}"
        )

        # Inject errors in anomaly window
        elapsed_min = (ts - START).total_seconds() / 60
        if 18 < elapsed_min < 22 and random.random() > 0.6:
            lines.append(
                f"Action.c(60): {ts_str}: Error -27796: "
                f"Failed to connect to server \"app01:8080\": connection timed out"
            )
        if elapsed_min > 25 and random.random() > 0.8:
            lines.append(
                f"Action.c(70): {ts_str}: Warning: Think time 5.2 sec exceeds "
                f"the defined think time by 250%"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[LR]   {path}  ({len(lines)} lines)")


# ── SQL Server PerfMon CSV ────────────────────────────────────────────────────

def gen_sql_csv(path: Path, n: int = 300) -> None:
    """
    Simulate a PerfMon CSV export with SQL Server counters.
    Injects anomalies at minutes 18-22: deadlock spike, PLE drop, blocked processes.
    """
    headers = [
        r"(PDH-CSV 4.0) (UTC)(0)",
        r"\\SQLSERVER01\SQLServer:Buffer Manager\Buffer cache hit ratio",
        r"\\SQLSERVER01\SQLServer:Buffer Manager\Page life expectancy",
        r"\\SQLSERVER01\SQLServer:SQL Statistics\Batch Requests/sec",
        r"\\SQLSERVER01\SQLServer:SQL Statistics\SQL Compilations/sec",
        r"\\SQLSERVER01\SQLServer:SQL Statistics\SQL Re-Compilations/sec",
        r"\\SQLSERVER01\SQLServer:General Statistics\User Connections",
        r"\\SQLSERVER01\SQLServer:General Statistics\Processes blocked",
        r"\\SQLSERVER01\SQLServer:Locks(_Total)\Lock Waits/sec",
        r"\\SQLSERVER01\SQLServer:Locks(_Total)\Deadlocks/sec",
        r"\\SQLSERVER01\SQLServer:Memory Manager\Total Server Memory (KB)",
        r"\\SQLSERVER01\SQLServer:Memory Manager\Target Server Memory (KB)",
        r"\\SQLSERVER01\SQLServer:Databases(_Total)\Transactions/sec",
    ]

    rows = [headers]
    ts = START
    ple = 900.0        # Page Life Expectancy starts healthy

    for i in range(n):
        ts += timedelta(seconds=10)
        elapsed_min = (ts - START).total_seconds() / 60

        # Anomaly window 18-22: PLE drops, deadlocks spike, blocked processes rise
        if 18 < elapsed_min < 22:
            buffer_hit  = random.uniform(82, 91)     # drops below 95% SLA
            ple         = max(50, ple - random.uniform(30, 80))
            batch_req   = random.uniform(800, 1200)
            compilations = random.uniform(80, 150)
            recompilations = random.uniform(15, 35)   # exceeds SLA
            connections = random.randint(180, 220)
            blocked     = random.randint(8, 20)       # exceeds SLA
            lock_waits  = random.uniform(12, 25)      # exceeds SLA
            deadlocks   = random.uniform(0.3, 1.2)    # exceeds SLA
        elif elapsed_min > 25:
            buffer_hit  = random.uniform(93, 97)
            ple         = min(900, ple + random.uniform(5, 15))
            batch_req   = random.uniform(400, 700)
            compilations = random.uniform(30, 60)
            recompilations = random.uniform(2, 8)
            connections = random.randint(120, 160)
            blocked     = random.randint(0, 2)
            lock_waits  = random.uniform(0.5, 2.0)
            deadlocks   = 0.0
        else:
            buffer_hit  = random.uniform(96, 99.5)   # healthy
            ple         = min(900, ple + random.uniform(1, 5))
            batch_req   = random.uniform(300, 600)
            compilations = random.uniform(20, 50)
            recompilations = random.uniform(0.5, 4)
            connections = random.randint(80, 130)
            blocked     = random.randint(0, 1)
            lock_waits  = random.uniform(0.1, 1.0)
            deadlocks   = 0.0

        total_mem   = 8_192_000 + random.uniform(-10000, 10000)
        target_mem  = 8_192_000
        transactions = batch_req * random.uniform(1.5, 2.5)

        rows.append([
            ts.strftime("%m/%d/%Y %H:%M:%S.000"),
            f"{buffer_hit:.4f}",
            f"{ple:.0f}",
            f"{batch_req:.2f}",
            f"{compilations:.2f}",
            f"{recompilations:.2f}",
            f"{connections}",
            f"{blocked}",
            f"{lock_waits:.4f}",
            f"{deadlocks:.4f}",
            f"{total_mem:.0f}",
            f"{target_mem:.0f}",
            f"{transactions:.2f}",
        ])

    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"[SQL]  {path}  ({n} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    drop = Path(__file__).parent.parent / "logs_drop"
    drop.mkdir(exist_ok=True)

    gen_iis(drop / "iis_u_ex240619.log")
    gen_blg_csv(drop / "perfmon_server01.csv")
    gen_lr_results(drop / "lr_results.csv")
    gen_lr_vuser_log(drop / "vuser_0.log")
    gen_sql_csv(drop / "sql_server01.csv")

    print("\n✓ Sample files written to:", drop)
    print("  Run the pipeline with:")
    print(f"  python main.py run {drop}/*.log {drop}/*.csv")
