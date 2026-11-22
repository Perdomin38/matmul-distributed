"""Single-process baselines, measured on the same machine, inputs and CPU placement rule
as the distributed method.

  basic : MatMul (Assignment 1, naive i-j-k), n in BASIC_SIZES, 1 thread
  seq   : ParallelMatMul seq (Assignment 3, blocked i-k-j), every size used by the
          distributed experiments, 1 thread
  par   : ParallelMatMul par (Assignment 3), p in PAR_THREADS, n in STRONG_SIZES

The programs are copied unchanged from Assignments 1 and 3 (src/java/MatMul.java and
src/java/ParallelMatMul.java). They time ONLY the multiplication kernel (inputs are read
and C is allocated before the timer); this boundary differs from the end-to-end time of
the distributed method, and the analysis never mixes the two without saying so.

CPU placement: a baseline with p threads is pinned to P-cores 0..p-1 (logical CPUs
0, 2, ..., 2(p-1)), i.e. one P-core per compute thread, exactly as the members.

Memory: peak resident memory (Windows peak working set) of the process, psutil, 20 ms.
Outputs: results/raw/baseline_raw.csv and baseline_status.csv.

Usage: python run_baselines.py [--methods basic seq par] [--sizes ...] [--output-dir DIR]"""
import argparse
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil

from common import (BASELINE_WARMUPS, BASIC_SIZES, BASIC_WARMUPS, HIGH_PRIORITY,
                    MEASURED_REPS, MEMORY_BUDGET_MB, MEMORY_SAMPLE_INTERVAL_S, PAR_THREADS,
                    RAW_DIR, STRONG_SIZES, TIME_BUDGET_S, all_sizes, baseline_command,
                    input_paths, p_core_cpus)
from run_distributed import reference_checksum


def run_process(cmd, cpus):
    start = time.perf_counter()
    proc = psutil.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc.cpu_affinity(cpus)
    affinity = " ".join(map(str, proc.cpu_affinity()))
    if HIGH_PRIORITY:
        proc.nice(psutil.HIGH_PRIORITY_CLASS if psutil.WINDOWS else -10)
    peak, status = 0.0, "OK"
    while proc.poll() is None:
        try:
            info = proc.memory_info()
            peak = max(peak, getattr(info, "peak_wset", info.rss) / 2**20)
        except psutil.Error:
            pass
        if time.perf_counter() - start > TIME_BUDGET_S:
            status = "TIMEOUT"
        elif peak > MEMORY_BUDGET_MB:
            status = "MEMORY_LIMIT"
        if status != "OK":
            proc.kill()
            break
        time.sleep(MEMORY_SAMPLE_INTERVAL_S)
    stdout, stderr = proc.communicate()
    wall = time.perf_counter() - start
    if status == "OK" and proc.returncode != 0:
        status = f"ERROR({proc.returncode}): {stderr.strip()[:200]}"
    return status, stdout, wall, peak, affinity


def configurations(methods, sizes):
    configs = []
    if "basic" in methods:
        configs += [("basic", 1, n) for n in BASIC_SIZES if sizes is None or n in sizes]
    if "seq" in methods:
        configs += [("seq", 1, n) for n in all_sizes() if sizes is None or n in sizes]
    if "par" in methods:
        configs += [("par", p, n) for n in STRONG_SIZES for p in PAR_THREADS
                    if sizes is None or n in sizes]
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["basic", "seq", "par"])
    parser.add_argument("--sizes", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    raw_f = open(out / "baseline_raw.csv", "w", newline="")
    st_f = open(out / "baseline_status.csv", "w", newline="")
    raw_w = csv.DictWriter(raw_f, fieldnames=["method", "threads", "n", "dtype", "phase", "run",
                                              "calls", "time_s", "checksum", "checksum_ok"])
    st_w = csv.DictWriter(st_f, fieldnames=["method", "threads", "n", "dtype", "warmups", "reps",
                                            "affinity", "started_at", "status", "wall_time_s",
                                            "peak_memory_mb"])
    raw_w.writeheader()
    st_w.writeheader()
    for method, p, n in configurations(args.methods, args.sizes):
        warmups = BASIC_WARMUPS if method == "basic" else BASELINE_WARMUPS
        a_path, b_path = input_paths(n)
        started = datetime.now().isoformat(timespec="seconds")
        print(f"{method:5s} p={p} n={n:5d} running ...", flush=True)
        status, stdout, wall, peak, affinity = run_process(
            baseline_command(method, p, n, a_path, b_path, warmups, MEASURED_REPS),
            p_core_cpus(0, p))
        ref = reference_checksum(n)
        for line in stdout.strip().splitlines()[1:]:
            f = line.split(",")
            if method == "basic":            # phase,run,time_s,checksum
                phase, run, t, checksum, calls = f[0], f[1], f[2], f[3], 1
            else:                            # phase,run,calls,time_s,checksum
                phase, run, calls, t, checksum = f
            ok = abs(float(checksum) - ref) <= 1e-9 * max(1.0, abs(ref))
            raw_w.writerow({"method": method, "threads": p, "n": n, "dtype": "float64",
                            "phase": phase, "run": int(run), "calls": int(calls),
                            "time_s": float(t), "checksum": checksum, "checksum_ok": ok})
        st_w.writerow({"method": method, "threads": p, "n": n, "dtype": "float64",
                       "warmups": warmups, "reps": MEASURED_REPS, "affinity": affinity,
                       "started_at": started, "status": status, "wall_time_s": round(wall, 3),
                       "peak_memory_mb": round(peak, 1)})
        raw_f.flush()
        st_f.flush()
        print(f"{method:5s} p={p} n={n:5d} {status} wall={wall:.1f}s peak={peak:.0f}MB", flush=True)
    raw_f.close()
    st_f.close()


if __name__ == "__main__":
    main()
