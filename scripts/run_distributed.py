"""Benchmark of the distributed (Hazelcast) multiplication.

Experiments (all settings in common.py):
  strong : n in STRONG_SIZES fixed, W in MEMBER_COUNTS members, 1 compute thread each
  weak   : n_W = WEAK_N1 * W^(1/3), so the multiply-adds per member (n_W^3 / W) stay
           constant, W in MEMBER_COUNTS, 1 compute thread each
  split  : n = SPLIT_N, 8 compute threads in total split as W members x t threads

One configuration = a fresh cluster of W member processes + one driver process (see
cluster.py). The driver performs WARMUPS end-to-end multiplications (not reported) and
then MEASURED_REPS measured ones. Every result checksum is compared with NumPy's
sum(A @ B) (relative tolerance 1e-9, as in Assignments 1-3).

Outputs (in --output-dir, default results/raw):
  dist_runs.csv    one row per multiplication: total and the three sequential phases
  dist_tasks.csv   one row per band task: the member-side phases (they overlap between
                   members and must not be added across members)
  dist_status.csv  one row per configuration: status, setup times, peak memory per process

Usage: python run_distributed.py [--experiments strong weak split] [--sizes ...]
                                 [--members ...] [--output-dir DIR]"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from cluster import parse_driver, run_cluster
from common import (MEASURED_REPS, MEMBER_COUNTS, PHYSICAL_MACHINES, RAW_DIR, SPLIT_N, SPLITS,
                    STRONG_SIZES, THREADS_PER_MEMBER, WARMUPS, WEAK_SIZES, input_paths)

_reference = {}


def reference_checksum(n):
    if n not in _reference:
        a_path, b_path = input_paths(n)
        A = np.fromfile(a_path, dtype="<f8").reshape(n, n)
        B = np.fromfile(b_path, dtype="<f8").reshape(n, n)
        _reference[n] = float(np.sum(A @ B))
    return _reference[n]


def configurations(experiments, sizes, members):
    configs = []
    if "strong" in experiments:
        for n in sizes or STRONG_SIZES:
            for w in members or MEMBER_COUNTS:
                configs.append(("strong", n, w, THREADS_PER_MEMBER))
    if "weak" in experiments:
        for w in members or MEMBER_COUNTS:
            configs.append(("weak", WEAK_SIZES[w], w, THREADS_PER_MEMBER))
    if "split" in experiments:
        for w, t in SPLITS:
            configs.append(("split", SPLIT_N, w, t))
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["strong", "weak", "split"])
    parser.add_argument("--sizes", nargs="+", type=int, default=None, help="strong sizes only")
    parser.add_argument("--members", nargs="+", type=int, default=None)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--reps", type=int, default=MEASURED_REPS)
    parser.add_argument("--output-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    label = ["experiment", "n", "members", "threads_per_member", "total_threads",
             "physical_machines", "dtype"]
    runs_f = open(out / "dist_runs.csv", "w", newline="")
    tasks_f = open(out / "dist_tasks.csv", "w", newline="")
    status_f = open(out / "dist_status.csv", "w", newline="")
    runs_w = csv.DictWriter(runs_f, fieldnames=label + [
        "phase", "run", "total_s", "distribute_s", "execute_s", "collect_s", "bytes_to_cluster",
        "bytes_from_cluster", "bytes_member_to_member", "max_task_s", "checksum", "checksum_ok"])
    tasks_w = csv.DictWriter(tasks_f, fieldnames=label + [
        "phase", "run", "member", "rows", "fetchA_s", "fetchB_s", "compute_s", "storeC_s",
        "task_s", "bytes_A", "bytes_B_local", "bytes_B_remote", "C_local", "address"])
    status_w = csv.DictWriter(status_f, fieldnames=label + [
        "warmups", "reps", "started_at", "status", "wall_time_s", "cluster_formation_s",
        "client_connect_s", "cluster_wait_s", "driver_peak_mb", "max_member_peak_mb",
        "sum_member_peak_mb", "total_peak_mb", "member_peaks_mb", "member_affinity",
        "driver_affinity", "cluster"])
    for f in (runs_w, tasks_w, status_w):
        f.writeheader()

    for exp, n, w, t in configurations(args.experiments, args.sizes, args.members):
        base = {"experiment": exp, "n": n, "members": w, "threads_per_member": t,
                "total_threads": w * t, "physical_machines": PHYSICAL_MACHINES, "dtype": "float64"}
        started = datetime.now().isoformat(timespec="seconds")
        print(f"{exp:6s} n={n:5d} W={w} t={t} running ...", flush=True)
        a_path, b_path = input_paths(n)
        r = run_cluster(w, t, n, a_path, b_path, args.warmups, args.reps,
                        tag=f"{exp}_n{n}_W{w}_t{t}")
        setup, runs, tasks = parse_driver(r["stdout"])
        ref = reference_checksum(n)
        for run in runs:
            mine = [k for k in tasks if k["phase"] == run["phase"] and k["run"] == run["run"]]
            ok = abs(float(run["checksum"]) - ref) <= 1e-9 * max(1.0, abs(ref))
            runs_w.writerow({**base, **run,
                             "bytes_member_to_member": sum(k["bytes_B_remote"] for k in mine),
                             "max_task_s": max(k["task_s"] for k in mine), "checksum_ok": ok})
        for k in tasks:
            tasks_w.writerow({**base, **k})
        status_w.writerow({
            **base, "warmups": args.warmups, "reps": args.reps, "started_at": started,
            "status": r["status"], "wall_time_s": round(r["wall_time_s"], 3),
            "cluster_formation_s": round(r["cluster_formation_s"], 3),
            "client_connect_s": setup["connect_s"] if setup else "",
            "cluster_wait_s": setup["cluster_wait_s"] if setup else "",
            "driver_peak_mb": round(r["driver_peak_mb"], 1),
            "max_member_peak_mb": round(r["max_member_peak_mb"], 1),
            "sum_member_peak_mb": round(r["sum_member_peak_mb"], 1),
            "total_peak_mb": round(r["total_peak_mb"], 1),
            "member_peaks_mb": " ".join(f"{x:.1f}" for x in r["member_peaks_mb"]),
            "member_affinity": r["member_affinity"], "driver_affinity": r["driver_affinity"],
            "cluster": r["cluster"]})
        for f in (runs_f, tasks_f, status_f):
            f.flush()
        meas = sorted(x["total_s"] for x in runs if x["phase"] == "measured")
        med = meas[len(meas) // 2] if meas else float("nan")
        bad = sum(1 for x in runs if abs(float(x["checksum"]) - ref) > 1e-9 * max(1.0, abs(ref)))
        print(f"{exp:6s} n={n:5d} W={w} t={t} {r['status']} wall={r['wall_time_s']:.1f}s "
              f"median_total={med:.3f}s total_peak={r['total_peak_mb']:.0f}MB bad_checksums={bad}",
              flush=True)
    for f in (runs_f, tasks_f, status_f):
        f.close()


if __name__ == "__main__":
    main()
