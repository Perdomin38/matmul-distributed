"""Correctness checks run BEFORE benchmarking.

Every distributed configuration of the study (W members x t threads, see common.py)
multiplies small test matrices end to end through a real local cluster, and the result
C written by the driver is compared with a known answer or with NumPy's A @ B (NumPy is
only the trusted reference). The baselines basic (Assignment 1) and seq (Assignment 3)
are checked the same way.

Tests
  known_2x2  : [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]            (exact)
               with W > 2 members some bands of A, B and C are EMPTY
  identity   : A x I = A, n = 37                                            (exact)
  zero       : A x 0 = 0, n = 37                                            (exact)
  random_157 : n = 157, not a multiple of 256 nor of W, so uneven bands,
               uneven B chunks and edge tiles are tested                    (tolerance)
  random_600 : n = 600, bands larger than one 256-tile                      (tolerance)

Tolerance: |C - C_ref| <= 1e-9 + 1e-9 |C_ref| (as in Assignments 1-3).
Bitwise check: every element of C is computed by one thread with the products added in
the same k order as seq, so the distributed result must be BITWISE identical to seq.
A band placed twice, a missing band, a wrong B chunk or a race would break it (the driver
also fills C with NaN before every run, so a band that never arrives cannot pass).
Each driver performs 2 multiplications on the same cluster and the last one is checked.
Results: results/raw/correctness.csv."""
import csv
import subprocess
import sys

import numpy as np

from cluster import run_cluster
from common import (MEMBER_COUNTS, RAW_DIR, SPLITS, TEST_DIR, THREADS_PER_MEMBER,
                    baseline_command)

RTOL = ATOL = 1e-9


def test_cases():
    rng = np.random.default_rng(12345)
    A37 = rng.uniform(-1, 1, (37, 37))
    A157, B157 = rng.uniform(-1, 1, (157, 157)), rng.uniform(-1, 1, (157, 157))
    A600, B600 = rng.uniform(-1, 1, (600, 600)), rng.uniform(-1, 1, (600, 600))
    return [
        ("known_2x2", np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]]),
         np.array([[19.0, 22.0], [43.0, 50.0]]), True),
        ("identity", A37, np.eye(37), A37, True),
        ("zero", A37, np.zeros((37, 37)), np.zeros((37, 37)), True),
        ("random_157", A157, B157, A157 @ B157, False),
        ("random_600", A600, B600, A600 @ B600, False),
    ]


def run_baseline(method, n, a_path, b_path, out_path):
    subprocess.run(baseline_command(method, 1, n, a_path, b_path, 0, 1, out_path),
                   check=True, capture_output=True)
    return np.fromfile(out_path, dtype="<f8").reshape(n, n)


def main():
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dist_configs = [(w, THREADS_PER_MEMBER) for w in MEMBER_COUNTS]
    dist_configs += [c for c in SPLITS if c not in dist_configs]
    rows, all_ok = [], True
    for name, A, B, expected, exact in test_cases():
        n = A.shape[0]
        a_path, b_path = TEST_DIR / f"{name}_A.bin", TEST_DIR / f"{name}_B.bin"
        A.astype("<f8").tofile(a_path)
        B.astype("<f8").tofile(b_path)
        seq = run_baseline("seq", n, a_path, b_path, TEST_DIR / f"{name}_C_seq.bin")
        results = [("seq", 1, 1, seq, "OK"),
                   ("basic", 1, 1, run_baseline("basic", n, a_path, b_path,
                                                TEST_DIR / f"{name}_C_basic.bin"), "OK")]
        for w, t in dist_configs:
            out = TEST_DIR / f"{name}_C_dist_W{w}_t{t}.bin"
            out.unlink(missing_ok=True)
            r = run_cluster(w, t, n, a_path, b_path, 0, 2, out_path=out,
                            tag=f"verify_{name}_W{w}_t{t}")
            C = np.fromfile(out, dtype="<f8").reshape(n, n) if out.exists() \
                else np.full((n, n), np.nan)
            results.append(("dist", w, t, C, r["status"]))
        for method, w, t, C, status in results:
            max_err = float(np.nanmax(np.abs(C - expected))) if not np.isnan(C).all() else float("nan")
            ok = status == "OK" and not np.isnan(C).any() and (
                bool(np.array_equal(C, expected)) if exact
                else bool(np.allclose(C, expected, rtol=RTOL, atol=ATOL)))
            same = bool(np.array_equal(C, seq))
            # basic (naive i-j-k) adds the products in a different order: only the
            # tolerance applies to it, not bitwise equality with seq.
            passed = ok and (same or method == "basic")
            all_ok &= passed
            rows.append({"test": name, "n": n, "method": method, "members": w,
                         "threads_per_member": t, "status": status,
                         "criterion": "exact" if exact else f"rtol={RTOL},atol={ATOL}",
                         "max_abs_error": f"{max_err:.3e}", "within_criterion": ok,
                         "bitwise_equal_to_seq": same, "passed": passed})
            print(f"{name:10s} n={n:3d} {method:5s} W={w} t={t} {status} "
                  f"max_abs_err={max_err:.3e} ok={ok} bitwise_equal_to_seq={same}", flush=True)
    with open(RAW_DIR / "correctness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
