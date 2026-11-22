"""Shared configuration for every script. All fixed experimental settings live here."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"
CLASSES_DIR = BUILD_DIR / "classes"
LIB_DIR = BUILD_DIR / "lib"
INPUT_DIR = ROOT / "data" / "inputs"
TEST_DIR = ROOT / "data" / "tests"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
LOG_DIR = RAW_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# ---- Inputs (same generator as Assignments 1-3) ------------------------------
SEED = 2026                     # numpy.random.default_rng(SEED + n), A then B
VALUE_RANGE = (-1.0, 1.0)       # uniform values in [-1, 1)

# ---- Execution environment -----------------------------------------------------
# LOCAL EXECUTION: all members, the driver and the baselines run on ONE laptop
# (i7-13620H: 6 P-cores with 2 hardware threads each = logical CPUs 0-11, where P-core c
# holds logical CPUs 2c and 2c+1; 4 E-cores = logical CPUs 12-15; 16 GB RAM).
# Members talk to each other and to the driver over the loopback interface.
PHYSICAL_MACHINES = 1
P_CORES = 6

# ---- CPU placement (affinity), the same rule for EVERY method -----------------------
# Without affinity Windows moved a single compute thread to an E-core (n = 3200, seq:
# 7.35 s pinned to an E-core, 3.55 s pinned to a P-core, 7.8 s unpinned), which would
# make the scaling depend on scheduler decisions. So every COMPUTE THREAD gets its own
# P-core, using the first hardware thread of the core (logical CPU 2c):
#   member m with t threads  -> P-cores m*t .. (m+1)*t - 1
#   baseline with p threads  -> P-cores 0 .. p-1
# The second hardware thread of each P-core is left idle. The driver (which does not
# compute) runs on the four E-cores. The JVM is told how many CPUs it has
# (-XX:ActiveProcessorCount) so that its own thread pools match its affinity.
DRIVER_CPUS = [12, 13, 14, 15]


def p_core_cpus(first_core, count):
    return [2 * c for c in range(first_core, first_core + count)]


def member_cpus(m, t):
    return p_core_cpus(m * t, t)


# ---- Strong scaling: fixed n, more members --------------------------------------
# n = 4800 was piloted and dropped: with 6 members every member needs a full copy of B,
# and the configuration did not fit in the memory left free on the laptop (see README).
STRONG_SIZES = [800, 1600, 3200, 4000]
MEMBER_COUNTS = [1, 2, 3, 4, 5, 6]   # W = worker processes (Hazelcast members), <= P_CORES
THREADS_PER_MEMBER = 1               # main study: one compute thread per member

# ---- Weak scaling: work per member fixed -----------------------------------------
# A dense product costs n^3 multiply-adds, so W members get the same work per member
# when n_W = n_1 * W^(1/3) (rounded). Keeping n (or n^2) per member fixed would not.
WEAK_N1 = 2400
WEAK_SIZES = {w: round(WEAK_N1 * w ** (1 / 3)) for w in MEMBER_COUNTS}

# ---- Workers vs processes: 6 compute threads in total, split differently ---------
SPLIT_N = 3200
SPLITS = [(1, 6), (2, 3), (3, 2), (6, 1)]   # (members W, threads per member t)

# ---- Baselines (single process, kernel-only timing) ------------------------------
# basic : MatMul.java of Assignment 1 (naive i-j-k), only where feasible
#         (Assignment 1: n = 1600 took 27 s per run; n = 3200 exceeded the 600 s budget)
# seq   : ParallelMatMul seq of Assignment 3 (blocked i-k-j, 256 x 256 tiles)
# par   : ParallelMatMul par of Assignment 3 with p threads
BASIC_SIZES = [800, 1600]
PAR_THREADS = [1, 2, 3, 4, 5, 6]

# ---- Measurement protocol --------------------------------------------------------
WARMUPS = 3                     # end-to-end warm-up runs per configuration (not reported)
MEASURED_REPS = 5
BASIC_WARMUPS = 2               # as in Assignment 1 for Java
BASELINE_WARMUPS = 5            # as in Assignment 3
TIME_BUDGET_S = 900             # wall time for one configuration (setup + all runs)
MEMORY_BUDGET_MB = 10000        # sum of the peaks of ALL processes of a configuration
MEMORY_SAMPLE_INTERVAL_S = 0.02
HIGH_PRIORITY = True

# ---- Toolchain ---------------------------------------------------------------------
JAVA = "java"
PYTHON = sys.executable
HAZELCAST_VERSION = "5.7.0"
# Flags recommended by Hazelcast for Java 17+ (module access used by the member).
JAVA_MODULE_FLAGS = [
    "--add-modules", "java.se",
    "--add-exports", "java.base/jdk.internal.ref=ALL-UNNAMED",
    "--add-opens", "java.base/java.lang=ALL-UNNAMED",
    "--add-opens", "java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens", "java.management/sun.management=ALL-UNNAMED",
    "--add-opens", "jdk.management/com.sun.management.internal=ALL-UNNAMED",
]
BASELINE_HEAP = "-Xmx3g"         # as in Assignments 1-3
# Heaps of the distributed processes are sized from the data each one holds, because the
# JVM (G1) lets its resident memory grow towards -Xmx before collecting: a fixed 5 GB
# heap made 6 members exhaust the RAM in a pilot. With S = 8 n^2 bytes (one matrix):
#   member: full B as received bytes + decoded (2S) and its share of A, B and C in the
#           map, as bytes and decoded (about 7S/W)
#   driver: A, B, C as double[][] plus the encoded A, B and the received C (6S)
# heap = 384 MB + 1.5 x that estimate.
HEAP_BASE_MB = 384
HEAP_FACTOR = 1.5


def _s_mb(n):
    return 8 * n * n / 2**20


def member_heap_mb(n, w):
    return int(HEAP_BASE_MB + HEAP_FACTOR * (2 * _s_mb(n) + 7 * _s_mb(n) / w))


def driver_heap_mb(n):
    return int(HEAP_BASE_MB + HEAP_FACTOR * 6 * _s_mb(n))


def classpath():
    return os.pathsep.join([str(CLASSES_DIR), str(LIB_DIR / "*")])


def member_command(cluster, t, n, w):
    return ([JAVA, f"-Xmx{member_heap_mb(n, w)}m", f"-XX:ActiveProcessorCount={t}"] + JAVA_MODULE_FLAGS
            + ["-cp", classpath(), "DistMember", cluster])


def driver_command(cluster, w, t, n, a_path, b_path, warmups, reps, out_path=None):
    args = [cluster, str(w), str(t), str(n), str(a_path), str(b_path), str(warmups), str(reps)]
    if out_path is not None:
        args.append(str(out_path))
    return ([JAVA, f"-Xmx{driver_heap_mb(n)}m", f"-XX:ActiveProcessorCount={len(DRIVER_CPUS)}"]
            + JAVA_MODULE_FLAGS + ["-cp", classpath(), "DistMatMul"] + args)


def baseline_command(method, p, n, a_path, b_path, warmups, reps, out_path=None):
    """method: basic (Assignment 1), seq or par (Assignment 3)."""
    if method == "basic":
        args = ["MatMul", str(n), str(a_path), str(b_path), str(warmups), str(reps)]
    else:
        args = ["ParallelMatMul", method, str(p), str(n), str(a_path), str(b_path),
                str(warmups), str(reps)]
    if out_path is not None:
        args.append(str(out_path))
    return [JAVA, BASELINE_HEAP, f"-XX:ActiveProcessorCount={p}", "-cp", str(CLASSES_DIR)] + args


def all_sizes():
    return sorted(set(STRONG_SIZES) | set(WEAK_SIZES.values()) | {SPLIT_N})


def input_paths(n):
    return INPUT_DIR / f"A_{n}.bin", INPUT_DIR / f"B_{n}.bin"
