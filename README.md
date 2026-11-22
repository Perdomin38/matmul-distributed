# Assignment 4: Distributed Matrix Multiplication

Big Data, Grado en Ciencia e Ingeniería de Datos, ULPGC, 2026–2027. Author: Daniel Perdomo Medina.

**Research question.** How does a distributed matrix multiplication implemented with
Hazelcast scale with the number of worker processes (members), and how much of its
end-to-end time is computation, data transfer and framework overhead, compared with the
validated sequential and shared-memory parallel versions of the earlier assignments?

Report: [`report/Assignment04_PerdomoMedina_Report.pdf`](report/Assignment04_PerdomoMedina_Report.pdf).

> **Local execution.** All processes run on **one** laptop and communicate over the loopback
> interface (127.0.0.1). The study measures framework behaviour, process overhead and the
> cost of moving data between JVM processes. It does **not** measure inter-machine network
> performance or multi-machine memory capacity.

## What is compared

| Method | Description |
|---|---|
| `dist` | Hazelcast 5.7.0 cluster of W member processes (the workers) plus one driver process (a Hazelcast client that holds A, B and C but does no computation). A and C are split into W bands of rows, B into W chunks of rows. Band m of A, chunk m of B and band m of C are stored on member m, and band task m runs on member m. Each task fetches the W−1 chunks of B it does not hold, computes its band with the blocked kernel of Assignment 3 (unchanged), and stores its C band locally. The driver then collects C. |
| `basic` | Naive i-j-k Java kernel of Assignment 1 ([matmul-basic-benchmark](https://github.com/Perdomin38/matmul-basic-benchmark), `MatMul.java`, copied unchanged). |
| `seq` / `par` | Blocked kernel of Assignment 3, sequential and with p Java threads ([matmul-parallel](https://github.com/Perdomin38/matmul-parallel), `ParallelMatMul.java`, copied unchanged). |

All methods use float64, `double[n][n]` in the JVM, and the same input files. The
distributed result is **bitwise identical** to `seq`.

**Timing boundaries.** `dist` is timed **end-to-end** in the driver. The clock starts when A
and B are in the driver's memory and stops when the complete C is back in it, so it
includes distribution, execution and collection. The baselines time only their **kernel**.
The two are never compared without saying so. The member-side phases (fetch, compute,
store) are also recorded, so a kernel-only comparison is possible.

**Experiments** (all settings in [`scripts/common.py`](scripts/common.py)):

* **Strong scaling:** n ∈ {800, 1600, 3200, 4000} with W ∈ {1, …, 6} members and 1 compute thread each.
* **Weak scaling:** n_W = 2400 · W^(1/3), so that the n³/W multiply-adds per member stay
  constant (n = 2400, 3024, 3461, 3810, 4104, 4361).
* **Workers vs processes:** n = 3200 with 6 compute threads in total, split as W members × t threads:
  1×6, 2×3, 3×2, 6×1.
* **Baselines:** `basic` at n = 800 and 1600 only. In Assignment 1, n = 3200 exceeded the 600 s
  budget. `seq` runs at every size, and `par` with p = 1…6 at the strong-scaling sizes.

**CPU placement (identical for every method).** Each compute thread is pinned to its own
P-core, on that core's first hardware thread (logical CPU 2c). Member m with t threads
gets P-cores m·t … (m+1)·t−1, and a baseline with p threads gets P-cores 0…p−1. The driver
runs on the four E-cores (CPUs 12–15). Every JVM gets `-XX:ActiveProcessorCount` equal to
its CPU count. This rule is needed because, without affinity, Windows moved a single
compute thread to an E-core: at n = 3200, `seq` took 7.35 s pinned to an E-core, 3.55 s
pinned to a P-core and 7.8 s unpinned. Six P-cores therefore limit W to 6.

**Protocol.** Each configuration starts a fresh cluster and driver. It runs 3 end-to-end
warm-ups, which are not reported, then 5 measured multiplications on the same cluster.
The baselines use 5 warm-ups (2 for `basic`) and 5 measured runs, as in Assignments 1 and 3.
Results are reported as the median and IQR. Every result checksum is compared with NumPy.
Memory is the peak working set of every process, sampled with psutil every 20 ms.
Budgets are 900 s of wall time per configuration and 10 000 MB for the sum of all its
processes' memory.

## Requirements

* Java JDK 24 (tested with Oracle JDK 24.0.1, HotSpot) and Maven 3.9 (tested with 3.9.9) on the PATH.
  Maven downloads the single dependency, `com.hazelcast:hazelcast:5.7.0`, into `build/lib`.
* Python 3.11 with the packages of `requirements.txt` (`pip install -r requirements.txt`).
  Python is used only to generate inputs, start and stop the processes, check results,
  sample memory and analyse the results.
* Free TCP ports 5701–5720 on 127.0.0.1.

## How to reproduce

From the `scripts/` folder:

```bash
python build.py            # mvn package + copy Hazelcast jar; writes results/raw/environment.txt
python generate_inputs.py  # data/inputs/A_n.bin, B_n.bin (n = 800, 1600, 3200 identical to Assignments 1-3)
python verify.py           # correctness through real clusters -> results/raw/correctness.csv
python run_baselines.py    # basic, seq, par -> results/raw/baseline_raw.csv, baseline_status.csv
python run_distributed.py  # strong, weak, split -> results/raw/dist_runs.csv, dist_tasks.csv, dist_status.csv
python run_baselines.py --methods seq --output-dir ../results/raw/seq_session2   # second seq session
python analyze.py          # tables (results/tables) and figures (results/figures)
```

**Two sessions of `seq`.** In the first baseline session, `seq` at n ≥ 3461 had IQRs of up to
11 % and ran up to 25 % slower than expected; the cause was not identified. A second session
was stable (IQR ≤ 1.2 %) and agrees with `par` at p = 1 and with the distributed compute phase
at W = 1. `analyze.py` therefore uses the second session as the `seq` reference for **all**
sizes. Both sessions are kept and compared in `results/tables/T7_seq_sessions.*`.

The console output of every step is in `results/raw/*_log.txt`, and the Hazelcast logs of every
member and driver are in `results/raw/logs/`.

A cluster can also be started by hand. Run each member in its own terminal from the project root:

```bash
java -Xmx2g -cp "build/classes;build/lib/*" DistMember mycluster        # repeat W times
java -Xmx2g -cp "build/classes;build/lib/*" DistMatMul mycluster 2 1 1600 data/inputs/A_1600.bin data/inputs/B_1600.bin 3 5
```

On Linux or macOS, use `:` instead of `;` in the classpath. The benchmark scripts also add the
Java module flags listed in `common.py` and the CPU affinity described above.

## Repository layout

```
src/java/DistMatMul.java      driver (Hazelcast client): partitioning, placement, timed phases
src/java/BandTask.java        task run on a member: fetch A band and B, compute, store C band
src/java/DistMember.java      worker process (Hazelcast member)
src/java/ClusterConfig.java   Hazelcast configuration (discovery, maps, executor)
src/java/RowCodec.java        rows <-> byte[] (little-endian float64)
src/java/MatMul.java          basic baseline, unchanged from Assignment 1
src/java/ParallelMatMul.java  seq / par baselines and the kernel, unchanged from Assignment 3
pom.xml                       Maven build (Hazelcast 5.7.0)
scripts/                      build, inputs, verification, benchmarks, analysis
results/raw/                  every raw measurement, status, correctness, environment, process logs
results/tables/               summary tables (CSV and Markdown)
results/figures/              figures used in the report
report/                       PDF of the report
```

Raw CSV columns: every row is labelled with `experiment, n, members, threads_per_member,
total_threads, physical_machines, dtype`. `dist_runs.csv` has one row per multiplication,
with `phase` set to warmup or measured, the times of the three sequential driver phases and
the payload bytes. `dist_tasks.csv` has one row per band task, with the member-side phases
and the local and remote bytes of B. `dist_status.csv` has one row per configuration, with
the status, setup times, peak memory of every process and the applied CPU affinity.

## Use of AI

An AI assistant (Claude, Anthropic) was used to help plan the experiment, write and debug the
code and scripts, and draft and revise the English of the report. All measurements come from
running this code on the machine described above; the author checked the code, the results and
the references.
