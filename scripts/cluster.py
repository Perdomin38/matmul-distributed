"""Start a local Hazelcast cluster of W member processes, run one driver against it,
measure every process, and stop the cluster.

Processes of one configuration (all on this machine):
  W member JVMs (DistMember)  = the worker processes
  1 driver JVM  (DistMatMul)  = Hazelcast client; holds A, B, C; does not compute

CPU placement: member m is pinned to member_cpus(m, t) (one P-core per compute thread)
and the driver to DRIVER_CPUS (the E-cores), right after each process is created; the
JVMs also get -XX:ActiveProcessorCount (common.py). The affinity actually applied is
read back and returned.

Memory metric: PEAK RESIDENT MEMORY of each process (Windows peak working set, read
with psutil every 20 ms). Reported per process: driver, every member, the largest
member, and the sum over all processes of the configuration.

Setup times: `cluster_formation_s` is the time from launching the member processes until
the LAST member printed "member_ready" (JVM start + JIT pre-warm + joining); the driver
reports its own connection time and the wait until all W members had joined. These
are measured once per configuration and are not part of the per-multiplication time.

Limits: if the configuration exceeds TIME_BUDGET_S (wall, setup included) or the sum of
the process peaks exceeds MEMORY_BUDGET_MB, everything is killed and the status is
TIMEOUT / MEMORY_LIMIT. No time is invented for it."""
import subprocess
import time
import uuid

import psutil

from common import (DRIVER_CPUS, HIGH_PRIORITY, LOG_DIR, MEMORY_BUDGET_MB,
                    MEMORY_SAMPLE_INTERVAL_S, TIME_BUDGET_S, driver_command, member_command,
                    member_cpus)


def _peak_mb(proc):
    info = proc.memory_info()
    return getattr(info, "peak_wset", info.rss) / 2**20


def _priority(proc):
    if HIGH_PRIORITY:
        try:
            proc.nice(psutil.HIGH_PRIORITY_CLASS if psutil.WINDOWS else -10)
        except psutil.Error as e:
            print(f"  warning: high priority not applied ({e!r})")


def _affinity(proc, cpus):
    proc.cpu_affinity(cpus)
    return " ".join(map(str, proc.cpu_affinity()))


def _kill(proc):
    try:
        proc.kill()
    except psutil.Error:
        pass


def run_cluster(w, t, n, a_path, b_path, warmups, reps, out_path=None, tag="run"):
    """Returns a dict with status, stdout, stderr and the measurements."""
    cluster = f"mm-{uuid.uuid4().hex[:12]}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    member_logs, members, affinities = [], [], []
    start_epoch = time.time()
    start = time.perf_counter()
    for m in range(w):
        log_path = LOG_DIR / f"{tag}_member{m}.log"
        log = open(log_path, "w")
        member_logs.append((log, log_path))
        p = psutil.Popen(member_command(cluster, t, n, w), stdout=log, stderr=subprocess.STDOUT, text=True)
        affinities.append(_affinity(p, member_cpus(m, t)))
        _priority(p)
        members.append(p)
    driver_log = LOG_DIR / f"{tag}_driver_stderr.log"
    drv_err = open(driver_log, "w")
    driver = psutil.Popen(driver_command(cluster, w, t, n, a_path, b_path, warmups, reps, out_path),
                          stdout=subprocess.PIPE, stderr=drv_err, text=True)
    driver_affinity = _affinity(driver, DRIVER_CPUS)
    _priority(driver)

    peaks = {p.pid: 0.0 for p in members + [driver]}
    status = "OK"
    stdout_chunks = []
    import threading

    def reader():
        for line in driver.stdout:
            stdout_chunks.append(line)

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    try:
        while driver.poll() is None:
            for p in members + [driver]:
                try:
                    peaks[p.pid] = max(peaks[p.pid], _peak_mb(p))
                except psutil.Error:
                    pass
            if any(p.poll() is not None for p in members):
                status = "ERROR(member exited)"
            elif time.perf_counter() - start > TIME_BUDGET_S:
                status = "TIMEOUT"
            elif sum(peaks.values()) > MEMORY_BUDGET_MB:
                status = "MEMORY_LIMIT"
            if status != "OK":
                _kill(driver)
                break
            time.sleep(MEMORY_SAMPLE_INTERVAL_S)
        driver.wait()
        th.join(timeout=10)
        wall = time.perf_counter() - start
        for p in members:
            try:
                peaks[p.pid] = max(peaks[p.pid], _peak_mb(p))
            except psutil.Error:
                pass
    finally:
        for p in members:
            _kill(p)
        for p in members:
            try:
                p.wait(timeout=30)
            except psutil.Error:
                pass
        for log, _ in member_logs:
            log.close()
        drv_err.close()

    stderr = driver_log.read_text(errors="replace")
    if status == "OK" and driver.returncode != 0:
        status = f"ERROR({driver.returncode}): {stderr.strip()[-300:]}"

    ready = []
    for _, path in member_logs:
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("member_ready "):
                ready.append(int(line.split()[1]) / 1000.0)
    formation = (max(ready) - start_epoch) if len(ready) == w else float("nan")
    member_peaks = [peaks[p.pid] for p in members]
    return {
        "cluster": cluster, "status": status, "stdout": "".join(stdout_chunks), "stderr": stderr,
        "wall_time_s": wall, "cluster_formation_s": formation,
        "driver_peak_mb": peaks[driver.pid], "member_peaks_mb": member_peaks,
        "max_member_peak_mb": max(member_peaks), "sum_member_peak_mb": sum(member_peaks),
        "total_peak_mb": sum(member_peaks) + peaks[driver.pid],
        "member_affinity": " | ".join(affinities), "driver_affinity": driver_affinity,
    }


def parse_driver(stdout):
    """Split the driver output into the setup tuple, run rows and task rows."""
    setup, runs, tasks = None, [], []
    for line in stdout.splitlines():
        f = line.strip().split(",")
        if f[0] == "setup":
            setup = {"connect_s": float(f[1]), "cluster_wait_s": float(f[2])}
        elif f[0] == "run":
            runs.append({"phase": f[1], "run": int(f[2]), "total_s": float(f[3]),
                         "distribute_s": float(f[4]), "execute_s": float(f[5]),
                         "collect_s": float(f[6]), "bytes_to_cluster": int(f[7]),
                         "bytes_from_cluster": int(f[8]), "checksum": f[9]})
        elif f[0] == "task":
            tasks.append({"phase": f[1], "run": int(f[2]), "member": int(f[3]), "rows": int(f[4]),
                          "fetchA_s": float(f[5]), "fetchB_s": float(f[6]),
                          "compute_s": float(f[7]), "storeC_s": float(f[8]),
                          "task_s": float(f[9]), "bytes_A": int(f[10]),
                          "bytes_B_local": int(f[11]), "bytes_B_remote": int(f[12]),
                          "C_local": f[13] == "true", "address": f[14]})
    return setup, runs, tasks
