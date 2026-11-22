"""Compile the Java code with Maven (Hazelcast is the only dependency, see pom.xml) and
record the environment (hardware, OS, tool versions, process/thread configuration) in
results/raw/environment.txt."""
import platform
import shutil
import subprocess

import psutil

from common import (DRIVER_CPUS, HAZELCAST_VERSION, JAVA, LIB_DIR, MEMBER_COUNTS, PAR_THREADS,
                    PHYSICAL_MACHINES, PYTHON, RAW_DIR, ROOT, SPLITS, member_cpus)


def version(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True, shell=isinstance(cmd, str))
    return (out.stdout + out.stderr).strip()


def ps(command):
    return version(["powershell", "-NoProfile", "-Command", command])


def main():
    mvn = shutil.which("mvn") or shutil.which("mvn.cmd")
    cmd = [mvn, "-B", "-q", "package", "dependency:copy-dependencies"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)
    jars = sorted(p.name for p in LIB_DIR.glob("*.jar"))

    import numpy
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    windows = platform.system() == "Windows"
    java_version = version([JAVA, "-version"]).splitlines()
    mvn_version = version([mvn, "-v"]).splitlines()
    lines = [
        f"CPU: {ps('(Get-CimInstance Win32_Processor).Name') if windows else platform.processor()}",
        f"Physical cores: {psutil.cpu_count(logical=False)}",
        f"Logical processors: {psutil.cpu_count(logical=True)}",
        "  (i7-13620H: 6 P-cores with Hyper-Threading = logical CPUs 0-11, P-core c = CPUs 2c, 2c+1;",
        "   4 E-cores without it = logical CPUs 12-15; mapping measured in Assignment 3)",
        f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB",
        f"OS: {platform.platform()}",
        f"Power plan: {ps('powercfg /getactivescheme') if windows else 'n/a'}",
        f"Physical machines used: {PHYSICAL_MACHINES} (LOCAL execution, loopback network 127.0.0.1)",
        "",
        f"Java: {java_version[0]} / {java_version[-1]}",
        f"Maven: {mvn_version[0] if mvn_version else 'unknown'}",
        f"Hazelcast: {HAZELCAST_VERSION} (jars in build/lib: {', '.join(jars)})",
        "Hazelcast settings: see src/java/ClusterConfig.java (TCP/IP join on 127.0.0.1,",
        "  maps mm-A/mm-B/mm-C with backup-count 0, executor mm-exec, phone-home off)",
        "JVM heaps: sized per configuration (common.member_heap_mb / driver_heap_mb);",
        "  baselines -Xmx3g. Every JVM gets -XX:ActiveProcessorCount = its number of CPUs.",
        f"Python: {platform.python_implementation()} {platform.python_version()}",
        f"NumPy (inputs and reference only): {numpy.__version__}",
        f"psutil (memory, affinity, priority): {psutil.__version__}",
        "",
        "Processes and threads:",
        f"  members W tested: {MEMBER_COUNTS}; splits (W, t): {SPLITS}; baseline p: {PAR_THREADS}",
        "  worker = compute thread; process = one JVM (member, driver or baseline)",
        "  member m with t compute threads is pinned to logical CPUs "
        f"{{2c : c = m t .. (m+1) t - 1}}, e.g. W=6,t=1: {[member_cpus(m, 1) for m in range(6)]}",
        f"  driver (Hazelcast client, does not compute) pinned to logical CPUs {DRIVER_CPUS}",
        "  baseline with p threads pinned to logical CPUs 0, 2, ..., 2(p-1)",
        "  Besides its t compute threads, every member JVM runs Hazelcast's own threads",
        "  (network I/O, partition operations, executor) and the JVM's JIT and GC threads;",
        "  these do not run the multiplication. No numerical library is used.",
    ]
    (RAW_DIR / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
