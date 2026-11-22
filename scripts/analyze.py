"""Tables (results/tables, CSV + Markdown) and figures (results/figures) from the raw CSVs.

Statistics: the MEDIAN of the measured repetitions is the reported value and the
variability is the IQR (Q3 - Q1). Warm-up runs are excluded. Times in seconds.

Additivity: the driver phases distribute + execute + collect are sequential, so they add
up to the end-to-end total. Inside `execute`, the W band tasks run CONCURRENTLY on
different members, so their phases are never summed across members. For the breakdown
the analysis uses the SLOWEST task of each run (the one the driver waits for): its own
phases are sequential (fetch A + fetch B, compute, store C) and
  dispatch_wait = execute - slowest task time
is the remaining framework cost (sending the tasks, scheduling them on the members,
returning the results). Per phase, the median over the 5 runs is shown.
Bytes are application payload (MB = 2^20 bytes); memory is peak working set (MB).

Sequential reference: seq was measured in two sessions (results/raw/baseline_raw.csv and
results/raw/seq_session2/baseline_raw.csv). In the first one the runs for n >= 3461 had
IQRs of up to 11 % and were up to 25 % slower than in the second (IQR <= 1.2 %), which agrees
with par p = 1 and with the distributed compute phase with W = 1. The SECOND session is used
as the seq reference for ALL sizes; both are kept and compared in table T7_seq_sessions."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (FIGURES_DIR, MEMBER_COUNTS, RAW_DIR, SPLIT_N, SPLITS, STRONG_SIZES,  # noqa: E402
                    TABLES_DIR, WEAK_SIZES)

MB = 2**20
# Categorical palette (fixed slot order) of the dataviz reference palette.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRAY = "#6b6b68"
INK = "#2b2b29"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#9a9a95", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
                     "axes.spines.top": False, "axes.spines.right": False})


def q(x, p):
    return float(np.percentile(x, p))


def save(df, name, floatfmt=".4g"):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    (TABLES_DIR / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=floatfmt) + "\n",
                                           encoding="utf-8")


def load_seq2():
    d = pd.read_csv(RAW_DIR / "seq_session2" / "baseline_raw.csv")
    st = pd.read_csv(RAW_DIR / "seq_session2" / "baseline_status.csv")
    d = d[d.phase == "measured"]
    assert d.checksum_ok.all()
    return d, st


def load():
    runs = pd.read_csv(RAW_DIR / "dist_runs.csv")
    tasks = pd.read_csv(RAW_DIR / "dist_tasks.csv")
    status = pd.read_csv(RAW_DIR / "dist_status.csv")
    braw = pd.read_csv(RAW_DIR / "baseline_raw.csv")
    bstat = pd.read_csv(RAW_DIR / "baseline_status.csv")
    runs = runs[runs.phase == "measured"]
    tasks = tasks[tasks.phase == "measured"]
    braw = braw[braw.phase == "measured"]
    assert runs.checksum_ok.all() and braw.checksum_ok.all(), "a checksum did not match"
    return runs, tasks, status, braw, bstat


KEY = ["experiment", "n", "members", "threads_per_member"]


def summarise_dist(runs, tasks, status):
    rows = []
    for key, g in runs.groupby(KEY):
        exp, n, w, t = key
        tk = tasks[(tasks.experiment == exp) & (tasks.n == n) & (tasks.members == w)
                   & (tasks.threads_per_member == t)]
        per_run = []
        for run, tg in tk.groupby("run"):
            slow = tg.loc[tg.task_s.idxmax()]
            ex = float(g[g.run == run].execute_s.iloc[0])
            per_run.append({"fetch": slow.fetchA_s + slow.fetchB_s, "compute": slow.compute_s,
                            "store": slow.storeC_s, "dispatch": ex - slow.task_s,
                            "max_compute": tg.compute_s.max(),
                            "imbalance": tg.compute_s.max() / tg.compute_s.median(),
                            "max_fetchB": tg.fetchB_s.max()})
        pr = pd.DataFrame(per_run)
        st = status[(status.experiment == exp) & (status.n == n) & (status.members == w)
                    & (status.threads_per_member == t)].iloc[0]
        rows.append({
            "experiment": exp, "n": n, "members": w, "threads_per_member": t, "runs": len(g),
            "total_s": g.total_s.median(), "q1_s": q(g.total_s, 25), "q3_s": q(g.total_s, 75),
            "iqr_pct": 100 * (q(g.total_s, 75) - q(g.total_s, 25)) / g.total_s.median(),
            "distribute_s": g.distribute_s.median(), "execute_s": g.execute_s.median(),
            "collect_s": g.collect_s.median(),
            "slow_fetch_s": pr.fetch.median(), "slow_compute_s": pr.compute.median(),
            "slow_store_s": pr.store.median(), "dispatch_wait_s": pr.dispatch.median(),
            "max_compute_s": pr.max_compute.median(), "max_fetchB_s": pr.max_fetchB.median(),
            "compute_imbalance": pr.imbalance.median(),
            "mb_to_cluster": g.bytes_to_cluster.iloc[0] / MB,
            "mb_member_to_member": g.bytes_member_to_member.iloc[0] / MB,
            "mb_from_cluster": g.bytes_from_cluster.iloc[0] / MB,
            "driver_peak_mb": st.driver_peak_mb, "max_member_peak_mb": st.max_member_peak_mb,
            "sum_member_peak_mb": st.sum_member_peak_mb, "total_peak_mb": st.total_peak_mb,
            "cluster_formation_s": st.cluster_formation_s, "client_connect_s": st.client_connect_s,
            "status": st.status})
    return pd.DataFrame(rows)


def summarise_baselines(braw, bstat):
    rows = []
    for (m, p, n), g in braw.groupby(["method", "threads", "n"]):
        st = bstat[(bstat.method == m) & (bstat.threads == p) & (bstat.n == n)].iloc[0]
        rows.append({"method": m, "threads": p, "n": n, "runs": len(g),
                     "calls_per_run": int(g.calls.median()), "median_s": g.time_s.median(),
                     "q1_s": q(g.time_s, 25), "q3_s": q(g.time_s, 75),
                     "iqr_pct": 100 * (q(g.time_s, 75) - q(g.time_s, 25)) / g.time_s.median(),
                     "peak_memory_mb": st.peak_memory_mb, "status": st.status})
    order = {"basic": 0, "seq": 1, "par": 2}
    return pd.DataFrame(rows).sort_values(["n", "method", "threads"],
                                          key=lambda s: s.map(order) if s.name == "method" else s)


def bl(base, method, n, p=1, col="median_s"):
    r = base[(base.method == method) & (base.n == n) & (base.threads == p)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def main():
    runs, tasks, status, braw, bstat = load()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    d = summarise_dist(runs, tasks, status)
    base1 = summarise_baselines(braw, bstat)
    seq2 = summarise_baselines(*load_seq2())
    sessions = base1[base1.method == "seq"][["n", "median_s", "iqr_pct"]].merge(
        seq2[["n", "median_s", "iqr_pct"]], on="n", suffixes=("_session1", "_session2"))
    sessions["ratio_1_over_2"] = sessions.median_s_session1 / sessions.median_s_session2
    save(sessions, "T7_seq_sessions")
    print(sessions.to_markdown(index=False, floatfmt=".4g"))
    base = pd.concat([base1[base1.method != "seq"], seq2]).sort_values(["n", "method", "threads"])
    base["session"] = np.where(base.method == "seq", 2, 1)
    save(d, "T0_distributed_full")
    save(base, "T1_baselines")
    print(base.to_markdown(index=False, floatfmt=".4g"))

    # ---- strong scaling ------------------------------------------------------------
    s = d[d.experiment == "strong"].copy()
    s["T1_dist_s"] = s.n.map(s[s.members == 1].set_index("n").total_s)
    s["speedup_vs_dist1"] = s.T1_dist_s / s.total_s
    s["efficiency_vs_dist1"] = s.speedup_vs_dist1 / s.members
    s["seq_kernel_s"] = [bl(base, "seq", n) for n in s.n]
    s["par_kernel_p_eq_W_s"] = [bl(base, "par", n, w) for n, w in zip(s.n, s.members)]
    s["speedup_e2e_vs_seq"] = s.seq_kernel_s / s.total_s
    s["speedup_compute_vs_seq"] = s.seq_kernel_s / s.max_compute_s
    s["comm_overhead_fraction"] = 1 - s.max_compute_s / s.total_s
    t2 = s[["n", "members", "total_s", "iqr_pct", "distribute_s", "execute_s", "collect_s",
            "slow_fetch_s", "slow_compute_s", "slow_store_s", "dispatch_wait_s", "max_compute_s",
            "compute_imbalance", "speedup_vs_dist1", "efficiency_vs_dist1", "seq_kernel_s",
            "par_kernel_p_eq_W_s", "speedup_e2e_vs_seq", "speedup_compute_vs_seq",
            "comm_overhead_fraction"]]
    save(t2, "T2_strong_scaling")
    print(t2.to_markdown(index=False, floatfmt=".4g"))
    t3 = s[["n", "members", "mb_to_cluster", "mb_member_to_member", "mb_from_cluster",
            "max_fetchB_s", "driver_peak_mb", "max_member_peak_mb", "sum_member_peak_mb",
            "total_peak_mb", "cluster_formation_s", "client_connect_s", "status"]]
    save(t3, "T3_transfers_memory_setup")
    print(t3.to_markdown(index=False, floatfmt=".4g"))

    # ---- comparison with the baselines at the common sizes ---------------------------
    comp = []
    for n in STRONG_SIZES:
        pb = base[(base.method == "par") & (base.n == n)]
        best = pb.loc[pb.median_s.idxmin()]
        sd = s[s.n == n]
        bd = sd.loc[sd.total_s.idxmin()]
        comp.append({"n": n, "basic_kernel_s": bl(base, "basic", n), "seq_kernel_s": bl(base, "seq", n),
                     "par_best_kernel_s": best.median_s, "par_best_p": int(best.threads),
                     "dist_best_total_s": bd.total_s, "dist_best_W": int(bd.members),
                     "dist_best_max_compute_s": bd.max_compute_s,
                     "dist_W1_total_s": float(sd[sd.members == 1].total_s.iloc[0]),
                     "seq_peak_mb": bl(base, "seq", n, col="peak_memory_mb"),
                     "par_best_peak_mb": best.peak_memory_mb,
                     "dist_best_total_peak_mb": bd.total_peak_mb})
    comp = pd.DataFrame(comp)
    save(comp, "T4_comparison_baselines")
    print(comp.to_markdown(index=False, floatfmt=".4g"))

    # ---- weak scaling --------------------------------------------------------------------
    wk = d[d.experiment == "weak"].copy().sort_values("members")
    t1w = float(wk[wk.members == 1].total_s.iloc[0])
    c1w = float(wk[wk.members == 1].max_compute_s.iloc[0])
    wk["madds_per_member_e9"] = wk.n.astype(float) ** 3 / wk.members / 1e9
    wk["weak_efficiency_e2e"] = t1w / wk.total_s
    wk["weak_efficiency_compute"] = c1w / wk.max_compute_s
    wk["seq_kernel_s"] = [bl(base, "seq", n) for n in wk.n]
    t5 = wk[["members", "n", "madds_per_member_e9", "total_s", "iqr_pct", "distribute_s",
             "execute_s", "collect_s", "slow_fetch_s", "max_compute_s", "dispatch_wait_s",
             "weak_efficiency_e2e", "weak_efficiency_compute", "mb_to_cluster",
             "mb_member_to_member", "max_member_peak_mb", "total_peak_mb", "seq_kernel_s"]]
    save(t5, "T5_weak_scaling")
    print(t5.to_markdown(index=False, floatfmt=".4g"))

    # ---- workers vs processes ----------------------------------------------------------
    sp = d[d.experiment == "split"].copy().sort_values("members")
    sp["par6_kernel_s"] = bl(base, "par", SPLIT_N, 6)
    t6 = sp[["members", "threads_per_member", "total_s", "iqr_pct", "distribute_s", "execute_s",
             "collect_s", "slow_fetch_s", "max_compute_s", "dispatch_wait_s", "mb_member_to_member",
             "max_member_peak_mb", "sum_member_peak_mb", "total_peak_mb", "par6_kernel_s"]]
    save(t6, "T6_processes_vs_threads")
    print(t6.to_markdown(index=False, floatfmt=".4g"))

    figures(s, wk, sp, base)


def stacked(ax, df, xlabels, title):
    parts = [("distribute_s", "distribute (driver)"), ("slow_fetch_s", "fetch A + B (task)"),
             ("slow_compute_s", "compute (task)"), ("slow_store_s", "store C (task)"),
             ("dispatch_wait_s", "dispatch + wait"), ("collect_s", "collect (driver)")]
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    for i, (col, lab) in enumerate(parts):
        v = df[col].clip(lower=0).to_numpy()
        ax.bar(x, v, 0.62, bottom=bottom, color=C[i], label=lab, edgecolor="white", linewidth=1)
        bottom += v
    for xi, tot in zip(x, df.total_s):
        ax.text(xi, tot * 1.01, f"{tot:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x, xlabels)
    ax.set_title(title, fontsize=9.5)
    ax.set_ylabel("time per multiplication (s)")


def figures(s, wk, sp, base):
    # Fig 1: strong scaling runtime, one panel per n
    fig, axs = plt.subplots(2, 2, figsize=(8.2, 6.0))
    for ax, n in zip(axs.flat, STRONG_SIZES):
        g = s[s.n == n].sort_values("members")
        w = g.members.to_numpy()
        ax.errorbar(w, g.total_s, yerr=[g.total_s - g.q1_s, g.q3_s - g.total_s], color=C[0],
                    marker="o", capsize=2, label="distributed, end-to-end")
        ax.plot(w, g.max_compute_s, color=C[2], marker="s", label="distributed, compute only")
        par = [bl(base, "par", n, p) for p in w]
        ax.plot(w, par, color=C[1], marker="^", linestyle="--", label="parallel A3 (p = W), kernel")
        ax.axhline(bl(base, "seq", n), color=GRAY, linestyle=":", label="sequential A3, kernel")
        if not np.isnan(bl(base, "basic", n)):
            ax.axhline(bl(base, "basic", n), color=C[3], linestyle="-.", label="basic A1, kernel")
        ax.set_yscale("log")
        ax.set_xticks(MEMBER_COUNTS)
        ax.set_title(f"n = {n}", fontsize=9.5)
        ax.set_xlabel("members W (distributed) / threads p (parallel)")
        ax.set_ylabel("time (s, log scale)")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURES_DIR / "fig1_strong_runtime.png", dpi=200)
    plt.close(fig)

    # Fig 2: strong-scaling speedup (end-to-end, relative to the distributed W = 1)
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for i, n in enumerate(STRONG_SIZES):
        g = s[s.n == n].sort_values("members")
        axs[0].plot(g.members, g.speedup_vs_dist1, color=C[i], marker="o", label=f"n = {n}")
        par = [bl(base, "seq", n) / bl(base, "par", n, p) for p in g.members]
        axs[1].plot(g.members, par, color=C[i], marker="^", linestyle="--", label=f"n = {n}")
    for ax, title in zip(axs, ["(a) distributed: $T_{dist}(1) / T_{dist}(W)$, end-to-end",
                               "(b) parallel A3: $T_{seq} / T_p$, kernel only"]):
        ax.plot(MEMBER_COUNTS, MEMBER_COUNTS, color=GRAY, linestyle=":", linewidth=1.2, label="ideal")
        ax.set_xticks(MEMBER_COUNTS)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("members W" if ax is axs[0] else "threads p")
    axs[0].set_ylabel("speedup")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_strong_speedup.png", dpi=200)
    plt.close(fig)

    # Fig 3: breakdown of the end-to-end time, strong scaling, n = 1600 and n = 4000
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for ax, n in zip(axs, [1600, STRONG_SIZES[-1]]):
        g = s[s.n == n].sort_values("members")
        stacked(ax, g, [str(w) for w in g.members], f"n = {n}")
        ax.set_xlabel("members W")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURES_DIR / "fig3_strong_breakdown.png", dpi=200)
    plt.close(fig)

    # Fig 4: weak scaling breakdown + efficiency
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    stacked(axs[0], wk, [f"{w}\n(n={n})" for w, n in zip(wk.members, wk.n)],
            "time per multiplication, work per member constant")
    axs[0].set_xlabel("members W (matrix size)")
    axs[1].plot(wk.members, wk.weak_efficiency_e2e, color=C[0], marker="o", label="end-to-end")
    axs[1].plot(wk.members, wk.weak_efficiency_compute, color=C[2], marker="s", label="compute only")
    axs[1].axhline(1, color=GRAY, linestyle=":", linewidth=1.2, label="ideal")
    axs[1].set_ylim(0, 1.15)
    axs[1].set_xticks(MEMBER_COUNTS)
    axs[1].set_xlabel("members W")
    axs[1].set_ylabel("weak-scaling efficiency $T(1)/T(W)$")
    axs[1].set_title("efficiency", fontsize=9.5)
    axs[1].legend(fontsize=8)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURES_DIR / "fig4_weak_scaling.png", dpi=200)
    plt.close(fig)

    # Fig 5: memory, strong scaling at the largest n
    n = STRONG_SIZES[-1]
    g = s[s.n == n].sort_values("members")
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(g.members, g.total_peak_mb, color=C[0], marker="o", label="all processes (sum)")
    ax.plot(g.members, g.sum_member_peak_mb, color=C[2], marker="s", label="all members (sum)")
    ax.plot(g.members, g.max_member_peak_mb, color=C[1], marker="^", label="largest member")
    ax.plot(g.members, g.driver_peak_mb, color=C[3], marker="D", label="driver")
    ax.axhline(bl(base, "seq", n, col="peak_memory_mb"), color=GRAY, linestyle=":",
               label="sequential A3 process")
    ax.set_xticks(MEMBER_COUNTS)
    ax.set_xlabel("members W")
    ax.set_ylabel("peak working set (MB)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Peak memory, n = {n}", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_memory.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
