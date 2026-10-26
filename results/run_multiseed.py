"""Multi-seed experiment runner with 95% confidence intervals (Section 3.5.2).

Place this file in the results/ folder next to settings.py and draw_results.py
and run it from the project folder:

    python results/run_multiseed.py            # 30 seeds (Section 3.5.2)
    python results/run_multiseed.py --seeds 10 # quicker check

CoEdge benchmark (strategy 7, Hu, Sun and Ren 2020)
    python results/run_multiseed.py --only 7 --reuse
        runs ONLY CoEdge for the 30 seeds and re-uses the HOM/MLHOM/RRHOM/LFHOM
        rows already in multiseed/raw.csv (the simulator is deterministic for a
        given seed, so those rows are unchanged); summary, tests and figures are
        then rebuilt for all five methods.  Without --reuse every method is run.
    CoEdge is compared with RRHOM (and with LFHOM for fairness).  These
    benchmark comparisons form their own Holm family, so the p-values of the
    Chapter 4-6 comparisons are exactly as before.

The RRHOM dispatch (strategy 6 -> the task's own relocation handler), the inter-ECN rate gamma/h and
the cloud as a relocation candidate were applied to simulator.py on 24 September 2026 (backups in
results/_backup_before_RRHOM_rename_2026-09-24/).

Outputs (in results/multiseed/):
  * raw.csv                one row per (experiment, x value, method, seed)
  * summary.csv            mean, 95% CI half-width, n per cell
  * tests.csv              paired t-test / Wilcoxon, Holm-adjusted p, Cohen's d
  * one PNG per metric and comparison, bars = mean, error bars = 95% CI,
    x axis in VMs per ECN (1..5), no title (the caption is in the thesis).
"""
import argparse
import csv
import io
import contextlib
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import settings                                  # noqa: E402
from simulator import Simulation                 # noqa: E402
from system_model.base_task import TaskLogType   # noqa: E402

import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402

try:
    from scipy import stats
except ImportError:                               # tests are skipped without SciPy
    stats = None

NAMES = {3: "HOM", 4: "MLHOM", 5: "LFHOM", 6: "RRHOM", 7: "CoEdge"}
COMPARISONS = [(3, 4), (4, 6), (6, 5)]           # Ch.4, Ch.5, Ch.6
BENCH_COMPARISONS = [(7, 6), (7, 5)]             # CoEdge benchmark vs RRHOM and vs LFHOM
FAMILY = {pair: "chapters" for pair in COMPARISONS}
FAMILY.update({pair: "coedge" for pair in BENCH_COMPARISONS})
METRICS = [("latency",          "Mean latency (s)"),
           ("weighted_latency", "Weighted mean latency (s)"),
           ("jain",             "Jain's fairness index"),
           ("weighted_jfi",     "Weighted Jain's fairness index"),
           ("utilization",      "Resource utilization"),
           ("load_variance",    "Variance of VM load (FLOP$^2$)"),
           ("migrations",       "Number of relocations")]


def run_once(strategy, seed, md_count, vm_count):
    sim = Simulation(layer_list=settings.algo.layer_list,
                     vm_flops=settings.vm_flops, cloud_flops=settings.cloud_flops,
                     vm_count=vm_count, ecn_count=settings.ecn_count,
                     du_count=settings.du_count, md_count=md_count)
    with contextlib.redirect_stdout(io.StringIO()):
        sim.start(offloading_strategy=strategy, random_state=seed)
    migrations = sum(1 for t in sim.task_list for log in t.log_list
                     if isinstance(log, dict)
                     and log.get("log_type") == TaskLogType.RELOCATE_TO_COMPUTATION_UNIT)
    return {"latency":          sim.get_latency(),
            "weighted_latency": sim.get_weighted_latency(),
            "jain":             sim.get_jains_fairness_index(),
            "weighted_jfi":     sim.get_weighted_jains_fairness_index(),
            "utilization":      sim.get_resources_utilization(),
            "load_variance":    sim.get_load_balancing(),
            "migrations":       migrations}


def ci95(xs):
    n = len(xs)
    m = st.mean(xs)
    if n < 2:
        return m, 0.0
    t = stats.t.ppf(0.975, n - 1) if stats else 2.045   # 2.045 = t(0.975, 29)
    return m, t * st.stdev(xs) / math.sqrt(n)


def paired_test(a, b):
    d = [x - y for x, y in zip(a, b)]
    sd = st.stdev(d) if len(d) > 1 else 0.0
    cohen_d = (st.mean(d) / sd) if sd > 0 else 0.0
    if stats is None or sd == 0:
        return float("nan"), cohen_d, "none"
    normal = stats.shapiro(d).pvalue >= 0.05 if len(d) >= 3 else True
    if normal:
        return stats.ttest_rel(a, b).pvalue, cohen_d, "paired t"
    return stats.wilcoxon(a, b).pvalue, cohen_d, "Wilcoxon"


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def draw(exp, xlabels, xname, metric, ylabel, a, b, cell, sig, outdir):
    import numpy as np
    x = np.arange(len(xlabels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for k, (strategy, colour, off) in enumerate([(a, "0.7", -w / 2), (b, "blue", w / 2)]):
        means = [cell[(exp, xv, strategy)][metric][0] for xv in xlabels]
        errs = [cell[(exp, xv, strategy)][metric][1] for xv in xlabels]
        bars = ax.bar(x + off, means, w, yerr=errs, capsize=3, color=colour,
                      edgecolor="black", linewidth=0.6, label=NAMES[strategy])
        if k == 1:   # proposed method: dotted border where not significant
            for bar, xv in zip(bars, xlabels):
                if not sig.get((exp, xv, metric, a, b), True):
                    bar.set_linestyle(":")
                    bar.set_linewidth(1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel(xname)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{exp}_{metric}_{NAMES[a]}_vs_{NAMES[b]}.png"), dpi=200)
    plt.close(fig)


def _truncate(path):
    """Overwrite file with zero bytes so no stale content or NUL padding survives."""
    with open(path, "w", encoding="utf-8") as f:
        f.truncate(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--quick", action="store_true", help="only 100 tasks / 1 VM per ECN (smoke test)")
    ap.add_argument("--only", type=int, nargs="+", default=None,
                    help="run only these strategies, e.g. --only 7 (CoEdge)")
    ap.add_argument("--reuse", action="store_true",
                    help="take the rows of the strategies that are not run from the existing raw.csv")
    args = ap.parse_args()
    seeds = list(range(1, args.seeds + 1))
    outdir = os.path.join(HERE, "multiseed")
    os.makedirs(outdir, exist_ok=True)

    all_pairs = COMPARISONS + BENCH_COMPARISONS
    strategies = sorted({s for pair in all_pairs for s in pair})
    to_run = [s for s in strategies if args.only is None or s in args.only]

    # rows of the strategies that are not run again (--reuse)
    reused = []
    if args.reuse:
        old = os.path.join(outdir, "raw.csv")
        if os.path.exists(old) and os.path.getsize(old) > 0:
            with open(old, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    code = [k for k, v in NAMES.items() if v == row["method"]]
                    if code and code[0] not in to_run and int(row["seed"]) in seeds:
                        reused.append(row)
        print(f"re-using {len(reused)} rows from raw.csv; running {[NAMES[s] for s in to_run]}")

    # Clear output CSVs before any simulation runs so no stale data or NUL bytes remain.
    for fname in ("raw.csv", "summary.csv", "tests.csv"):
        _truncate(os.path.join(outdir, fname))

    experiments = {
        # name: (list of (x label, md_count, vm_count), x-axis name)
        "tasks": ([(str(md * settings.ecn_count * settings.du_count), md, settings.vm_count)
                   for md in (settings.task_list[:1] if args.quick else settings.task_list)],
                  "Number of tasks (5 VMs per ECN)"),
        "vms": ([(str(v), settings.md_count, v) for v in (settings.vm_list[:1] if args.quick else settings.vm_list)],
                "VMs per ECN (300 tasks)"),
    }
    raw = {}
    with open(os.path.join(outdir, "raw.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["experiment", "x", "method", "seed"] + [m for m, _ in METRICS])
        valid_x = {(exp, xv) for exp, (points, _) in experiments.items() for xv, _, _ in points}
        for row in reused:
            if (row["experiment"], row["x"]) not in valid_x:
                continue
            code = [k for k, v in NAMES.items() if v == row["method"]][0]
            r = {m: float(row[m]) for m, _ in METRICS}
            raw.setdefault((row["experiment"], row["x"], code), []).append(r)
            wr.writerow([row["experiment"], row["x"], row["method"], row["seed"]] + [r[m] for m, _ in METRICS])
        for exp, (points, _) in experiments.items():
            for xv, md, vm in points:
                for s in to_run:
                    for seed in seeds:
                        r = run_once(s, seed, md, vm)
                        raw.setdefault((exp, xv, s), []).append(r)
                        wr.writerow([exp, xv, NAMES[s], seed] + [r[m] for m, _ in METRICS])
                        f.flush()
                    print(exp, xv, NAMES[s], "done", flush=True)

    cell = {k: {m: ci95([r[m] for r in v]) for m, _ in METRICS} for k, v in raw.items()}
    with open(os.path.join(outdir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["experiment", "x", "method", "metric", "mean", "ci95_half_width", "n"])
        for (exp, xv, s), mv in cell.items():
            for m, (mean, hw) in mv.items():
                wr.writerow([exp, xv, NAMES[s], m, mean, hw, len(raw[(exp, xv, s)])])

    sig = {}
    rows = []
    for exp, (points, _) in experiments.items():
        for xv, _, _ in points:
            for m, _ in METRICS:
                for a, b in all_pairs:
                    if (exp, xv, a) not in raw or (exp, xv, b) not in raw:
                        continue
                    xa = [r[m] for r in raw[(exp, xv, a)]]
                    xb = [r[m] for r in raw[(exp, xv, b)]]
                    p, d, test = paired_test(xa, xb)
                    rows.append([exp, xv, m, NAMES[a], NAMES[b], test, p, d, FAMILY[(a, b)]])
    # Holm correction over the comparisons made at the same configuration and metric,
    # separately for the chapter comparisons and for the CoEdge benchmark comparisons
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r[0], r[1], r[2], r[8]), []).append(i)
    for idx in groups.values():
        ps = [rows[i][6] for i in idx]
        adj = holm([1.0 if math.isnan(p) else p for p in ps])
        for i, pa in zip(idx, adj):
            rows[i].insert(8, pa)
            exp, xv, m, na, nb = rows[i][:5]
            a = [k for k, v in NAMES.items() if v == na][0]
            b = [k for k, v in NAMES.items() if v == nb][0]
            sig[(exp, xv, m, a, b)] = pa < 0.05
    with open(os.path.join(outdir, "tests.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["experiment", "x", "metric", "method_a", "method_b", "test", "p", "cohen_d", "p_holm", "holm_family"])
        wr.writerows(rows)

    for exp, (points, xname) in experiments.items():
        xl = [xv for xv, _, _ in points]
        for m, ylabel in METRICS:
            for a, b in all_pairs:
                if all((exp, xv, s) in cell for xv in xl for s in (a, b)):
                    draw(exp, xl, xname, m, ylabel, a, b, cell, sig, outdir)
    print("written to", outdir)


if __name__ == "__main__":
    main()
