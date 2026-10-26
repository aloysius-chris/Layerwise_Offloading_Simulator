"""RRHOM (strategy 6) vs the CoEdge benchmark (strategy 7) only.

Runs the two methods for 30 seeds on the same two experiments as
run_multiseed.py (number of tasks; VMs per ECN), and writes every report for
this single comparison into results/multiseed_coedge/ - the thesis results in
results/multiseed/ are not touched.

Run from the project folder:
    python results/run_coedge_vs_rrhom.py              # 30 seeds, all CPU cores
    python results/run_coedge_vs_rrhom.py --jobs 1     # single process
    python results/run_coedge_vs_rrhom.py --seeds 5    # quick check
then the figures:
    python results/multiseed/plot_multiseed.py --dir results/multiseed_coedge --methods CoEdge RRHOM

Outputs (results/multiseed_coedge/):
  raw.csv       one row per (experiment, x, method, seed)
  summary.csv   mean, 95% CI half-width, n
  tests.csv     paired t / Wilcoxon, Holm-adjusted p (over the metrics of one cell), Cohen's d
  report.md     one table per metric: CoEdge, RRHOM, difference %, p_Holm, winner
  <exp>_<metric>_CoEdge_vs_RRHOM.png   bar charts (mean, 95% CI)
"""
import argparse
import csv
import math
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import settings                                                  # noqa: E402
from run_multiseed import (run_once, ci95, paired_test, holm,    # noqa: E402
                           draw, METRICS, NAMES)

A, B = 7, 6                                   # baseline CoEdge, proposed RRHOM
LOWER_BETTER = {"latency", "weighted_latency", "load_variance", "migrations"}


def _job(args):
    exp, xv, md, vm, s, seed = args
    return (exp, xv, s, seed), run_once(s, seed, md, vm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    seeds = list(range(1, args.seeds + 1))
    outdir = os.path.join(HERE, "multiseed_coedge")
    os.makedirs(outdir, exist_ok=True)

    experiments = {
        "tasks": ([(str(md * settings.ecn_count * settings.du_count), md, settings.vm_count)
                   for md in settings.task_list], "Number of tasks (5 VMs per ECN)"),
        "vms": ([(str(v), settings.md_count, v) for v in settings.vm_list], "VMs per ECN (300 tasks)"),
    }
    jobs = [(exp, xv, md, vm, s, seed)
            for exp, (points, _) in experiments.items()
            for xv, md, vm in points for s in (A, B) for seed in seeds]
    # longest runs first (RRHOM, many tasks) so the cores stay busy
    jobs.sort(key=lambda j: (j[4] != B, -(j[2] * j[3] if j[0] == "tasks" else j[2] * 5)))

    results = {}
    done = 0
    with Pool(max(1, args.jobs)) as pool:
        for key, r in pool.imap_unordered(_job, jobs):
            results[key] = r
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} runs done", flush=True)

    raw = {}
    with open(os.path.join(outdir, "raw.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["experiment", "x", "method", "seed"] + [m for m, _ in METRICS])
        for exp, (points, _) in experiments.items():
            for xv, _, _ in points:
                for s in (A, B):
                    for seed in seeds:
                        r = results[(exp, xv, s, seed)]
                        raw.setdefault((exp, xv, s), []).append(r)
                        wr.writerow([exp, xv, NAMES[s], seed] + [r[m] for m, _ in METRICS])

    cell = {k: {m: ci95([r[m] for r in v]) for m, _ in METRICS} for k, v in raw.items()}
    with open(os.path.join(outdir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["experiment", "x", "method", "metric", "mean", "ci95_half_width", "n"])
        for (exp, xv, s), mv in cell.items():
            for m, (mean, hw) in mv.items():
                wr.writerow([exp, xv, NAMES[s], m, mean, hw, len(raw[(exp, xv, s)])])

    # one test per (experiment, x, metric); Holm over the metrics of the same cell
    rows, sig = [], {}
    for exp, (points, _) in experiments.items():
        for xv, _, _ in points:
            cell_rows = []
            for m, _ in METRICS:
                xa = [r[m] for r in raw[(exp, xv, A)]]
                xb = [r[m] for r in raw[(exp, xv, B)]]
                p, d, test = paired_test(xa, xb)
                cell_rows.append([exp, xv, m, NAMES[A], NAMES[B], test, p, d])
            adj = holm([1.0 if math.isnan(r[6]) else r[6] for r in cell_rows])
            for r, pa in zip(cell_rows, adj):
                r.append(pa)
                sig[(exp, xv, r[2], A, B)] = pa < 0.05
            rows.extend(cell_rows)
    with open(os.path.join(outdir, "tests.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["experiment", "x", "metric", "method_a", "method_b", "test", "p", "cohen_d", "p_holm"])
        wr.writerows(rows)

    for exp, (points, xname) in experiments.items():
        xl = [xv for xv, _, _ in points]
        for m, ylabel in METRICS:
            draw(exp, xl, xname, m, ylabel, A, B, cell, sig, outdir)

    # markdown report
    test_of = {(r[0], r[1], r[2]): r for r in rows}
    lines = ["# RRHOM vs CoEdge (30 seeds, mean and 95% CI)", "",
             "Difference = (RRHOM - CoEdge) / CoEdge.  Winner is decided by the mean; "
             "'n.s.' = p_Holm >= 0.05.", ""]
    for m, ylabel in METRICS:
        lines += [f"## {ylabel}  ({'lower' if m in LOWER_BETTER else 'higher'} is better)", "",
                  "| Experiment | x | CoEdge | RRHOM | Difference | p_Holm | Winner |",
                  "|---|---|---|---|---|---|---|"]
        for exp, (points, xname) in experiments.items():
            for xv, _, _ in points:
                ca, ha = cell[(exp, xv, A)][m]
                cb, hb = cell[(exp, xv, B)][m]
                diff_txt = f"{(cb - ca) / ca * 100:+.1f}%" if ca else "—"
                pa = test_of[(exp, xv, m)][8]
                better_b = cb < ca if m in LOWER_BETTER else cb > ca
                win = ("n.s." if pa >= 0.05 else ("RRHOM" if better_b else "CoEdge")) if ca != cb else "tie"
                lines.append(f"| {exp} | {xv} | {ca:.4g} ± {ha:.2g} | {cb:.4g} ± {hb:.2g} | "
                             f"{diff_txt} | {pa:.2e} | {win} |")
        lines.append("")
    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("written to", outdir)


if __name__ == "__main__":
    main()
