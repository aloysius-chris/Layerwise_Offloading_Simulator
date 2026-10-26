# CoEdge benchmark (strategy 7)

Hu, L., Sun, G. and Ren, Y. (2020) "CoEdge: Exploiting the Edge-Cloud Collaboration
for Faster Deep Learning", IEEE Access 8: 100533-41.

## Files added / changed (backups in results/_backup_before_CoEdge_2026-09-30/)

| File | Change |
|---|---|
| system_model/coedge_task.py | NEW - CoEdge task: Algorithm 1 (greedy growth of Src) + Eq. (4) latency-minimum partition |
| simulator.py | strategy 7 = CoEdge (import, name, task creation, event dispatch) |
| results/run_multiseed.py | method 7 "CoEdge"; comparisons CoEdge vs RRHOM and CoEdge vs LFHOM in their own Holm family; new options --only and --reuse |
| results/multiseed/plot_multiseed.py | CoEdge added automatically when present (pairs RRHOM vs CoEdge, LFHOM vs CoEdge, group B3, supplementary figures) |
| results/all.py | strategy 7 added with label CoEdge |
| results/coedge.py | NEW - single-seed CoEdge vs RRHOM figures (like rrhom.py) |

## RRHOM vs CoEdge only (recommended for the benchmark chapter)

    python results/run_coedge_vs_rrhom.py
    python results/multiseed/plot_multiseed.py --dir results/multiseed_coedge --methods CoEdge RRHOM

Runs ONLY strategies 6 and 7 (30 seeds, all CPU cores) and writes everything to
results/multiseed_coedge/ (raw.csv, summary.csv, tests.csv, report.md, bar charts,
figures/). results/multiseed/ (the thesis results) is not touched.

## How to run all methods (from the project folder)

1. Add CoEdge to the existing 30-seed results (fast, about 15 min; the other methods are re-used):

       python results/run_multiseed.py --only 7 --reuse

   or re-run everything from scratch (slow, all five methods):

       python results/run_multiseed.py

2. Rebuild the thesis figures:

       python results/multiseed/plot_multiseed.py

   New folders: figures/sequential/bench_RRHOM_vs_CoEdge, figures/sequential/bench_LFHOM_vs_CoEdge,
   figures/groups/b3_proposed_vs_CoEdge; CoEdge also appears in the supplementary figures.

3. Optional single-seed plots:  python results/coedge.py   /   python results/all.py

Outputs in results/multiseed/: raw.csv and summary.csv now contain method "CoEdge";
tests.csv has a new column holm_family ("chapters" for Ch.4-6, "coedge" for the benchmark).
