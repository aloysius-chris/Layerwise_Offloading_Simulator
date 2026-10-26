"""Figures for RRHOM vs CoEdge ONLY - no command-line options needed.

Reads  results/multiseed_coedge/summary.csv and raw.csv
       (produced by results/run_coedge_vs_rrhom.py)
Writes results/multiseed_coedge/figures/
         sequential/RRHOM_vs_CoEdge/   bar_*.png, line_*.png
         supplementary/                box, violin, CDF, heatmap, radar, effect size, scatter
The table report is results/multiseed_coedge/report.md

Run it from PyCharm (right-click > Run) or:  python results/plot_coedge_vs_rrhom.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "multiseed"))
import plot_multiseed  # noqa: E402

data_dir = os.path.join(HERE, "multiseed_coedge")
if not os.path.exists(os.path.join(data_dir, "summary.csv")):
    sys.exit("results/multiseed_coedge/summary.csv not found - run results/run_coedge_vs_rrhom.py first")

sys.argv = [sys.argv[0], "--dir", data_dir, "--methods", "CoEdge", "RRHOM"]
plot_multiseed.main()
print("\nCoEdge vs RRHOM figures:", os.path.join(data_dir, "figures"))
print("Table report:           ", os.path.join(data_dir, "report.md"))
