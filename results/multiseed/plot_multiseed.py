"""
plot_multiseed.py  –  Publication-quality figures from summary.csv / raw.csv.

Figure groups
─────────────
 A. Sequential pairwise comparisons  (contribution chain)
       Ch.4  MLHOM  vs  HOM
       Ch.5  RRHOM  vs  MLHOM
       Ch.6  LFHOM  vs  RRHOM
    One bar-chart + one line-trend figure per (pair × experiment × metric).

 B. Group comparisons
       B1. Three contributions together: MLHOM, RRHOM, LFHOM
       B2. All four methods: HOM, MLHOM, RRHOM, LFHOM
    Grouped bar charts + line trends.

 C. Supplementary  (all 4 methods)
       Heatmap, radar/spider, CDF, pairwise effect-size, box-plots, violins

Metric groups
─────────────
 primary          : Latency · Weighted JFI · Utilization · Load Variance
 weighted_latency : Weighted Latency   (separate)

Significance filter
───────────────────
 Each figure is only written if at least one (experiment, x-value) cell shows
 a statistically significant pairwise difference  (p < 0.05, paired t / Wilcoxon).

CoEdge benchmark
────────────────
 When summary.csv contains the CoEdge benchmark (strategy 7), it is added
 automatically: sequential pairs "RRHOM vs CoEdge" and "LFHOM vs CoEdge",
 group B3 (CoEdge + the three contributions), and all supplementary figures.

Run:
    python results/multiseed/plot_multiseed.py
    python results/multiseed/plot_multiseed.py --dir results/multiseed_coedge --methods CoEdge RRHOM
        (only the RRHOM vs CoEdge comparison, figures in results/multiseed_coedge/figures)
"""

import csv
import itertools
import math
import os
import statistics as st

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── paths ─────────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
SUMMARY_CSV = os.path.join(HERE, "summary.csv")
RAW_CSV     = os.path.join(HERE, "raw.csv")
OUT_DIR     = os.path.join(HERE, "figures")

# ── cosmetics ─────────────────────────────────────────────────────────────────
METHODS = ["HOM", "MLHOM", "RRHOM", "LFHOM"]   # contribution order

METHOD_COLORS = {
    "HOM":   "#888888",
    "MLHOM": "#2166AC",
    "RRHOM": "#4DAC26",
    "LFHOM": "#D6604D",
    "CoEdge": "#7B3294",      # external benchmark (Hu, Sun and Ren 2020)
}

# For each proposed method, which method is its direct benchmark
BENCHMARKS = {"MLHOM": "HOM", "RRHOM": "MLHOM", "LFHOM": "RRHOM", "HOM": None}

# Cumulative method subsets for progressive heatmaps
HEATMAP_SUBSETS = [
    (["HOM", "MLHOM"],              "1_HOM_MLHOM"),
    (["HOM", "MLHOM", "RRHOM"],     "2_add_RRHOM"),
    (["HOM", "MLHOM", "RRHOM", "LFHOM"], "3_all"),
]

EXPERIMENTS = {
    "tasks": "Number of tasks",
    "vms":   "VMs per ECN",
}

# Sequential contribution pairs:  (baseline, proposed, output_folder)
SEQ_PAIRS = [
    ("HOM",   "MLHOM", "ch4_MLHOM_vs_HOM"),
    ("MLHOM", "RRHOM", "ch5_RRHOM_vs_MLHOM"),
    ("RRHOM", "LFHOM", "ch6_LFHOM_vs_RRHOM"),
]

# Method groups for combined comparisons
METHOD_GROUPS = [
    (["MLHOM", "RRHOM", "LFHOM"], "b1_three_contributions"),
    (["HOM",   "MLHOM", "RRHOM", "LFHOM"], "b2_all_four"),
]

# Metric groups: (key, y-axis label)
METRIC_GROUPS = {
    "primary": [
        ("latency",       "Mean latency (s)"),
        ("weighted_jfi",  "Weighted Jain's fairness index"),
        ("utilization",   "Resource utilization"),
        ("load_variance", "Variance of VM load"),
    ],
    "weighted_latency": [
        ("weighted_latency", "Weighted mean latency (s)"),
    ],
}
# All metrics used for raw loading and supplementary figures
ALL_METRICS_KEYS = [
    "latency", "weighted_latency", "jain", "weighted_jfi",
    "utilization", "load_variance", "migrations",
]
# Flatten primary + weighted into one list for most iteration
PLOT_METRICS = METRIC_GROUPS["primary"] + METRIC_GROUPS["weighted_latency"]

LOWER_BETTER = {"latency", "weighted_latency", "load_variance"}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# ── helpers: sub-directories ──────────────────────────────────────────────────

def _outdir(*parts):
    """Create and return a subdirectory under OUT_DIR."""
    d = os.path.join(OUT_DIR, *parts)
    os.makedirs(d, exist_ok=True)
    return d


# ── data loading ──────────────────────────────────────────────────────────────

def load_summary():
    """Return {(exp, x, method, metric): (mean, ci_hw, n)}."""
    data = {}
    with open(SUMMARY_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["experiment"], row["x"], row["method"], row["metric"])
            data[key] = (float(row["mean"]), float(row["ci95_half_width"]), int(row["n"]))
    return data


def load_raw():
    """Return {(exp, x, method): [dict metric->float]}."""
    data = {}
    with open(RAW_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        available = set(reader.fieldnames or [])
        for row in reader:
            key = (row["experiment"], row["x"], row["method"])
            entry = {}
            for m in ALL_METRICS_KEYS:
                if m in available and row[m] not in ("", None):
                    try:
                        entry[m] = float(row[m])
                    except ValueError:
                        pass
            data.setdefault(key, []).append(entry)
    return data


def xs_for(summary, exp):
    """Sorted x-value strings for an experiment."""
    return sorted({k[1] for k in summary if k[0] == exp}, key=lambda v: int(v))


# ── statistics helpers ────────────────────────────────────────────────────────

def paired_p(a, b):
    if not HAS_SCIPY or len(a) < 3:
        return float("nan")
    diff = [x - y for x, y in zip(a, b)]
    sd = st.stdev(diff) if len(diff) > 1 else 0.0
    if sd == 0:
        return float("nan")
    normal = scipy_stats.shapiro(diff).pvalue >= 0.05 if len(diff) >= 3 else True
    if normal:
        return float(scipy_stats.ttest_rel(a, b).pvalue)
    return float(scipy_stats.wilcoxon(a, b).pvalue)


def cohens_d_paired(a, b):
    diff = [x - y for x, y in zip(a, b)]
    sd = st.stdev(diff) if len(diff) > 1 else 0.0
    return st.mean(diff) / sd if sd > 0 else float("nan")


def sig_stars(p):
    if math.isnan(p):   return ""
    if p < 0.001:       return "***"
    if p < 0.01:        return "**"
    if p < 0.05:        return "*"
    return ""


def is_significant_pair(raw, exp, metric, ma, mb, alpha=0.05):
    """True if the pair (ma, mb) shows p < alpha at ANY x-value."""
    xs = sorted({k[1] for k in raw if k[0] == exp}, key=lambda v: int(v))
    for xv in xs:
        a = [r[metric] for r in raw.get((exp, xv, ma), []) if metric in r]
        b = [r[metric] for r in raw.get((exp, xv, mb), []) if metric in r]
        if a and b:
            p = paired_p(a, b)
            if not math.isnan(p) and p < alpha:
                return True
    return False


def is_significant_group(raw, exp, metric, methods, alpha=0.05):
    """True if ANY pair in methods is significant at any x-value."""
    for ma, mb in itertools.combinations(methods, 2):
        if is_significant_pair(raw, exp, metric, ma, mb, alpha):
            return True
    return False


# ── shared drawing utility ────────────────────────────────────────────────────

def _sig_bracket(ax, xi, x_pos_a, x_pos_b, stars, y_top, step):
    """Draw a significance bracket at height y_top."""
    ax.annotate("", xy=(x_pos_b, y_top), xytext=(x_pos_a, y_top),
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
    ax.text((x_pos_a + x_pos_b) / 2, y_top + step * 0.1,
            stars, ha="center", va="bottom", fontsize=7)


# ══════════════════════════════════════════════════════════════════════════════
# A.  Sequential pairwise comparison figures
# ══════════════════════════════════════════════════════════════════════════════

def _seq_bar_one(summary, raw, ma, mb, exp, xname, metric, ylabel, outdir):
    """Two-bar grouped chart for a single (pair, experiment, metric).
    The proposed method's bar (mb) gets a thick border when it beats the baseline (ma).
    """
    if not is_significant_pair(raw, exp, metric, ma, mb):
        return False

    xs = xs_for(summary, exp)
    x  = np.arange(len(xs))
    w  = 0.32
    fig, ax = plt.subplots(figsize=(6.4, 3.8))

    for i, m in enumerate([ma, mb]):
        means = [summary.get((exp, xv, m, metric), (0, 0, 0))[0] for xv in xs]
        errs  = [summary.get((exp, xv, m, metric), (0, 0, 0))[1] for xv in xs]
        off   = (i - 0.5) * w

        # thick border when the proposed method beats the baseline at this x-value
        base_means = [summary.get((exp, xv, ma, metric), (0, 0, 0))[0] for xv in xs]
        linewidths = []
        for xv_idx, (mv, bv) in enumerate(zip(means, base_means)):
            if m == mb:
                beats = mv < bv if metric in LOWER_BETTER else mv > bv
                linewidths.append(2.2 if beats else 0.6)
            else:
                linewidths.append(0.6)

        bars = ax.bar(x + off, means, w, yerr=errs, capsize=3,
                      color=METHOD_COLORS[m],
                      edgecolor="black", label=m,
                      error_kw={"elinewidth": 0.8, "ecolor": "black"})
        # set linewidth per bar AFTER creation so it isn't overridden
        for bar, lw in zip(bars.patches, linewidths):
            bar.set_linewidth(lw)

    ax.set_xticks(x)
    ax.set_xticklabels(xs)
    ax.set_xlabel(xname)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()

    fname = os.path.join(outdir, f"bar_{exp}_{metric}.png")
    fig.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  {fname}")
    return True


def _seq_line_one(summary, raw, ma, mb, exp, xname, metric, ylabel, outdir):
    """Two-line trend chart with CI shading for a single (pair, experiment, metric)."""
    if not is_significant_pair(raw, exp, metric, ma, mb):
        return False

    xs = xs_for(summary, exp)
    xi = np.arange(len(xs))
    fig, ax = plt.subplots(figsize=(6, 3.8))

    for m in [ma, mb]:
        means = np.array([summary.get((exp, xv, m, metric), (0, 0, 0))[0] for xv in xs])
        errs  = np.array([summary.get((exp, xv, m, metric), (0, 0, 0))[1] for xv in xs])
        ax.plot(xi, means, marker="o", color=METHOD_COLORS[m],
                label=m, linewidth=1.8, markersize=5)
        ax.fill_between(xi, means - errs, means + errs,
                        alpha=0.15, color=METHOD_COLORS[m])

    ax.set_xticks(xi)
    ax.set_xticklabels(xs)
    ax.set_xlabel(xname)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()

    fname = os.path.join(outdir, f"line_{exp}_{metric}.png")
    fig.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  {fname}")
    return True


def fig_sequential(summary, raw):
    """Group A: one bar + one line figure per (pair, experiment, metric)."""
    total = 0
    for ma, mb, folder in SEQ_PAIRS:
        pair_label = f"{mb} vs {ma}"
        print(f"\n  -- Sequential: {pair_label} --")
        for exp, xname in EXPERIMENTS.items():
            for mgroup, metrics in METRIC_GROUPS.items():
                for metric, ylabel in metrics:
                    d = _outdir("sequential", folder)
                    n  = _seq_bar_one(summary, raw, ma, mb, exp, xname,
                                      metric, ylabel, d)
                    n += _seq_line_one(summary, raw, ma, mb, exp, xname,
                                       metric, ylabel, d)
                    total += n
    return total


# ══════════════════════════════════════════════════════════════════════════════
# B.  Group comparison figures
# ══════════════════════════════════════════════════════════════════════════════

def _group_bars_one(summary, raw, methods, exp, xname, metric, ylabel, outdir):
    """Grouped bar chart for a list of methods — skipped if not significant.
    Each bar gets a thick border when the method beats its direct benchmark.
    """
    if not is_significant_group(raw, exp, metric, methods):
        return False

    xs    = xs_for(summary, exp)
    x     = np.arange(len(xs))
    n_m   = len(methods)
    total_w = 0.72
    w = total_w / n_m
    fig, ax = plt.subplots(figsize=(7, 4))

    for i, m in enumerate(methods):
        means = [summary.get((exp, xv, m, metric), (0, 0, 0))[0] for xv in xs]
        errs  = [summary.get((exp, xv, m, metric), (0, 0, 0))[1] for xv in xs]
        off   = (i - (n_m - 1) / 2.0) * w

        bench = BENCHMARKS.get(m)
        bench_means = (
            [summary.get((exp, xv, bench, metric), (0, 0, 0))[0] for xv in xs]
            if bench and bench in methods else None
        )

        bars = ax.bar(x + off, means, w, yerr=errs, capsize=3,
                      color=METHOD_COLORS[m],
                      edgecolor="black", label=m,
                      error_kw={"elinewidth": 0.8, "ecolor": "black"})

        if bench_means is not None:
            for bar, mv, bv in zip(bars.patches, means, bench_means):
                beats = mv < bv if metric in LOWER_BETTER else mv > bv
                bar.set_linewidth(2.2 if beats else 0.6)
        else:
            for bar in bars.patches:
                bar.set_linewidth(0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(xs)
    ax.set_xlabel(xname)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()

    fname = os.path.join(outdir, f"bar_{exp}_{metric}.png")
    fig.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  {fname}")
    return True


def _group_lines_one(summary, raw, methods, exp, xname, metric, ylabel, outdir):
    """Line-trend chart for a group of methods — skipped if not significant."""
    if not is_significant_group(raw, exp, metric, methods):
        return False

    xs  = xs_for(summary, exp)
    xi  = np.arange(len(xs))
    fig, ax = plt.subplots(figsize=(6, 3.8))

    for m in methods:
        means = np.array([summary.get((exp, xv, m, metric), (0, 0, 0))[0] for xv in xs])
        errs  = np.array([summary.get((exp, xv, m, metric), (0, 0, 0))[1] for xv in xs])
        ax.plot(xi, means, marker="o", color=METHOD_COLORS[m],
                label=m, linewidth=1.8, markersize=5)
        ax.fill_between(xi, means - errs, means + errs,
                        alpha=0.15, color=METHOD_COLORS[m])

    ax.set_xticks(xi)
    ax.set_xticklabels(xs)
    ax.set_xlabel(xname)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()

    fname = os.path.join(outdir, f"line_{exp}_{metric}.png")
    fig.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  {fname}")
    return True


def fig_groups(summary, raw):
    """Group B: bar + line figures for each method group × metric."""
    total = 0
    for methods, folder in METHOD_GROUPS:
        label = " + ".join(methods)
        print(f"\n  -- Group: {label} --")
        for exp, xname in EXPERIMENTS.items():
            for mgroup, metrics in METRIC_GROUPS.items():
                for metric, ylabel in metrics:
                    d = _outdir("groups", folder)
                    total += _group_bars_one(summary, raw, methods, exp,
                                             xname, metric, ylabel, d)
                    total += _group_lines_one(summary, raw, methods, exp,
                                              xname, metric, ylabel, d)
    return total


# ══════════════════════════════════════════════════════════════════════════════
# C.  Supplementary figures  (all 4 methods, PLOT_METRICS only)
# ══════════════════════════════════════════════════════════════════════════════

def _supp_dir():
    return _outdir("supplementary")


# ── C1. Box plots ─────────────────────────────────────────────────────────────

def fig_boxplots(raw):
    total = 0
    for exp, xname in EXPERIMENTS.items():
        xs = sorted({k[1] for k in raw if k[0] == exp}, key=lambda v: int(v))
        for metric, ylabel in PLOT_METRICS:
            if not is_significant_group(raw, exp, metric, METHODS):
                continue
            ncols = len(xs)
            fig, axes = plt.subplots(1, ncols, figsize=(2.6 * ncols, 4), sharey=True)
            if ncols == 1:
                axes = [axes]
            for ax, xv in zip(axes, xs):
                data = [[r[metric] for r in raw.get((exp, xv, m), []) if metric in r]
                        for m in METHODS]
                bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                                medianprops={"color": "black", "linewidth": 1.5},
                                flierprops={"markersize": 3, "alpha": 0.5},
                                whiskerprops={"linewidth": 0.8},
                                capprops={"linewidth": 0.8})
                for patch, m in zip(bp["boxes"], METHODS):
                    patch.set_facecolor(METHOD_COLORS[m])
                    patch.set_alpha(0.82)
                ax.set_xticks(range(1, len(METHODS) + 1))
                ax.set_xticklabels(METHODS, rotation=35, ha="right", fontsize=7)
                ax.set_title(f"x={xv}", fontsize=9)
            axes[0].set_ylabel(ylabel)
            fig.tight_layout()
            fname = os.path.join(_supp_dir(), f"box_{exp}_{metric}.png")
            fig.savefig(fname, bbox_inches="tight", dpi=200)
            plt.close(fig)
            print(f"  {fname}")
            total += 1
    return total


# ── C2. Violin plots ──────────────────────────────────────────────────────────

def fig_violins(raw):
    total = 0
    for exp, xname in EXPERIMENTS.items():
        xs = sorted({k[1] for k in raw if k[0] == exp}, key=lambda v: int(v))
        for metric, ylabel in PLOT_METRICS:
            if not is_significant_group(raw, exp, metric, METHODS):
                continue
            ncols = len(xs)
            fig, axes = plt.subplots(1, ncols, figsize=(2.6 * ncols, 4), sharey=True)
            if ncols == 1:
                axes = [axes]
            for ax, xv in zip(axes, xs):
                data = [[r[metric] for r in raw.get((exp, xv, m), []) if metric in r]
                        for m in METHODS]
                data = [d if len(d) > 1 else [0, 0] for d in data]
                parts = ax.violinplot(data, positions=range(1, len(METHODS) + 1),
                                      showmedians=True, showextrema=True)
                for pc, m in zip(parts["bodies"], METHODS):
                    pc.set_facecolor(METHOD_COLORS[m])
                    pc.set_alpha(0.72)
                for key in ("cmedians", "cmaxes", "cmins", "cbars"):
                    if key in parts:
                        parts[key].set_linewidth(0.9)
                ax.set_xticks(range(1, len(METHODS) + 1))
                ax.set_xticklabels(METHODS, rotation=35, ha="right", fontsize=7)
                ax.set_title(f"x={xv}", fontsize=9)
            axes[0].set_ylabel(ylabel)
            fig.tight_layout()
            fname = os.path.join(_supp_dir(), f"violin_{exp}_{metric}.png")
            fig.savefig(fname, bbox_inches="tight", dpi=200)
            plt.close(fig)
            print(f"  {fname}")
            total += 1
    return total


# ── C3. Heatmap ───────────────────────────────────────────────────────────────

def fig_heatmap(summary):
    """Cumulative heatmaps: one figure per subset of methods so the reader
    can see how each new contribution shifts the ranking picture.

    Subset 1: HOM + MLHOM          (Ch.4 baseline)
    Subset 2: + RRHOM               (Ch.5 adds RRHOM)
    Subset 3: + LFHOM               (Ch.6 adds LFHOM — full picture)

    Normalisation is column-wise across ALL four methods so colours stay
    comparable across the three figures.
    """
    total = 0
    for exp, xname in EXPERIMENTS.items():
        xs = xs_for(summary, exp)
        for metric, ylabel in PLOT_METRICS:
            # full 4-method matrix used for normalisation (so scales are consistent)
            full_matrix = np.array([
                [summary.get((exp, xv, m, metric), (np.nan, 0, 0))[0] for xv in xs]
                for m in METHODS
            ])
            if np.all(np.isnan(full_matrix)):
                continue
            col_min = np.nanmin(full_matrix, axis=0)
            col_max = np.nanmax(full_matrix, axis=0)
            cmap = "RdYlGn_r" if metric in LOWER_BETTER else "RdYlGn"

            for subset_methods, subset_tag in HEATMAP_SUBSETS:
                subset_idx = [METHODS.index(m) for m in subset_methods]
                matrix = full_matrix[subset_idx, :]
                normed = (matrix - col_min) / (col_max - col_min + 1e-30)
                normed = np.clip(normed, 0, 1)

                fig, ax = plt.subplots(
                    figsize=(max(5, len(xs) * 1.3), max(2.2, len(subset_methods) * 0.7)))
                im = ax.imshow(normed, aspect="auto", cmap=cmap, vmin=0, vmax=1)
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels(xs)
                ax.set_yticks(range(len(subset_methods)))
                ax.set_yticklabels(subset_methods)
                ax.set_xlabel(xname)
                ax.set_title(ylabel, fontsize=10)
                for r, m in enumerate(subset_methods):
                    for c, xv in enumerate(xs):
                        val = matrix[r, c]
                        if not np.isnan(val):
                            ax.text(c, r, f"{val:.3g}", ha="center", va="center",
                                    fontsize=8, color="black")
                cb = plt.colorbar(im, ax=ax)
                cb.set_label("rank within column\n(0=best, 1=worst)", fontsize=8)
                fig.tight_layout()
                fname = os.path.join(_supp_dir(),
                                     f"heatmap_{exp}_{metric}_{subset_tag}.png")
                fig.savefig(fname, bbox_inches="tight", dpi=200)
                plt.close(fig)
                print(f"  {fname}")
                total += 1
    return total


# ── C4. Radar / spider ────────────────────────────────────────────────────────

def fig_radar(summary):
    # Use primary metrics only for radar
    radar_metrics = [k for k, _ in METRIC_GROUPS["primary"]]
    radar_labels  = [v for _, v in METRIC_GROUPS["primary"]]
    total = 0
    for exp, xname in EXPERIMENTS.items():
        xs    = xs_for(summary, exp)
        mid_x = xs[len(xs) // 2]
        raw_vals = {
            m: [summary.get((exp, mid_x, m, k), (0, 0, 0))[0] for k in radar_metrics]
            for m in METHODS
        }
        norm = {m: [] for m in METHODS}
        for ki, k in enumerate(radar_metrics):
            col  = [raw_vals[m][ki] for m in METHODS]
            lo, hi = min(col), max(col)
            span = hi - lo if hi != lo else 1.0
            for m in METHODS:
                v = (raw_vals[m][ki] - lo) / span
                norm[m].append(1 - v if k in LOWER_BETTER else v)

        N      = len(radar_labels)
        angles = [n / float(N) * 2 * math.pi for n in range(N)] + [0]
        fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={"polar": True})
        for m in METHODS:
            vals = norm[m] + norm[m][:1]
            ax.plot(angles, vals, linewidth=1.8, color=METHOD_COLORS[m], label=m)
            ax.fill(angles, vals, alpha=0.08, color=METHOD_COLORS[m])
        ax.set_thetagrids(np.degrees(angles[:-1]),
                          [lb.replace(" ", "\n") for lb in radar_labels], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), frameon=False)
        ax.set_title(f"Normalised primary metrics — {xname}={mid_x}\n"
                     "(higher = better on all axes)", fontsize=9, pad=18)
        fig.tight_layout()
        fname = os.path.join(_supp_dir(), f"radar_{exp}_x{mid_x}.png")
        fig.savefig(fname, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"  {fname}")
        total += 1
    return total


# ── C5. Pairwise Cohen's d effect-size heatmap ───────────────────────────────

def fig_effect_size(raw):
    # Only the sequential pairs (the story pairs) + all-pairs
    pairs  = list(itertools.combinations(METHODS, 2))
    total  = 0
    for exp, xname in EXPERIMENTS.items():
        xs = sorted({k[1] for k in raw if k[0] == exp}, key=lambda v: int(v))
        for metric, ylabel in PLOT_METRICS:
            matrix = np.full((len(pairs), len(xs)), np.nan)
            any_large = False
            for xi, xv in enumerate(xs):
                for pi, (ma, mb) in enumerate(pairs):
                    a = [r[metric] for r in raw.get((exp, xv, ma), []) if metric in r]
                    b = [r[metric] for r in raw.get((exp, xv, mb), []) if metric in r]
                    if a and b:
                        d = abs(cohens_d_paired(a, b))
                        matrix[pi, xi] = d
                        if not math.isnan(d) and d >= 0.2:
                            any_large = True
            if not any_large:
                continue
            fig, ax = plt.subplots(figsize=(max(5, len(xs) * 1.3),
                                             max(3, len(pairs) * 0.75)))
            im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xs)
            ax.set_yticks(range(len(pairs)))
            ax.set_yticklabels([f"{a} vs {b}" for a, b in pairs], fontsize=8)
            ax.set_xlabel(xname)
            ax.set_title(f"|Cohen's d| — {ylabel}", fontsize=10)
            for r in range(len(pairs)):
                for c in range(len(xs)):
                    v = matrix[r, c]
                    if not math.isnan(v):
                        ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                                fontsize=8, color="black")
            plt.colorbar(im, ax=ax, label="|Cohen's d|  (0.2=small  0.5=med  0.8=large)")
            fig.tight_layout()
            fname = os.path.join(_supp_dir(), f"effect_size_{exp}_{metric}.png")
            fig.savefig(fname, bbox_inches="tight", dpi=200)
            plt.close(fig)
            print(f"  {fname}")
            total += 1
    return total


# ── C6. Empirical CDF of latency at maximum x ────────────────────────────────

def fig_cdf(raw):
    total = 0
    for exp, xname in EXPERIMENTS.items():
        xs    = sorted({k[1] for k in raw if k[0] == exp}, key=lambda v: int(v))
        max_x = xs[-1]
        # Use latency and weighted_latency
        for metric, ylabel in [("latency", "Latency (s)"),
                                ("weighted_latency", "Weighted latency (s)")]:
            if not any(metric in r
                       for m in METHODS
                       for r in raw.get((exp, max_x, m), [])):
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            for m in METHODS:
                vals = sorted(r[metric] for r in raw.get((exp, max_x, m), [])
                              if metric in r)
                if not vals:
                    continue
                cdf = np.arange(1, len(vals) + 1) / len(vals)
                ax.step(vals, cdf, where="post",
                        color=METHOD_COLORS[m], label=m, linewidth=1.8)
            ax.set_xlabel(ylabel)
            ax.set_ylabel("Empirical CDF")
            ax.set_title(f"CDF of {ylabel} — {xname}={max_x}", fontsize=10)
            ax.legend(frameon=False)
            fig.tight_layout()
            fname = os.path.join(_supp_dir(), f"cdf_{metric}_{exp}.png")
            fig.savefig(fname, bbox_inches="tight", dpi=200)
            plt.close(fig)
            print(f"  {fname}")
            total += 1
    return total


# ── C7. Latency vs Weighted JFI scatter ──────────────────────────────────────

def fig_scatter(raw):
    total = 0
    for exp, xname in EXPERIMENTS.items():
        xs    = sorted({k[1] for k in raw if k[0] == exp}, key=lambda v: int(v))
        ncols = len(xs)
        for (mx, my, xlabel, ylabel) in [
            ("latency",  "weighted_jfi",
             "Mean latency (s)", "Weighted Jain's fairness index"),
            ("weighted_latency", "weighted_jfi",
             "Weighted latency (s)", "Weighted Jain's fairness index"),
        ]:
            fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3.5), sharey=True)
            if ncols == 1:
                axes = [axes]
            for ax, xv in zip(axes, xs):
                for m in METHODS:
                    pts = [r for r in raw.get((exp, xv, m), [])
                           if mx in r and my in r]
                    if pts:
                        ax.scatter([r[mx] for r in pts], [r[my] for r in pts],
                                   color=METHOD_COLORS[m], alpha=0.55, s=18, label=m)
                ax.set_xlabel(xlabel, fontsize=9)
                ax.set_title(f"x={xv}", fontsize=9)
            axes[0].set_ylabel(ylabel)
            handles = [mpatches.Patch(color=METHOD_COLORS[m], label=m) for m in METHODS]
            axes[-1].legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
            fig.tight_layout()
            fname = os.path.join(_supp_dir(),
                                  f"scatter_{mx}_vs_{my}_{exp}.png")
            fig.savefig(fname, bbox_inches="tight", dpi=200)
            plt.close(fig)
            print(f"  {fname}")
            total += 1
    return total


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _enable_coedge(summary):
    """Add the CoEdge benchmark (strategy 7) to every figure group when its
    results are present in summary.csv; nothing changes otherwise."""
    if not any(k[2] == "CoEdge" for k in summary):
        return False
    if "CoEdge" not in METHODS:
        METHODS.insert(0, "CoEdge")                       # external benchmark first
    SEQ_PAIRS.extend([
        ("CoEdge", "RRHOM", "bench_RRHOM_vs_CoEdge"),
        ("CoEdge", "LFHOM", "bench_LFHOM_vs_CoEdge"),
    ])
    METHOD_GROUPS.append((["CoEdge", "MLHOM", "RRHOM", "LFHOM"], "b3_proposed_vs_CoEdge"))
    HEATMAP_SUBSETS.append((["CoEdge", "HOM", "MLHOM", "RRHOM", "LFHOM"], "4_with_CoEdge"))
    return True


def _configure_from_cli():
    """--dir DIR     : read summary.csv/raw.csv from DIR and write DIR/figures
       --methods A B : draw ONLY these methods (A = baseline, B = proposed),
                       e.g.  --dir results/multiseed_coedge --methods CoEdge RRHOM"""
    import argparse
    global SUMMARY_CSV, RAW_CSV, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--methods", nargs="+", default=None)
    args = ap.parse_args()
    if args.dir:
        d = os.path.abspath(args.dir)
        SUMMARY_CSV = os.path.join(d, "summary.csv")
        RAW_CSV = os.path.join(d, "raw.csv")
        OUT_DIR = os.path.join(d, "figures")
    if args.methods:
        ms = args.methods
        METHODS[:] = ms
        SEQ_PAIRS[:] = [(ms[0], ms[-1], f"{ms[-1]}_vs_{ms[0]}")] if len(ms) >= 2 else []
        METHOD_GROUPS[:] = [(ms, "group_" + "_".join(ms))] if len(ms) > 2 else []
        HEATMAP_SUBSETS[:] = [(ms, "_".join(ms))]
        BENCHMARKS.clear()
        BENCHMARKS.update({ms[-1]: ms[0]} if len(ms) >= 2 else {})
        return True
    return False


def main():
    only_selected = _configure_from_cli()
    print("Loading summary.csv and raw.csv …")
    summary = load_summary()
    raw     = load_raw()
    if not only_selected and _enable_coedge(summary):
        print("  CoEdge benchmark found: added to all figure groups")
    print(f"  {len(summary)} summary cells, "
          f"{sum(len(v) for v in raw.values())} raw rows loaded")

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUT_DIR}\n")

    print("=" * 60)
    print("GROUP A — Sequential pairwise comparisons")
    print("  (figures skipped where p >= 0.05 at all x-values)")
    print("=" * 60)
    n = fig_sequential(summary, raw)
    print(f"  → {n} figures written")

    print("\n" + "=" * 60)
    print("GROUP B — Method-group comparisons")
    print("  B1. MLHOM + RRHOM + LFHOM")
    print("  B2. HOM + MLHOM + RRHOM + LFHOM")
    print("=" * 60)
    n = fig_groups(summary, raw)
    print(f"  → {n} figures written")

    print("\n" + "=" * 60)
    print("GROUP C — Supplementary  (all 4 methods)")
    print("=" * 60)

    print("\n  [C1] Box plots …")
    n = fig_boxplots(raw)
    print(f"       {n} figures written")

    print("\n  [C2] Violin plots …")
    n = fig_violins(raw)
    print(f"       {n} figures written")

    print("\n  [C3] Heatmaps …")
    n = fig_heatmap(summary)
    print(f"       {n} figures written")

    print("\n  [C4] Radar charts …")
    n = fig_radar(summary)
    print(f"       {n} figures written")

    print("\n  [C5] Cohen's d effect-size heatmaps …")
    n = fig_effect_size(raw)
    print(f"       {n} figures written")

    print("\n  [C6] Empirical CDFs …")
    n = fig_cdf(raw)
    print(f"       {n} figures written")

    print("\n  [C7] Scatter plots …")
    n = fig_scatter(raw)
    print(f"       {n} figures written")

    print(f"\nDone — all figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
