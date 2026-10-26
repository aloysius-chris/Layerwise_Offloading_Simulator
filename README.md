# Layer-wise offloading of DNN inference: simulator and results

This repository holds the discrete-event simulator and the result files for the article *Layer-Wise Heuristics for Latency- and Fairness-Aware Offloading of Deep Learning Inference in Edge–Cloud Networks* (IEEE Access, under review).

It reproduces every result in the article:
- MLHOM, RRHOM and LFHOM against HOM and CoEdge on a CU–DU edge–cloud network (10 ECNs, 5 DUs per ECN, 1–5 VMs per ECN, a cloud);
- 100–500 inference tasks drawn from six CNNs (AlexNet, GoogLeNet, ResNet-50, DenseNet-121, MobileNetV2, EfficientNet-B0);
- 30 seeds, with paired tests and Holm correction.

## Requirements

- Python 3.11 or later, with `numpy`, `scipy` and `matplotlib`.
- The experiments read the per-layer profiles from `machine_learning/model_profiles.json`, so TensorFlow is **not** needed to run them.
- TensorFlow/Keras and `keras-flops` are needed only to rebuild that file: `python -m machine_learning.build_model_profiles`.

## Layout

| Path | Content |
|---|---|
| `simulator.py` | Event loop, system construction, metrics |
| `system_model/` | Units (VM, cloud, ECN, DU, MD), links, and one task class per method |
| `machine_learning/` | Model definitions, the layer profiler, and `model_profiles.json` |
| `results/settings.py` | Experiment parameters (Table 5 of the article) |
| `results/run_multiseed.py` | 30-seed experiments, 95% confidence intervals, paired tests, Holm correction |
| `results/run_coedge_vs_rrhom.py` | CoEdge benchmark runs |
| `results/multiseed/`, `results/multiseed_coedge/` | Per-trial results (`raw.csv`), summaries with 95% CIs, and test results used in the article |
| `experiments/heterogeneous_relocation.py` | Supplementary experiment of the article (relocation test switched on, heterogeneous VMs) |
| `experiments/heterogeneous_raw.csv` | Per-trial results of that experiment |

Methods are selected by the strategy number passed to `Simulation.start(offloading_strategy=…)`:

| Strategy | Method |
|---|---|
| 3 | HOM |
| 4 | MLHOM |
| 5 | LFHOM |
| 6 | RRHOM |
| 7 | CoEdge |

## Reproducing the article

Run all commands from the repository root.

1. Main experiments (all methods, 30 seeds). This is slow; LFHOM dominates the run time.

   ```
   python results/run_multiseed.py
   ```

2. CoEdge benchmark:

   ```
   python results/run_coedge_vs_rrhom.py
   python results/multiseed/plot_multiseed.py --dir results/multiseed_coedge --methods CoEdge RRHOM
   ```

3. Supplementary experiment (relocation test enabled, heterogeneous VMs). The second argument is the number of worker processes.

   ```
   python experiments/heterogeneous_relocation.py run 7
   ```

The simulator is deterministic for a given seed, so these commands reproduce the files in `results/` and `experiments/` exactly.

## Implementation notes

These notes match the description in the article.

- **LFHOM** (`system_model/load_balancing_fairness_heuristic_method_task.py`, `_select_fairest_layer`): whenever a VM becomes free, it executes the next layer of the waiting task that gives the highest predicted FLOP-weighted Jain index over all tasks. Ties go to the task with the larger accumulated latency.
- **RRHOM** (strategy 6): in the main experiments, a VM executes a started task to completion (`VirtualMachine.handle_task`), so the relocation test is not reached. The relocation count is zero in `results/multiseed/raw.csv`. The supplementary script evaluates the test at every layer boundary.
- **Cloud ordering** (RRHOM and LFHOM, `Cloud.execute_layer`): the cloud serves next the task whose own clock is latest. Ties go to the task offloaded to the cloud first.

## Licence and citation

[To be added by the authors: choose a licence, for example MIT or BSD-3-Clause, and add the article citation and the Zenodo DOI once available.]
