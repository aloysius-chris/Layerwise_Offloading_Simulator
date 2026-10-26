import os
import sys
import settings
from draw_results import draw_result
from simulator import *
from log_execution import (extract, build_comparison, _write_csv,
                           TASK_EVENTS_COLS, VM_TIMELINE_COLS,
                           TASK_SUMMARY_COLS, COMPARISON_COLS, OUTDIR)

# ── execution logging flag ────────────────────────────────────────────────────
# Set to True to write task_events.csv / vm_timeline.csv / task_summary.csv /
# comparison.csv into results/execution_logs/ during the run.
ENABLE_LOGGING = False

# ── logging helpers ───────────────────────────────────────────────────────────
_CTX  = ["experiment", "x_label", "x_value"]
_TE_COLS  = _CTX + TASK_EVENTS_COLS
_VT_COLS  = _CTX + VM_TIMELINE_COLS
_TS_COLS  = _CTX + TASK_SUMMARY_COLS
_CMP_COLS = _CTX + COMPARISON_COLS

_STRAT_NAMES = {5: "LFHOM", 6: "RRHOM"}
_log_state   = {"first_data": True, "first_cmp": True}

if ENABLE_LOGGING:
    os.makedirs(OUTDIR, exist_ok=True)


def _log_sim(sim, strategy_id, experiment, x_label, x_value):
    """Extract log rows from sim and append them to the CSV files."""
    strat_name = _STRAT_NAMES.get(strategy_id, str(strategy_id))
    te, vt, ts = extract(sim, settings.random_state, strat_name)
    ctx = dict(experiment=experiment, x_label=x_label, x_value=x_value)
    for row in te: row.update(ctx)
    for row in vt: row.update(ctx)
    for row in ts: row.update(ctx)
    mode = "w" if _log_state["first_data"] else "a"
    _write_csv(os.path.join(OUTDIR, "task_events.csv"),  _TE_COLS,  te,  mode)
    _write_csv(os.path.join(OUTDIR, "vm_timeline.csv"),   _VT_COLS,  vt,  mode)
    _write_csv(os.path.join(OUTDIR, "task_summary.csv"),  _TS_COLS,  ts,  mode)
    _log_state["first_data"] = False
    return {strat_name: ts}


def _log_cmp(summaries_by_strategy, experiment, x_label, x_value):
    """Build and append comparison rows for a completed (RRHOM + LFHOM) pair."""
    ctx = dict(experiment=experiment, x_label=x_label, x_value=x_value)
    cmp_rows = build_comparison(summaries_by_strategy)
    for row in cmp_rows: row.update(ctx)
    mode = "w" if _log_state["first_cmp"] else "a"
    _write_csv(os.path.join(OUTDIR, "comparison.csv"), _CMP_COLS, cmp_rows, mode)
    _log_state["first_cmp"] = False


# ── Number of task experiment ─────────────────────────────────────────────────
print("Number of task experiment:")

latency = {}
weighted_latency = {}
jains_fairness_index = {}
weighted_jfi = {}
load_balancing = {}
resources_utilization = {}

for offloading_strategy in [6, 5]:
    latency[offloading_strategy] = []
    weighted_latency[offloading_strategy] = []
    jains_fairness_index[offloading_strategy] = []
    weighted_jfi[offloading_strategy] = []
    load_balancing[offloading_strategy] = []
    resources_utilization[offloading_strategy] = []

for md_count in settings.task_list:
    print("md_count:" + str(md_count))
    _task_summaries = {}

    for offloading_strategy in [6, 5]:
        simulation = Simulation(layer_list=settings.algo.layer_list,
                                vm_flops=settings.vm_flops, cloud_flops=settings.cloud_flops,
                                vm_count=settings.vm_count, ecn_count=settings.ecn_count, du_count=settings.du_count,
                                md_count=md_count)

        simulation.start(offloading_strategy=offloading_strategy, random_state=settings.random_state)

        latency[offloading_strategy].append(simulation.get_latency())
        print("Latency = " + str(simulation.get_latency()))
        weighted_latency[offloading_strategy].append(simulation.get_weighted_latency())
        print("Weighted Latency = " + str(simulation.get_weighted_latency()))
        jains_fairness_index[offloading_strategy].append(simulation.get_jains_fairness_index())
        print("Jain's Fairness Index = " + str(simulation.get_jains_fairness_index()))
        weighted_jfi[offloading_strategy].append(simulation.get_weighted_jains_fairness_index())
        print("Weighted JFI = " + str(simulation.get_weighted_jains_fairness_index()))
        load_balancing[offloading_strategy].append(simulation.get_load_balancing())
        print("Load Balancing = " + str(simulation.get_load_balancing()))
        resources_utilization[offloading_strategy].append(simulation.get_resources_utilization())
        print("Resources Utilization = " + str(simulation.get_resources_utilization()))

        if ENABLE_LOGGING:
            x_val = md_count * settings.ecn_count * settings.du_count
            _task_summaries.update(_log_sim(simulation, offloading_strategy,
                                            "tasks", "Number of Tasks", x_val))

    if ENABLE_LOGGING:
        _log_cmp(_task_summaries, "tasks", "Number of Tasks",
                 md_count * settings.ecn_count * settings.du_count)

draw_result(x_label='Number of Tasks', y_label="Latency (s)",
            x_labels=['100', '200', '300', '400', '500'],
            value1_label='RRHOM', value2_label='LFHOM',
            title='Comparison of the latency of RRHOM and LFHOM by different number of deep learning tasks',
            value1=latency[6], value2=latency[5])

draw_result(x_label='Number of Tasks', y_label="Weighted Latency (s)",
            x_labels=['100', '200', '300', '400', '500'],
            value1_label='RRHOM', value2_label='LFHOM',
            title='Comparison of the weighted latency of RRHOM and LFHOM by different number of deep learning tasks',
            value1=weighted_latency[6], value2=weighted_latency[5])

draw_result(x_label='Number of Tasks', y_label="Jain's Fairness Index",
            x_labels=['100', '200', '300', '400', '500'],
            value1_label='RRHOM', value2_label='LFHOM',
            title="Comparison of the Jain's fairness index of RRHOM and LFHOM by different number of deep learning tasks",
            value1=jains_fairness_index[6], value2=jains_fairness_index[5])

draw_result(x_label='Number of Tasks', y_label="Weighted Jain's Fairness Index",
            x_labels=['100', '200', '300', '400', '500'],
            value1_label='RRHOM', value2_label='LFHOM',
            title="Comparison of the weighted Jain's fairness index of RRHOM and LFHOM by different number of deep learning tasks",
            value1=weighted_jfi[6], value2=weighted_jfi[5])

draw_result(x_label='Number of Tasks', y_label='Load Balancing',
            x_labels=['100', '200', '300', '400', '500'],
            value1_label='RRHOM', value2_label='LFHOM',
            title='Comparison of the load balancing of RRHOM and LFHOM by different number of deep learning tasks',
            value1=load_balancing[6], value2=load_balancing[5])

draw_result(x_label='Number of Tasks', y_label='Resources Utilization',
            x_labels=['100', '200', '300', '400', '500'],
            value1_label='RRHOM', value2_label='LFHOM',
            title='Comparison of the Resources utilization of RRHOM and LFHOM by different number of deep learning tasks',
            value1=resources_utilization[6], value2=resources_utilization[5])


# ── Number of virtual machine experiment ──────────────────────────────────────
print("Number of virtual machine experiment:")
print("-------------------------------------")

latency = {}
weighted_latency = {}
jains_fairness_index = {}
weighted_jfi = {}
load_balancing = {}
resources_utilization = {}

for offloading_strategy in [6, 5]:
    latency[offloading_strategy] = []
    weighted_latency[offloading_strategy] = []
    jains_fairness_index[offloading_strategy] = []
    weighted_jfi[offloading_strategy] = []
    load_balancing[offloading_strategy] = []
    resources_utilization[offloading_strategy] = []

for vm_count in settings.vm_list:
    print("vm_count:" + str(vm_count))
    _vm_summaries = {}

    for offloading_strategy in [6, 5]:
        simulation = Simulation(layer_list=settings.algo.layer_list,
                                vm_flops=settings.vm_flops, cloud_flops=settings.cloud_flops,
                                vm_count=vm_count, ecn_count=settings.ecn_count, du_count=settings.du_count,
                                md_count=settings.md_count)

        simulation.start(offloading_strategy=offloading_strategy, random_state=settings.random_state)

        latency[offloading_strategy].append(simulation.get_latency())
        print("Latency = " + str(simulation.get_latency()))
        weighted_latency[offloading_strategy].append(simulation.get_weighted_latency())
        print("Weighted Latency = " + str(simulation.get_weighted_latency()))
        jains_fairness_index[offloading_strategy].append(simulation.get_jains_fairness_index())
        print("Jain's Fairness Index = " + str(simulation.get_jains_fairness_index()))
        weighted_jfi[offloading_strategy].append(simulation.get_weighted_jains_fairness_index())
        print("Weighted JFI = " + str(simulation.get_weighted_jains_fairness_index()))
        load_balancing[offloading_strategy].append(simulation.get_load_balancing())
        print("Load Balancing = " + str(simulation.get_load_balancing()))
        resources_utilization[offloading_strategy].append(simulation.get_resources_utilization())
        print("Resources Utilization = " + str(simulation.get_resources_utilization()))

        if ENABLE_LOGGING:
            _vm_summaries.update(_log_sim(simulation, offloading_strategy,
                                          "vms", "Number of VMs per ECN", vm_count))

    if ENABLE_LOGGING:
        _log_cmp(_vm_summaries, "vms", "Number of VMs per ECN", vm_count)

draw_result(x_label='Number of VMs per ECN', y_label="Latency (s)",
            x_labels=settings.vm_label_list,
            value1_label='RRHOM', value2_label='LFHOM',
            title='Comparison of the latency of RRHOM and LFHOM by different number of VMs per ECN',
            value1=latency[6], value2=latency[5])

draw_result(x_label='Number of VMs per ECN', y_label="Weighted Latency (s)",
            x_labels=settings.vm_label_list,
            value1_label='RRHOM', value2_label='LFHOM',
            title='Comparison of the weighted latency of RRHOM and LFHOM by different number of VMs per ECN',
            value1=weighted_latency[6], value2=weighted_latency[5])

draw_result(x_label='Number of VMs per ECN', y_label="Jain's Fairness Index",
            x_labels=settings.vm_label_list,
            value1_label='RRHOM', value2_label='LFHOM',
            title="Comparison of the Jain's fairness index of RRHOM and LFHOM by different number of VMs per ECN",
            value1=jains_fairness_index[6], value2=jains_fairness_index[5])

draw_result(x_label='Number of VMs per ECN', y_label="Weighted Jain's Fairness Index",
            x_labels=settings.vm_label_list,
            value1_label='RRHOM', value2_label='LFHOM',
            title="Comparison of the weighted Jain's fairness index of RRHOM and LFHOM by different number of VMs per ECN",
            value1=weighted_jfi[6], value2=weighted_jfi[5])

draw_result(x_label='Number of VMs per ECN', y_label='Load Balancing',
            x_labels=settings.vm_label_list,
            value1_label='RRHOM', value2_label='LFHOM',
            title='Comparison of the load balancing of RRHOM and LFHOM by different number of VMs per ECN',
            value1=load_balancing[6], value2=load_balancing[5])

draw_result(x_label='Number of VMs per ECN', y_label='Resources Utilization',
            x_labels=settings.vm_label_list,
            value1_label='RRHOM', value2_label='LFHOM',
            title='Comparison of the Resources utilization of RRHOM and LFHOM by different number of VMs per ECN',
            value1=resources_utilization[6], value2=resources_utilization[5])


# ── Capacity of virtual machine experiment ────────────────────────────────────
print("Capacity of virtual machine experiment:")
print("-------------------------------------")

latency = {}
weighted_latency = {}
jains_fairness_index = {}
weighted_jfi = {}
load_balancing = {}
resources_utilization = {}

for offloading_strategy in [3, 5]:
    latency[offloading_strategy] = []
    weighted_latency[offloading_strategy] = []
    jains_fairness_index[offloading_strategy] = []
    weighted_jfi[offloading_strategy] = []
    load_balancing[offloading_strategy] = []
    resources_utilization[offloading_strategy] = []

for vm_flops in settings.vm_flops_list:
    print("vm_flops:" + str(vm_flops))

    for offloading_strategy in [3, 5]:
        simulation = Simulation(layer_list=settings.algo.layer_list,
                                vm_flops=vm_flops, cloud_flops=settings.cloud_flops,
                                vm_count=settings.md_count, ecn_count=settings.ecn_count, du_count=settings.du_count,
                                md_count=settings.md_count)

        simulation.start(offloading_strategy=offloading_strategy, random_state=settings.random_state)

        latency[offloading_strategy].append(simulation.get_latency())
        print("Latency = " + str(simulation.get_latency()))
        weighted_latency[offloading_strategy].append(simulation.get_weighted_latency())
        print("Weighted Latency = " + str(simulation.get_weighted_latency()))
        jains_fairness_index[offloading_strategy].append(simulation.get_jains_fairness_index())
        print("Jain's Fairness Index = " + str(simulation.get_jains_fairness_index()))
        weighted_jfi[offloading_strategy].append(simulation.get_weighted_jains_fairness_index())
        print("Weighted JFI = " + str(simulation.get_weighted_jains_fairness_index()))
        load_balancing[offloading_strategy].append(simulation.get_load_balancing())
        print("Load Balancing = " + str(simulation.get_load_balancing()))
        resources_utilization[offloading_strategy].append(simulation.get_resources_utilization())
        print("Resources Utilization = " + str(simulation.get_resources_utilization()))

draw_result(x_label='Virtual Machines Flops', y_label="Latency (s)",
            x_labels=settings.vm_flops_list,
            value1_label='HOM', value2_label='LFHOM',
            title='Comparison of the latency of HOM and LFHOM by different number of VMs per ECN flops.',
            value1=latency[3], value2=latency[5])

draw_result(x_label='Virtual Machines Flops', y_label="Weighted Latency (s)",
            x_labels=settings.vm_flops_list,
            value1_label='HOM', value2_label='LFHOM',
            title='Comparison of the weighted latency of HOM and LFHOM by different number of VMs per ECN flops.',
            value1=weighted_latency[3], value2=weighted_latency[5])

draw_result(x_label='Virtual Machines Flops', y_label="Jain's Fairness Index",
            x_labels=settings.vm_flops_list,
            value1_label='HOM', value2_label='LFHOM',
            title="Comparison of the Jain's fairness index of HOM and LFHOM by different number of VMs per ECN flops.",
            value1=jains_fairness_index[3], value2=jains_fairness_index[5])

draw_result(x_label='Virtual Machines Flops', y_label="Weighted Jain's Fairness Index",
            x_labels=settings.vm_flops_list,
            value1_label='HOM', value2_label='LFHOM',
            title="Comparison of the weighted Jain's fairness index of HOM and LFHOM by different number of VMs per ECN flops.",
            value1=weighted_jfi[3], value2=weighted_jfi[5])

draw_result(x_label='Virtual Machines Flops', y_label='Load Balancing',
            x_labels=settings.vm_flops_list,
            value1_label='HOM', value2_label='LFHOM',
            title='Comparison of the load balancing of HOM and LFHOM by different number of VMs per ECN flops.',
            value1=load_balancing[3], value2=load_balancing[5])

draw_result(x_label='Virtual Machines Flops', y_label='Resources Utilization',
            x_labels=settings.vm_flops_list,
            value1_label='HOM', value2_label='LFHOM',
            title='Comparison of the Resources utilization of HOM and LFHOM by different number of VMs per ECN flops.',
            value1=resources_utilization[3], value2=resources_utilization[5])
