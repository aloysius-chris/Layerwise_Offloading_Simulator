import settings
from draw_results import draw_result
from simulator import *

print("Number of task experiment:")
print("--------------------------")

latency = {}
jains_fairness_index = {}
load_balancing = {}
resources_utilization = {}

for offloading_strategy in [1, 2, 3, 4, 5, 6, 7]:
    latency[offloading_strategy] = []
    jains_fairness_index[offloading_strategy] = []
    load_balancing[offloading_strategy] = []
    resources_utilization[offloading_strategy] = []

for md_count in settings.task_list:
    print("md_count:" + str(md_count))

    for offloading_strategy in [1, 2, 3, 4, 5, 6, 7]:
        simulation = Simulation(layer_list=settings.algo.layer_list,
                                vm_flops=settings.vm_flops, cloud_flops=settings.cloud_flops,
                                vm_count=settings.vm_count, ecn_count=settings.ecn_count, du_count=settings.du_count,
                                md_count=md_count)

        simulation.start(offloading_strategy=offloading_strategy, random_state=settings.random_state)

        latency[offloading_strategy].append(simulation.get_latency())
        print("Latency = " + str(simulation.get_latency()))
        jains_fairness_index[offloading_strategy].append(simulation.get_jains_fairness_index())
        print("Jain's Fairness Index = " + str(simulation.get_jains_fairness_index()))
        load_balancing[offloading_strategy].append(simulation.get_load_balancing())
        print("Load Balancing = " + str(simulation.get_load_balancing()))
        resources_utilization[offloading_strategy].append(simulation.get_resources_utilization())
        print("Resources Utilization = " + str(simulation.get_resources_utilization()))

draw_result(x_label='Number of Tasks', y_label="Latency (s)",
            x_labels=settings.task_label_list,
            value_labels=['LO', 'CO', 'HOM', 'MLHOM', 'LFHOM', 'RRHOM', 'CoEdge'],
            title='Comparison of the latency of all by different number of deep learning tasks.',
            values=latency)

draw_result(x_label='Number of Tasks', y_label="Jain's Fairness Index",
            x_labels=settings.task_label_list,
            value_labels=['LO', 'CO', 'HOM', 'MLHOM', 'LFHOM', 'RRHOM', 'CoEdge'],
            title="Comparison of the Jain's fairness index of all by different number of deep learning tasks.",
            values=jains_fairness_index)

draw_result(x_label='Number of Tasks', y_label='Load Balancing',
            x_labels=settings.task_label_list,
            value_labels=['LO', 'CO', 'HOM', 'MLHOM', 'LFHOM', 'RRHOM', 'CoEdge'],
            title='Comparison of the load balancing of all by different number of deep learning tasks.',
            values=load_balancing)

draw_result(x_label='Number of Tasks', y_label='Resources Utilization',
            x_labels=settings.task_label_list,
            value_labels=['LO', 'CO', 'HOM', 'MLHOM', 'LFHOM', 'RRHOM', 'CoEdge'],
            title='Comparison of the Resources utilization of all by different number of deep learning tasks.',
            values=resources_utilization)

print("Number of virtual machine experiment:")
print("-------------------------------------")

latency = {}
jains_fairness_index = {}
load_balancing = {}
resources_utilization = {}

for offloading_strategy in [1, 2, 3, 4, 5, 6, 7]:
    latency[offloading_strategy] = []
    jains_fairness_index[offloading_strategy] = []
    load_balancing[offloading_strategy] = []
    resources_utilization[offloading_strategy] = []

for vm_count in settings.vm_list:
    print("vm_count:" + str(vm_count))

    for offloading_strategy in [1, 2, 3, 4, 5, 6, 7]:
        simulation = Simulation(layer_list=settings.algo.layer_list,
                                vm_flops=settings.vm_flops, cloud_flops=settings.cloud_flops,
                                vm_count=vm_count, ecn_count=settings.ecn_count, du_count=settings.du_count,
                                md_count=settings.md_count)

        simulation.start(offloading_strategy=offloading_strategy, random_state=settings.random_state)

        latency[offloading_strategy].append(simulation.get_latency())
        print("Latency = " + str(simulation.get_latency()))
        jains_fairness_index[offloading_strategy].append(simulation.get_jains_fairness_index())
        print("Jain's Fairness Index = " + str(simulation.get_jains_fairness_index()))
        load_balancing[offloading_strategy].append(simulation.get_load_balancing())
        print("Load Balancing = " + str(simulation.get_load_balancing()))
        resources_utilization[offloading_strategy].append(simulation.get_resources_utilization())
        print("Resources Utilization = " + str(simulation.get_resources_utilization()))

draw_result(x_label='Number of VMs per ECN', y_label="Latency (s)",
            x_labels=settings.vm_label_list,
            value_labels=['LO', 'CO', 'HOM', 'MLHOM', 'LFHOM', 'RRHOM', 'CoEdge'],
            title='Comparison of the latency of all by different number of deep learning tasks.',
            values=latency)

draw_result(x_label='Number of VMs per ECN', y_label="Jain's Fairness Index",
            x_labels=settings.vm_label_list,
            value_labels=['LO', 'CO', 'HOM', 'MLHOM', 'LFHOM', 'RRHOM', 'CoEdge'],
            title="Comparison of the Jain's fairness index of all by different number of deep learning tasks.",
            values=jains_fairness_index)

draw_result(x_label='Number of VMs per ECN', y_label='Load Balancing',
            x_labels=settings.vm_label_list,
            value_labels=['LO', 'CO', 'HOM', 'MLHOM', 'LFHOM', 'RRHOM', 'CoEdge'],
            title='Comparison of the load balancing of all by different number of deep learning tasks.',
            values=load_balancing)

draw_result(x_label='Number of VMs per ECN', y_label='Resources Utilization',
            x_labels=settings.vm_label_list,
            value_labels=['LO', 'CO', 'HOM', 'MLHOM', 'LFHOM', 'RRHOM', 'CoEdge'],
            title='Comparison of the Resources utilization of all by different number of deep learning tasks.',
            values=resources_utilization)

print("Capacity of virtual machine experiment:")
print("-------------------------------------")

latency = {}
jains_fairness_index = {}
load_balancing = {}
resources_utilization = {}

for offloading_strategy in [1, 2, 3, 4, 5, 6, 7]:
    latency[offloading_strategy] = []
    jains_fairness_index[offloading_strategy] = []
    load_balancing[offloading_strategy] = []
    resources_utilization[offloading_strategy] = []

for vm_flops in settings.vm_flops_list:
    print("vm_flops:" + str(vm_flops))

    for offloading_strategy in [1, 2, 3, 4, 5, 6, 7]:
        simulation = Simulation(layer_list=settings.algo.layer_list,
                                vm_flops=vm_flops, cloud_flops=settings.cloud_flops,
                                vm_count=settings.md_count, ecn_count=settings.ecn_count, du_count=settings.du_count,
                                md_count=settings.md_count)

        simulation.start(offloading_strategy=offloading_strategy, random_state=settings.random_state)

        latency[offloading_strategy].append(simulation.get_latency())
        print("Latency = " + str(simulation.get_latency()))
        jains_fairness_index[offloading_strategy].append(simulation.get_jains_fairness_index())
        print("Jain's Fairness Index = " + str(simulation.get_jains_fairness_index()))
        load_balancing[offloading_strategy].append(simulation.get_load_balancing())
        print("Load Balancing = " + str(simulation.get_load_balancing()))
        resources_utilization[offloading_strategy].append(simulation.get_resources_utilization())
        print("Resources Utilization = " + str(simulation.get_resources_utilization()))

draw_result(x_label='Virtual Machines Flops', y_label="Latency (s)",
            x_labels=settings.vm_flops_list,
            value_labels=['LO', 'CO', 'HOM', 'MLHOM', 'LFHOM', 'RRHOM', 'CoEdge'],
            title='Comparison of the latency of all by different number of deep learning tasks.',
            values=latency)

draw_result(x_label='Virtual Machines Flops', y_label="Jain's Fairness Index",
            x_labels=settings.vm_flops_list,
            value_labels=['LO', 'CO', 'HOM', 'MLHOM', 'LFHOM', 'RRHOM', 'CoEdge'],
            title="Comparison of the Jain's fairness index of all by different number of deep learning tasks.",
            values=jains_fairness_index)

draw_result(x_label='Virtual Machines Flops', y_label='Load Balancing',
            x_labels=settings.vm_flops_list,
            value_labels=['LO', 'CO', 'HOM', 'MLHOM', 'LFHOM', 'RRHOM', 'CoEdge'],
            title='Comparison of the load balancing of all by different number of deep learning tasks.',
            values=load_balancing)

draw_result(x_label='Virtual Machines Flops', y_label='Resources Utilization',
            x_labels=settings.vm_flops_list,
            value_labels=['LO', 'CO', 'HOM', 'MLHOM', 'LFHOM', 'RRHOM', 'CoEdge'],
            title='Comparison of the Resources utilization of all by different number of deep learning tasks.',
            values=resources_utilization)
