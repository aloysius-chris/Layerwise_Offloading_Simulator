import settings
from draw_results import draw_result
from simulator import *

print("Number of task experiment:")
print("--------------------------")

latency = {}
jains_fairness_index = {}
load_balancing = {}
resources_utilization = {}

for offloading_strategy in [3, 4]:
    latency[offloading_strategy] = []
    jains_fairness_index[offloading_strategy] = []
    load_balancing[offloading_strategy] = []
    resources_utilization[offloading_strategy] = []

for md_count in settings.task_list:
    print("md_count:" + str(md_count))

    for offloading_strategy in [3, 4]:
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
            value1_label='HOM', value2_label='MLHOM',
            title='Comparison of the latency of HOM and MLHOM by different number of deep learning tasks',
            value1=latency[3], value2=latency[4])

draw_result(x_label='Number of Tasks', y_label="Jain's Fairness Index",
            x_labels=settings.task_label_list,
            value1_label='HOM', value2_label='MLHOM',
            title="Comparison of the Jain's fairness index of HOM and MLHOM by different number of deep learning tasks",
            value1=jains_fairness_index[3], value2=jains_fairness_index[4])

draw_result(x_label='Number of Tasks', y_label='Load Balancing',
            x_labels=settings.task_label_list,
            value1_label='HOM', value2_label='MLHOM',
            title='Comparison of the load balancing of HOM and MLHOM by different number of deep learning tasks',
            value1=load_balancing[3], value2=load_balancing[4])

draw_result(x_label='Number of Tasks', y_label='Resources Utilization',
            x_labels=settings.task_label_list,
            value1_label='HOM', value2_label='MLHOM',
            title='Comparison of the Resources utilization of HOM and MLHOM by different number of deep learning tasks',
            value1=resources_utilization[3], value2=resources_utilization[4])

print("Number of virtual machine experiment:")
print("-------------------------------------")

latency = {}
jains_fairness_index = {}
load_balancing = {}
resources_utilization = {}

for offloading_strategy in [3, 4]:
    latency[offloading_strategy] = []
    jains_fairness_index[offloading_strategy] = []
    load_balancing[offloading_strategy] = []
    resources_utilization[offloading_strategy] = []

for vm_count in settings.vm_list:
    print("vm_count:" + str(vm_count))

    for offloading_strategy in [3, 4]:
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
            value1_label='HOM', value2_label='MLHOM',
            title='Comparison of the latency of HOM and MLHOM by different number of VMs per ECN',
            value1=latency[3], value2=latency[4])

draw_result(x_label='Number of VMs per ECN', y_label="Jain's Fairness Index",
            x_labels=settings.vm_label_list,
            value1_label='HOM', value2_label='MLHOM',
            title="Comparison of the Jain's fairness index of HOM and MLHOM by different number of VMs per ECN",
            value1=jains_fairness_index[3], value2=jains_fairness_index[4])

draw_result(x_label='Number of VMs per ECN', y_label='Load Balancing',
            x_labels=settings.vm_label_list,
            value1_label='HOM', value2_label='MLHOM',
            title='Comparison of the load balancing of HOM and MLHOM by different number of VMs per ECN',
            value1=load_balancing[3], value2=load_balancing[4])

draw_result(x_label='Number of VMs per ECN', y_label='Resources Utilization',
            x_labels=settings.vm_label_list,
            value1_label='HOM', value2_label='MLHOM',
            title='Comparison of the Resources utilization of HOM and MLHOM by different number of VMs per ECN',
            value1=resources_utilization[3], value2=resources_utilization[4])

"""
print("Capacity of virtual machine experiment:")
print("-------------------------------------")

latency = {}
jains_fairness_index = {}
load_balancing = {}
resources_utilization = {}

for offloading_strategy in [3, 4]:
    latency[offloading_strategy] = []
    jains_fairness_index[offloading_strategy] = []
    load_balancing[offloading_strategy] = []
    resources_utilization[offloading_strategy] = []

for vm_flops in settings.vm_flops_list:
    print("vm_flops:" + str(vm_flops))

    for offloading_strategy in [3, 4]:
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
            value1_label='HOM', value2_label='MLHOM',
            title='Comparison of the latency of HOM and MLHOM by different number of VMs per ECN flops.',
            value1=latency[3], value2=latency[4])

draw_result(x_label='Virtual Machines Flops', y_label="Jain's Fairness Index",
            x_labels=settings.vm_flops_list,
            value1_label='HOM', value2_label='MLHOM',
            title="Comparison of the Jain's fairness index of HOM and MLHOM by different number of VMs per ECN flops.",
            value1=jains_fairness_index[3], value2=jains_fairness_index[4])

draw_result(x_label='Virtual Machines Flops', y_label='Load Balancing',
            x_labels=settings.vm_flops_list,
            value1_label='HOM', value2_label='MLHOM',
            title='Comparison of the load balancing of HOM and MLHOM by different number of VMs per ECN flops.',
            value1=load_balancing[3], value2=load_balancing[4])

draw_result(x_label='Virtual Machines Flops', y_label='Resources Utilization',
            x_labels=settings.vm_flops_list,
            value1_label='HOM', value2_label='MLHOM',
            title='Comparison of the Resources utilization of HOM and MLHOM by different number of VMs per ECN flops.',
            value1=resources_utilization[3], value2=resources_utilization[4])
"""