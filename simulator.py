import random
import statistics
from enum import Enum

from system_model.load_balancing_fairness_heuristic_method_task import LoadBalancingFairnessHeuristicTask
from system_model.minimum_latency_fairness_heuristic_method_task import MinimumLatencyFairnessHeuristicTask
from system_model.shortest_path_utility import Graph, dijsktra
from system_model.network import Network
from system_model.cloud import Cloud
from system_model.edge_computing_node import EdgeComputingNode
from system_model.task_utility import TaskUtility
from system_model.virtual_machine import VirtualMachine
from system_model.distribution_unit import DistributionUnit
from system_model.mobile_device import MobileDevice
from system_model.local_task import LocalTask
from system_model.cloud_task import CloudTask
from system_model.shortest_path_heuristic_task import ShortestPathHeuristicTask
from machine_learning.layer import Layer
from system_model.minimum_latency_heuristic_method_task import MinimumLatencyHeuristicTask
from system_model.load_balncing_heuristic_method_task import \
    LoadBalancingHeuristicTask
from system_model.latency_fairness_heuristic_method_task import LatencyFairnessHeuristicTask
from system_model.coedge_task import CoEdgeTask


class Simulation:
    def __init__(self,
                 sort_by_latency=False,
                 ecn_count=None,
                 vm_count=None,
                 du_count=None,
                 md_count=None,
                 vm_flops=None,
                 cloud_flops=None,
                 layer_list=None):
        if ecn_count is None:
            ecn_count = int(input("Enter number of ecn i.e. 10:"))
        if vm_count is None:
            vm_count = int(input("Enter number of vm i.e. 5:"))
        if du_count is None:
            du_count = int(input("Enter number of du i.e. 5:"))
        if md_count is None:
            md_count = int(input("Enter number of md i.e. 50:"))
        if vm_flops is None:
            vm_flops = int(input("Enter  vm flops i.e. 30:"))
        if cloud_flops is None:
            cloud_flops = int(input("Enter cloud flops i.e. 1000:"))

        self.log_list = []

        if layer_list is None:
            layer_list = []
            layer_list_count = int(input("Enter task layer list count 2:"))

            layer_list = []
            for layer_i in range(layer_list_count):
                layer = Layer()
                for layer_index in range(layer_list_count):
                    print("Layer " + str(layer_i + 1))
                    layer.name = input("Enter layer name:")
                    layer.data_size = int(input("Enter layer data size:"))
                    layer.flops_size = int(input("Enter layer computation flops:"))
                    layer_list.append(layer)

        self.offloading_strategy = 0
        self.layer_list = layer_list

        self.sort_by_latency = sort_by_latency

        self.cloud = Cloud()
        self.edge_computing_node_list = []
        self.virtual_machine_list = []
        self.distribution_unit_list = []
        self.mobile_device_list = []
        self.task_list = []

        self.edge_computing_node_to_cloud_network = Network(self.cloud)
        self.edge_computing_node_to_cloud_network.name = "ecn_to_cloud"
        self.edge_computing_node_to_cloud_network.speed = 220 * 1024 * 1024 / 8

        self.cloud_flops = cloud_flops

        self.cloud.name = 'cloud'
        self.cloud.flops = self.cloud_flops

        self.edge_computing_node_count = ecn_count
        for ecn_i in range(self.edge_computing_node_count):
            ecn = EdgeComputingNode()
            ecn.name = "ecn" + str(ecn_i + 1)

            self.virtual_machine_flops = vm_flops

            self.virtual_machine_count = vm_count
            for vm_i in range(self.virtual_machine_count):
                vm = VirtualMachine()
                vm.name = "vm" + str(vm_i + 1) + "_" + ecn.name
                vm.flops = self.virtual_machine_flops

                vm.edge_computing_node = ecn
                ecn.virtual_machine_list.append(vm)

                self.virtual_machine_list.append(vm)

            du_ecn_network = Network(ecn)
            du_ecn_network.name = "du_to_" + ecn.name
            du_ecn_network.speed = 540 * 1024 * 1024 / 8

            self.distribution_unit_count = du_count
            for du_i in range(self.distribution_unit_count):
                du = DistributionUnit()
                du.name = "du" + str(du_i + 1) + "_" + ecn.name

                md_du_network = Network(du)
                md_du_network.name = "md_to_" + du.name
                md_du_network.speed = 1200 * 1024 * 1024 / 8

                self.mobile_device_count = md_count
                for md_i in range(self.mobile_device_count):
                    md = MobileDevice()
                    md.name = "md" + str(md_i + 1) + "_" + du.name

                    md.distribution_unit_network = md_du_network
                    md.distribution_unit = du
                    du.mobile_device_list.append(md)

                    self.mobile_device_list.append(md)

                du.edge_computing_node_network = du_ecn_network
                du.edge_computing_node = ecn
                ecn.distribution_unit_list.append(du)

                self.distribution_unit_list.append(du)

            ecn.cloud_network = self.edge_computing_node_to_cloud_network
            ecn.cloud = self.cloud
            self.cloud.edge_computing_node_list.append(ecn)

            self.edge_computing_node_list.append(ecn)

        ecn_to_ecn_speed = 540 * 1024 * 1024 / 8

        for ecn_i in range(self.edge_computing_node_count):
            ecn = self.edge_computing_node_list[ecn_i]
            if ecn_i == 0:
                to_ecn = self.edge_computing_node_list[1]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[2]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

            if ecn_i == 1:
                to_ecn = self.edge_computing_node_list[0]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[3]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[4]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

            if ecn_i == 2:
                to_ecn = self.edge_computing_node_list[0]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[3]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[6]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

            if ecn_i == 3:
                to_ecn = self.edge_computing_node_list[1]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[2]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[5]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

            if ecn_i == 4:
                to_ecn = self.edge_computing_node_list[1]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[5]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[7]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

            if ecn_i == 5:
                to_ecn = self.edge_computing_node_list[3]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[4]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[6]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[7]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[8]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

            if ecn_i == 6:
                to_ecn = self.edge_computing_node_list[2]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[5]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[8]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

            if ecn_i == 7:
                to_ecn = self.edge_computing_node_list[4]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[5]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)
                to_ecn = self.edge_computing_node_list[9]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

            if ecn_i == 8:
                to_ecn = self.edge_computing_node_list[5]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[6]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[9]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

            if ecn_i == 9:
                to_ecn = self.edge_computing_node_list[7]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

                to_ecn = self.edge_computing_node_list[8]
                ecn.edge_computing_node_list.append(to_ecn)
                ecn.edge_computing_node_network[to_ecn] = Network(to_ecn, ecn.name + "_" + to_ecn.name,
                                                                  ecn_to_ecn_speed)

        g = Graph()

        for ecn_i in self.edge_computing_node_list:
            for ecn_j in ecn_i.edge_computing_node_list:
                g.add_edge(ecn_i, ecn_j, 1)

        for ecn_i in self.edge_computing_node_list:
            for ecn_j in self.edge_computing_node_list:
                ecn_i.shortest_path[ecn_j] = dijsktra(g, ecn_i, ecn_j)

        inter_ecn_speed = 100 * 1024 * 1024 * 1024 / 8
        for vm_source in self.virtual_machine_list:
            ecn_source = vm_source.edge_computing_node
            for vm_target in self.virtual_machine_list:
                ecn_target = vm_target.edge_computing_node
                if vm_source != vm_target:
                    if ecn_source == ecn_target:
                        vm_source.direct_computation_unit_list.append([vm_target, inter_ecn_speed])
                    else:
                        # rate of an h-hop inter-ECN path = gamma / h (Eq. 4-12); the path lists the ECNs, so h = nodes - 1
                        hops = max(1, len(ecn_source.shortest_path[ecn_target]) - 1)
                        vm_source.direct_computation_unit_list.append([vm_target, ecn_to_ecn_speed / hops])
            # the cloud is also a relocation candidate (Algorithm 2), reached at the ECN-cloud rate delta
            vm_source.direct_computation_unit_list.append([self.cloud, self.edge_computing_node_to_cloud_network.speed])

        if layer_list is None:
            layer_list = []
        self.layer_list = layer_list

    def get_offloading_strategy_string(self):
        if self:
            pass
        if self.offloading_strategy == 1:
            return "LO"

        if self.offloading_strategy == 2:
            return "Co"

        if self.offloading_strategy == 3:
            return "HOM"

        if self.offloading_strategy == 4:
            return "Minimum Latency HOM"

        if self.offloading_strategy == 5:
            return "Latency Fairness HOM"

        if self.offloading_strategy == 6:
            return "Residual Relocation HOM (RRHOM)"

        if self.offloading_strategy == 7:
            return "CoEdge (Hu, Sun and Ren 2020)"

    def sort_task_list_by_end_time(self):
        task_list = self.task_list.copy()
        task_list.sort(key=lambda x: (x.end_time, x.name))
        return task_list

    def start(self, offloading_strategy=0, random_state=0, task_list=None):
        random.seed(random_state)

        self.offloading_strategy = offloading_strategy

        print("Strategy " + self.get_offloading_strategy_string())

        if task_list is not None:
            self.task_list = task_list
        else:
            task_list = random.choices(range(len(self.mobile_device_list)), k=len(self.mobile_device_list))

            for task_i in task_list:
                task = None
                if self.offloading_strategy == 1:
                    task = LocalTask(self.mobile_device_list[task_i])

                if self.offloading_strategy == 2:
                    task = CloudTask(self.mobile_device_list[task_i])

                if self.offloading_strategy == 3:
                    task = ShortestPathHeuristicTask(self.mobile_device_list[task_i])

                if self.offloading_strategy == 4:
                    task = MinimumLatencyHeuristicTask(self.mobile_device_list[task_i])

                if self.offloading_strategy == 5:
                    task = LoadBalancingFairnessHeuristicTask(self.mobile_device_list[task_i])

                if self.offloading_strategy == 6:
                    task = LoadBalancingHeuristicTask(self.mobile_device_list[task_i])

                if self.offloading_strategy == 7:
                    # CoEdge benchmark: layer partition + node sequence decided once on arrival
                    task = CoEdgeTask(self.mobile_device_list[task_i])

                task.name = self.mobile_device_list[task_i].name + "_task"
                task.set_simulation(self)
                if all(isinstance(item, list) for item in self.layer_list):
                    task.layer_list = random.choice(self.layer_list)
                else:
                    task.layer_list = self.layer_list
                task.current_layer = task.layer_list[0]


                self.task_list.append(task)

        while True:
            sorted_task_list = self.sort_task_list_by_end_time()

            next_task = next((task_layer for task_layer in sorted_task_list if not task_layer.completed), None)

            if next_task is None:
                break
            else:
                if getattr(next_task.next_sub_task, "__name__", "") == "response_to_in_virtual_machine_state":
                    next_task_vm = next_task.next_sub_task_parameter_dict["virtual_machine"]

                    self.log_list.append({"handle_task": next_task,
                                          "time": next_task.end_time,
                                          "action": "VirtualMachine.handle_task_event",
                                          "parameters": next_task.next_sub_task_parameter_dict})

                    if self.offloading_strategy == 6:
                        # RRHOM (Algorithm 2): after every executed layer the task's own
                        # handler checks whether its residual layers should be relocated to
                        # another VM (any ECN) or to the cloud, and relocates them if so.
                        next_task_vm.handle_task(next_task)
                    elif self.offloading_strategy == 7:
                        # CoEdge: the task's own handler runs its current planned segment
                        # on this VM and then forwards the activation to the next node.
                        next_task.execute_next_sub_task()
                    elif self.offloading_strategy == 5:
                        # LFHOM: the task's own handler checks for relocation, handles the
                        # VM-busy wait, and selects the layer to execute next using the
                        # greedy JFI-maximisation heuristic (_select_fairest_layer).
                        next_task.execute_next_sub_task()
                    else:
                        if self.offloading_strategy == 4:
                            next_task_vm.handle_task(next_task)
                        else:
                            next_task_vm.handle_task_layer()
                else:
                    next_task.execute_next_sub_task()

    def get_latency(self):
        latency = 0
        for task in self.task_list:
            latency += task.get_latency_time() / len(self.task_list)
        return latency

    def get_jains_fairness_index(self):
        latency = [task.get_latency_time() for task in self.task_list]
        n = len(latency)
        sum_x = sum(latency)
        sum_x2 = sum(x ** 2 for x in latency)
        return (sum_x ** 2) / (n * sum_x2)

    def get_weighted_jains_fairness_index(self):
        """Weighted JFI: normalise each task's latency by its total model FLOPs
        before computing Jain's index.  A task with k× more FLOPs is expected
        to take k× longer; perfect fairness gives WJFI = 1.0."""
        n = len(self.task_list)
        ys = []
        for task in self.task_list:
            x = task.get_latency_time()
            w = task.get_flops_size()
            ys.append(x / w if w > 0 else x)
        sum_y  = sum(ys)
        sum_y2 = sum(y ** 2 for y in ys)
        if sum_y2 <= 0:
            return 1.0
        return (sum_y ** 2) / (n * sum_y2)

    def get_weighted_latency(self):
        """FLOPs-weighted mean latency: each task's latency is weighted by its
        total model FLOPs, so compute-heavy tasks contribute proportionally more."""
        total_w = sum(task.get_flops_size() for task in self.task_list)
        if total_w <= 0:
            return self.get_latency()
        return sum(task.get_latency_time() * task.get_flops_size()
                   for task in self.task_list) / total_w

    def get_resources_utilization(self):
        max_time = max(self.cloud.end_time,  max([vm.end_time for vm in self.virtual_machine_list]))
        total_resources = max_time * (self.cloud.flops + sum([vm.flops for vm in self.virtual_machine_list]))
        used_resources = sum([task.get_flops_size() for task in self.task_list])

        return used_resources / total_resources

    def get_load_balancing(self):
        load_balancing = []
        for vm in self.virtual_machine_list:
            load_balancing.append(vm.executed_flops)

        return statistics.variance(load_balancing)

    def get_used_vm_count(self):
        vm_count = 0
        for vm in self.virtual_machine_list:
            if len(vm.task_list) > 0:
                vm_count += 1
        return vm_count

    def add_log(self, log, log_source):
        log_source.log_list.append(log)

        self_log = log.copy()
        self_log["log_source"] = log_source
        self.log_list.append(self_log)

        result = ""
        for k, v in self_log.items():
            if "time" in k:
                v = str(v)[0:6]
            if isinstance(v, Enum):
                v = v.value
            result += k + ": " + str(v) + ", "
        #print("{" + result[:-2] + "}")

    def print_log_list(self):
        print(self)
        print("-------------")

        for log in self.log_list:
            result = ""
            for k, v in log.items():
                if "time" in k:
                    v = str(v)[0:6]
                if isinstance(v, Enum):
                    v = v.value
                result += k + ": " + str(v) + ", "
            print("{" + result[:-2] + "}")
