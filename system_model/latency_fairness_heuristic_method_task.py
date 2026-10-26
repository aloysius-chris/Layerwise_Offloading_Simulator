from system_model.base_task import BaseTask, TaskLogType
from system_model.shortest_path_heuristic_task import ShortestPathHeuristicTask
from system_model.task_utility import TaskUtility


class LatencyFairnessHeuristicTask(ShortestPathHeuristicTask):
    def __init__(self, mobile_device):
        BaseTask.__init__(self, mobile_device)

    def response_to_in_edge_computing_node(self):
        ecn = self.next_sub_task_parameter_dict["edge_computing_node"]
        cloud = ecn.cloud

        vm_list = TaskUtility.sort_virtual_machine_by_end_time_and_transfer_time(self.simulator, ecn, self)
        vm = vm_list[0]

        if vm.get_free_time() + self.get_remaining_flops_size() / vm.flops > \
                cloud.get_free_time() + self.get_remaining_flops_size() / cloud.flops:
            TaskUtility.transfer_task_to_cloud(self, cloud)
        else:
            if vm in ecn.virtual_machine_list:
                self._assign_to_vm(vm)
            else:
                TaskUtility.transfer_task_to_edge_computing_node_virtual_machine(self, vm.edge_computing_node, vm)

    def response_in_transit_to_edge_computing_node(self):
        current_ecn = self.next_sub_task_parameter_dict["current_edge_computing_node"]
        target_ecn = self.next_sub_task_parameter_dict["target_edge_computing_node"]
        path = self.next_sub_task_parameter_dict["path"]
        vm = self.next_sub_task_parameter_dict["virtual_machine"]

        if current_ecn == target_ecn:
            self._assign_to_vm(vm)
        else:
            next_ecn_index = path.index(current_ecn) + 1
            next_ecn = path[next_ecn_index]

            if self.end_time < current_ecn.edge_computing_node_network[next_ecn].end_time:
                self.end_time = current_ecn.edge_computing_node_network[next_ecn].end_time
                self.simulator.add_log({"log_type": TaskLogType.WAIT_FOR_INTER_EDGE_COMPUTING_NODE_NETWORK,
                                        "end_time": self.end_time}, self)
            else:
                TaskUtility.transfer_task_between_edge_computing_nodes(self, current_ecn, next_ecn)
                self.next_sub_task_parameter_dict["current_edge_computing_node"] = next_ecn

    def response_to_in_virtual_machine_state(self):
        vm = self.next_sub_task_parameter_dict["virtual_machine"]

        current_execution_time = vm.get_free_time()
        current_execution_vm = vm

        best_execution_time = current_execution_time
        best_execution_computation_unit = current_execution_vm
        best_computation_unit_transfer_speed = 0
        for computation_unit in vm.direct_computation_unit_list:
            expected_execution_time = computation_unit[0].get_free_time() + \
                                        self.get_current_data_size() / computation_unit[1] + \
                                        self.get_remaining_flops_size() / computation_unit[0].flops
            if expected_execution_time < best_execution_time:
                best_execution_time = expected_execution_time
                best_execution_computation_unit = computation_unit[0]
                best_computation_unit_transfer_speed = computation_unit[1]

        if best_execution_computation_unit != current_execution_vm and best_execution_time < current_execution_time - .01:
            self.end_time += self.get_current_data_size() / best_computation_unit_transfer_speed

            self.simulator.add_log({"log_type": TaskLogType.RELOCATE_TO_COMPUTATION_UNIT,
                                    "current_virtual_machine": current_execution_vm,
                                    "computation_unit": best_execution_computation_unit,
                                    "end_time": self.end_time}, self)

            best_execution_computation_unit.task_list.append(self)
            current_execution_vm.task_list.remove(self)

            if best_execution_computation_unit == self.simulator.cloud:
                self.next_sub_task_parameter_dict["cloud"] = best_execution_computation_unit
                self.next_sub_task_parameter_dict.pop("virtual_machine")
                self.next_sub_task_parameter_dict["cloud"].execute_layer()
                self.next_sub_task = self.response_to_in_cloud_state
            else:
                self.next_sub_task_parameter_dict["virtual_machine"] = best_execution_computation_unit
                self.next_sub_task_parameter_dict["edge_computing_node"] = best_execution_computation_unit.edge_computing_node
        else:
            vm.handle_task_layer()

    def response_to_in_cloud_state(self):
        cloud = self.next_sub_task_parameter_dict["cloud"]
        if self.end_time < cloud.end_time:
            self.end_time = cloud.end_time
            self.simulator.add_log({"log_type": TaskLogType.WAIT_FOR_CLOUD,
                                    "end_time": self.end_time}, self)
        else:
            cloud.execute_layer()

    def _assign_to_vm(self, vm):
        vm.locked = True
        self.next_sub_task_parameter_dict = {"virtual_machine": vm}
        self.next_sub_task = self.response_to_in_virtual_machine_state
        vm.insert_task_by_fairness(self)
