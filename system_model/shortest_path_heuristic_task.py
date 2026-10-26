from system_model.base_task import BaseTask, TaskLogType
from system_model.task_utility import TaskUtility


class ShortestPathHeuristicTask(BaseTask):
    def __init__(self, mobile_device):
        BaseTask.__init__(self, mobile_device)

    def response_to_in_edge_computing_node(self):
        ecn = self.next_sub_task_parameter_dict["edge_computing_node"]
        local_vm_list = TaskUtility.sort_free_virtual_machine_list_in_edge_computing_node_by_end_time(
            ecn,
            self.end_time)

        if len(local_vm_list) > 0:
            TaskUtility.transfer_task_to_virtual_machine(self, local_vm_list[0])
        else:
            vm_list = TaskUtility.sort_free_virtual_machine_list_by_distance_from_edge_computing_node(
                self.simulator,
                ecn,
                self.end_time)

            if len(vm_list) == 0:
                TaskUtility.transfer_task_to_cloud(self, ecn.cloud)
            else:
                TaskUtility.transfer_task_to_edge_computing_node_virtual_machine(self, ecn, vm_list[0])

    def response_in_transit_to_edge_computing_node(self):
        current_ecn = self.next_sub_task_parameter_dict["current_edge_computing_node"]
        target_ecn = self.next_sub_task_parameter_dict["target_edge_computing_node"]
        path = self.next_sub_task_parameter_dict["path"]
        vm = self.next_sub_task_parameter_dict["virtual_machine"]
        if current_ecn == target_ecn:
            TaskUtility.transfer_task_to_virtual_machine(self, vm)
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
