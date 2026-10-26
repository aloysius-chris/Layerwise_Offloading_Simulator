from system_model.base_task import BaseTask, TaskLogType
from system_model.shortest_path_heuristic_task import ShortestPathHeuristicTask
from system_model.task_utility import TaskUtility


class MinimumLatencyHeuristicTask(ShortestPathHeuristicTask):
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
                TaskUtility.transfer_task_to_virtual_machine(self, vm)
            else:
                TaskUtility.transfer_task_to_edge_computing_node_virtual_machine(self, vm.edge_computing_node, vm)

    def response_to_in_virtual_machine_state(self):
        vm = self.next_sub_task_parameter_dict["virtual_machine"]
        if self.end_time < vm.end_time:
            self.end_time = vm.end_time
            self.simulator.add_log({"log_type": TaskLogType.WAIT_FOR_VIRTUAL_MACHINE,
                                    "end_time": self.end_time}, self)
        else:
            vm.execute_task(self)

    def response_to_in_cloud_state(self):
        cloud = self.next_sub_task_parameter_dict["cloud"]
        if self.end_time < cloud.end_time:
            self.end_time = cloud.end_time
            self.simulator.add_log({"log_type": TaskLogType.WAIT_FOR_CLOUD,
                                    "end_time": self.end_time}, self)
        else:
            cloud.execute_task(self)
