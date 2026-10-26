from system_model.base_task import BaseTask, TaskLogType
from system_model.task_utility import TaskUtility


class LocalTask(BaseTask):
    def __init__(self, mobile_device):
        BaseTask.__init__(self, mobile_device)

    def response_to_in_edge_computing_node(self):
        vm_list = TaskUtility.sort_virtual_machine_list_in_edge_computing_node_by_end_time(
            self.next_sub_task_parameter_dict["edge_computing_node"])
        if len(vm_list) > 0:
            vm = vm_list[0]

            if self.end_time < vm.end_time:
                self.end_time = vm.end_time
                self.simulator.add_log({"log_type": TaskLogType.WAIT_FOR_VIRTUAL_MACHINE,
                                        "end_time": self.end_time}, self)
            else:
                TaskUtility.transfer_task_to_virtual_machine(self, vm)
