from system_model.base_task import BaseTask
from system_model.task_utility import TaskUtility


class CloudTask(BaseTask):
    def __init__(self, mobile_device):
        BaseTask.__init__(self, mobile_device)

    def response_to_in_edge_computing_node(self):
        ecn = self.next_sub_task_parameter_dict["edge_computing_node"]
        if self.end_time < ecn.cloud_network.end_time:
            self.end_time = ecn.cloud_network.end_time
        else:
            TaskUtility.transfer_task_to_cloud(self, ecn.cloud)
