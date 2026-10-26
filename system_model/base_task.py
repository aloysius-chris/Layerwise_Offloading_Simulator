from enum import Enum
from system_model.task_utility import  TaskUtility


class TaskLogType(Enum):
    WAIT_FOR_DISTRIBUTION_UNIT_NETWORK = "Wait for Distribution Unit Network"
    WAIT_FOR_EDGE_COMPUTING_NODE_NETWORK = "Wait for Edge Computing Node Network"
    WAIT_FOR_INTER_EDGE_COMPUTING_NODE_NETWORK = "Wait for Inter Edge Computing Node Network"
    WAIT_FOR_VIRTUAL_MACHINE = "Wait for Virtual Machine"
    WAIT_FOR_CLOUD = "Wait for Virtual Machine"
    RELOCATE_TO_COMPUTATION_UNIT = "Relocate to Computation Unit"


class BaseTask:
    def __init__(self, mobile_device):
        self.simulator = None
        self.name = None
        self.layer_list = None
        self.current_layer = None
        self.mobile_device = mobile_device
        self.start_time = 0
        self.execution_time = 0
        self.remaining_time = -1
        self.end_time = 0
        self.in_transit = False
        self.completed = False

        self.log_list = []

        self.next_sub_task_parameter_dict = {}
        self.next_sub_task = self.response_to_in_mobile_device_state

    def set_simulation(self, simulation):
        self.simulator = simulation

    def get_data_size(self):
        return self.layer_list[0].data_size

    def get_current_data_size(self):
        return self.current_layer.data_size

    def get_flops_size(self):
        flops_size = 0
        for layer in self.layer_list:
            flops_size += layer.flops_size
        return flops_size

    def get_current_flops_size(self):
        return self.current_layer.flops_size

    def get_remaining_flops_size(self):
        flops_size = 0
        start = False
        for layer in self.layer_list:
            if layer == self.current_layer:
                start = True
            if start:
                flops_size += layer.flops_size
        return flops_size

    def get_current_layer(self):
        return self.current_layer.flops_size

    def move_to_next_layer(self):
        self.current_layer = self.layer_list[self.layer_list.index(self.current_layer) + 1]

    def is_last_layer(self):
        return self.current_layer == self.layer_list[-1]

    def get_executed_time(self, time):
        return time - (self.end_time - self.execution_time)

    def set_current_layer_to_time(self, time):
        target_ratio = (time - (self.end_time - self.execution_time)) / self.execution_time

        flops_size = self.get_flops_size()

        ratio = 0
        for layer in self.layer_list:
            if ratio + layer.flops_size / flops_size > target_ratio:
                self.current_layer = layer
                break
            ratio += layer.flops_size / flops_size

    def get_latency_time(self):
        return self.end_time - self.start_time

    def get_expected_latency(self):
        return self.end_time + self.remaining_time - self.start_time

    def response_to_in_mobile_device_state(self):
        if self.end_time < self.mobile_device.distribution_unit_network.end_time:
            self.end_time = self.mobile_device.distribution_unit_network.end_time
            self.simulator.add_log({"log_type": TaskLogType.WAIT_FOR_DISTRIBUTION_UNIT_NETWORK,
                                    "end_time": self.end_time}, self)
        else:
            TaskUtility.transfer_task_to_distribution_unit(self)

    def response_to_in_distribution_unit_state(self):
        if self.end_time < self.mobile_device.distribution_unit.edge_computing_node_network.end_time:
            self.end_time = self.mobile_device.distribution_unit.edge_computing_node_network.end_time
            self.simulator.add_log({"log_type": TaskLogType.WAIT_FOR_EDGE_COMPUTING_NODE_NETWORK,
                                    "end_time": self.end_time}, self)
        else:
            TaskUtility.transfer_task_to_edge_computing(self)

    def response_to_in_edge_computing_node(self):
        pass

    def response_to_in_virtual_machine_state(self):
        if self.end_time < self.next_sub_task_parameter_dict["virtual_machine"].end_time:
            self.end_time = self.next_sub_task_parameter_dict["virtual_machine"].end_time
            self.simulator.add_log({"log_type": TaskLogType.WAIT_FOR_VIRTUAL_MACHINE,
                                    "end_time": self.end_time}, self)
        else:
            self.next_sub_task_parameter_dict["virtual_machine"].execute_task(self)

    def response_to_in_cloud_state(self):
        if self.end_time < self.next_sub_task_parameter_dict["cloud"].end_time:
            self.end_time = self.next_sub_task_parameter_dict["cloud"].end_time
            self.simulator.add_log({"log_type": TaskLogType.WAIT_FOR_CLOUD,
                                    "end_time": self.end_time}, self)
        else:
            self.next_sub_task_parameter_dict["cloud"].execute_task(self)

    def execute_next_sub_task(self):
        self.next_sub_task()

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

    def __str__(self):
        return self.name
