from enum import Enum


class CloudLogType(Enum):
    EXECUTE_TASK = "Execute Task in Cloud"
    EXECUTE_TASK_LAYER = "Execute Task Layer"


class Cloud:
    def __init__(self):
        self.name = None
        self.flops = 0
        self.edge_computing_node_list = []
        self.task_list = []
        self.end_time = 0
        self.booked = False

        self.log_list = []

    def execute_task(self, task):
        start_from = max(task.end_time, self.end_time)
        self.end_time = start_from
        task.end_time = self.end_time
        task.execution_time = task.get_remaining_flops_size() / self.flops
        end_to = self.end_time + task.execution_time
        self.end_time = end_to
        task.end_time = self.end_time

        task.completed = True

        task.simulator.add_log({"log_type": CloudLogType.EXECUTE_TASK,
                                "task": task,
                                "start_time": start_from,
                                "end_time": end_to}, self)
        task.simulator.add_log({"log_type": CloudLogType.EXECUTE_TASK,
                                "cloud": self,
                                "start_time": start_from,
                                "end_time": end_to}, task)

    def execute_layer(self):
        task = [task for task in self.task_list if not task.completed][0]
        for exiting_task in self.task_list:
            if not exiting_task.completed and exiting_task.get_latency_time() > task.get_latency_time():
                task = exiting_task

        self.end_time = max(task.end_time, self.end_time)
        start_from = self.end_time
        self.end_time += task.get_current_flops_size() / self.flops
        task.execution_time += task.get_current_flops_size() / self.flops
        task.remaining_time = (task.get_remaining_flops_size() - task.get_current_flops_size()) / self.flops
        task.end_time = self.end_time
        end_to = self.end_time
        self.booked = False

        current_layer = task.current_layer

        if task.is_last_layer():
            task.completed = True
            task.next_sub_task = None
        else:
            task.move_to_next_layer()
            task.next_sub_task = task.response_to_in_cloud_state

        task.simulator.add_log({"log_type": CloudLogType.EXECUTE_TASK,
                                "task": task,
                                "start_time": start_from,
                                "end_time": end_to,
                                "layer": current_layer}, self)
        task.simulator.add_log({"log_type": CloudLogType.EXECUTE_TASK,
                                "cloud": self,
                                "start_time": start_from,
                                "end_time": end_to,
                                "layer": current_layer}, task)

    def get_free_time(self):
        free_time = self.end_time
        for task in self.task_list:
            free_time += task.get_remaining_flops_size() / self.flops
        return free_time

    def __str__(self):
        return self.name
