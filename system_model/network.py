from enum import Enum


class NetworkLogType(Enum):
    TRANSFER_TASK = "Transfer Task"


class Network:
    def __init__(self, target, name=None, speed=0):
        self.name = name
        self.target = target
        self.speed = speed
        self.end_time = 0

        self.log_list = []

    def transfer(self, source, task):
        start_from = max(task.end_time, self.end_time)
        self.end_time = start_from
        task.end_time = self.end_time
        task.execution_time = task.get_current_data_size() / self.speed
        end_to = self.end_time + task.execution_time
        self.end_time = end_to
        task.end_time = self.end_time

        task.simulator.add_log({"log_type": NetworkLogType.TRANSFER_TASK,
                                "task": task,
                                "start_time": start_from,
                                "end_time": end_to,
                                "source": source,
                                "target": self.target}, self)

        task.simulator.add_log({"log_type": NetworkLogType.TRANSFER_TASK,
                                "network": self,
                                "task": task,
                                "start_time": start_from,
                                "end_time": end_to,
                                "target": self.target}, task)

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
