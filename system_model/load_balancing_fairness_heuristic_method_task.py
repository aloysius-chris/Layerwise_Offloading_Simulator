from system_model.base_task import BaseTask, TaskLogType
from system_model.shortest_path_heuristic_task import ShortestPathHeuristicTask
from system_model.task_utility import TaskUtility


class LoadBalancingFairnessHeuristicTask(ShortestPathHeuristicTask):
    """LFHOM task (offloading strategy 5).

    Edge-computing-node level (same as LBHOM):
        the task is placed on the virtual machine (or the cloud) with the
        earliest expected finish time.

    Virtual-machine level (differs from LBHOM):
        * a task is never moved to another virtual machine of the same ECN;
          it stays on the VM that the ECN assigned it to;
        * every time the VM becomes free it executes the current layer of the
          waiting task whose execution gives the highest predicted Jain's
          fairness index over the latencies of all tasks in the simulation
          (greedy, one layer at a time).
    """

    # LBHOM may move a task to a less loaded VM of the same ECN.  LFHOM keeps the
    # task on its VM and schedules its layers for fairness instead.
    ALLOW_RELOCATION_WITHIN_ECN = False

    def __init__(self, mobile_device):
        BaseTask.__init__(self, mobile_device)

    # ------------------------------------------------------------------ ECN
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

    # ------------------------------------------------------------------- VM
    def _find_better_computation_unit(self, vm):
        # LFHOM never relocates: the task stays on its originally assigned VM
        # and the fairness heuristic selects which layer to execute next.
        return None, 0

    def response_to_in_virtual_machine_state(self):
        vm = self.next_sub_task_parameter_dict["virtual_machine"]

        if self.end_time < vm.end_time:
            # the VM is busy: wait until it becomes free
            self.end_time = vm.end_time
            self.simulator.add_log({"log_type": TaskLogType.WAIT_FOR_VIRTUAL_MACHINE,
                                    "end_time": self.end_time}, self)
        else:
            # the VM is free: run the layer that gives the best latency fairness
            candidate = self._select_fairest_layer(vm)
            if candidate is None:
                vm.execute_layer()
            else:
                vm.execute_task_layer(candidate)

    def _find_better_computation_unit(self, vm):
        """Return (unit, transfer_speed) of a computation unit that would finish
        this task clearly sooner than the current VM, or (None, 0).
        VMs of the same ECN are excluded unless ALLOW_RELOCATION_WITHIN_ECN."""
        current_execution_time = vm.get_free_time()
        best_execution_time = current_execution_time
        best_unit = None
        best_transfer_speed = 0

        for unit, transfer_speed in vm.direct_computation_unit_list:
            if not self.ALLOW_RELOCATION_WITHIN_ECN and \
                    getattr(unit, "edge_computing_node", None) is vm.edge_computing_node:
                continue
            expected_execution_time = unit.get_free_time() + \
                self.get_current_data_size() / transfer_speed + \
                self.get_remaining_flops_size() / unit.flops
            if expected_execution_time < best_execution_time:
                best_execution_time = expected_execution_time
                best_unit = unit
                best_transfer_speed = transfer_speed

        if best_unit is not None and best_execution_time < current_execution_time - .01:
            return best_unit, best_transfer_speed
        return None, 0

    def _relocate_to(self, vm, unit, transfer_speed):
        self.end_time += self.get_current_data_size() / transfer_speed

        self.simulator.add_log({"log_type": TaskLogType.RELOCATE_TO_COMPUTATION_UNIT,
                                "current_virtual_machine": vm,
                                "computation_unit": unit,
                                "end_time": self.end_time}, self)

        unit.task_list.append(self)
        vm.task_list.remove(self)

        if unit == self.simulator.cloud:
            self.next_sub_task_parameter_dict["cloud"] = unit
            self.next_sub_task_parameter_dict.pop("virtual_machine")
            unit.execute_layer()
            self.next_sub_task = self.response_to_in_cloud_state
        else:
            self.next_sub_task_parameter_dict["virtual_machine"] = unit
            self.next_sub_task_parameter_dict["edge_computing_node"] = unit.edge_computing_node

    # ------------------------------------------------- fairness heuristic
    @staticmethod
    def _is_waiting_for_virtual_machine(task):
        return getattr(task.next_sub_task, "__name__", "") == "response_to_in_virtual_machine_state"

    @staticmethod
    def _estimated_latency(task, default_flops):
        """Expected final latency of a task that is not on the VM being scheduled."""
        if task.completed:
            return task.get_latency_time()
        unit = task.next_sub_task_parameter_dict.get("virtual_machine") or \
            task.next_sub_task_parameter_dict.get("cloud")
        if unit is None:
            # still on the mobile device, the distribution unit or the ECN
            return task.end_time + task.get_remaining_flops_size() / default_flops - task.start_time
        return max(task.end_time, unit.end_time) + task.get_remaining_flops_size() / unit.flops - task.start_time

    @staticmethod
    def _weighted_latency(task, latency):
        """Normalise latency by the task's total model FLOPs so that heavier
        models are expected to take proportionally longer.  The resulting value
        is latency-per-FLOP; equal values across tasks means perfect fairness."""
        w = task.get_flops_size()
        return latency / w if w > 0 else latency

    def _select_fairest_layer(self, vm):
        """Greedy heuristic.  Among the tasks waiting on this VM, choose the one
        whose current layer, if executed next, yields the highest predicted
        weighted Jain's fairness index  (sum y)^2 / (n * sum y^2)  where
        y_i = latency_i / total_flops_i  (latency per unit FLOP).
        Tasks with more FLOPs are therefore expected to take longer, so the
        heuristic naturally gives them more VM time without penalising them."""
        now = self.end_time
        waiting = [t for t in vm.task_list
                   if not t.completed and t.end_time <= now and self._is_waiting_for_virtual_machine(t)]
        if not waiting:
            return None
        if len(waiting) == 1:
            return waiting[0]

        tasks_on_vm = [t for t in vm.task_list if not t.completed]
        on_vm_ids = {id(t) for t in tasks_on_vm}

        # weighted latencies of tasks not on this VM are independent of the choice
        fixed_sum = 0.0
        fixed_sum_sq = 0.0
        for t in self.simulator.task_list:
            if id(t) in on_vm_ids:
                continue
            x = self._estimated_latency(t, vm.flops)
            y = self._weighted_latency(t, x)
            fixed_sum += y
            fixed_sum_sq += y * y
        n = len(self.simulator.task_list)

        def predicted_weighted_jfi(candidate):
            vm_end_after = max(candidate.end_time, vm.end_time) + candidate.get_current_flops_size() / vm.flops
            sum_y = fixed_sum
            sum_y2 = fixed_sum_sq
            for t in tasks_on_vm:
                if t is candidate:
                    remaining_after = t.get_remaining_flops_size() - t.get_current_flops_size()
                    x = vm_end_after + remaining_after / vm.flops - t.start_time
                else:
                    x = max(t.end_time, vm_end_after) + t.get_remaining_flops_size() / vm.flops - t.start_time
                y = self._weighted_latency(t, x)
                sum_y += y
                sum_y2 += y * y
            if sum_y2 <= 0:
                return 1.0
            return (sum_y ** 2) / (n * sum_y2)

        # ties are broken in favour of the task that has waited the longest
        return max(waiting, key=lambda t: (predicted_weighted_jfi(t), t.get_latency_time()))

    # ---------------------------------------------------------------- cloud
    def response_to_in_cloud_state(self):
        cloud = self.next_sub_task_parameter_dict["cloud"]
        if self.end_time < cloud.end_time:
            self.end_time = cloud.end_time
            self.simulator.add_log({"log_type": TaskLogType.WAIT_FOR_CLOUD,
                                    "end_time": self.end_time}, self)
        else:
            cloud.execute_layer()
