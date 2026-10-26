from system_model.base_task import BaseTask, TaskLogType
from system_model.shortest_path_heuristic_task import ShortestPathHeuristicTask
from system_model.task_utility import TaskUtility


class MinimumLatencyFairnessHeuristicTask(ShortestPathHeuristicTask):
    """LFHOM task built on top of the minimum-latency heuristic (MLHOM).

    Offloading (exactly as MinimumLatencyHeuristicTask):
        every virtual machine of the system is ranked by its expected finish
        time for this task (queued work + transfer time + execution time);
        the best one is compared with the cloud and the task is offloaded to
        whichever finishes it sooner.  On the cloud the whole task is
        executed, as in MLHOM.

    Virtual-machine execution (differs from MLHOM):
        MLHOM runs the whole task in one block.  Here the VM runs tasks layer
        by layer.  Every time the VM becomes free it applies a latency
        fairness heuristic: for each task waiting on the VM it predicts the
        final latency of every task in the system if that task's current
        layer were executed next (the chosen task advances, the other tasks
        on the VM are delayed by the layer's execution time, tasks elsewhere
        are unaffected), computes Jain's fairness index over the predicted
        latencies, and executes the layer of the task that gives the highest
        index.  A task never leaves the VM it was offloaded to.

    Latency budget (LAYER_BUDGET):
        Every reordering that improves fairness delays the other waiting
        tasks, so it raises the mean latency.  LAYER_BUDGET bounds that cost:
        a fairer task may go first only if its next layer is at most
        (1 + LAYER_BUDGET) times as long as the layer MLHOM's order would run
        (continue the task in progress, otherwise first come first served).
        LAYER_BUDGET = 0    -> mean latency equal to MLHOM (within ~1 %),
                               but almost no fairness gain;
        LAYER_BUDGET = None -> no bound, maximum fairness gain, highest
                               mean latency (default).
    """

    LAYER_BUDGET = None

    def __init__(self, mobile_device):
        BaseTask.__init__(self, mobile_device)

    # ------------------------------------------------------------------ ECN
    def response_to_in_edge_computing_node(self):
        # identical to MinimumLatencyHeuristicTask
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
    def response_to_in_virtual_machine_state(self):
        vm = self.next_sub_task_parameter_dict["virtual_machine"]

        if self.end_time < vm.end_time:
            # the VM is busy: wait until it becomes free
            self.end_time = vm.end_time
            self.simulator.add_log({"log_type": TaskLogType.WAIT_FOR_VIRTUAL_MACHINE,
                                    "end_time": self.end_time}, self)
        else:
            # the VM is free: run the layer that gives the best latency fairness
            task = self._select_fairest_layer(vm)
            if task is None:
                vm.execute_layer()
            else:
                vm.execute_task_layer(task)

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

    def _select_fairest_layer(self, vm):
        """Greedy heuristic.  Among the tasks waiting on this VM, choose the one
        whose current layer, if executed next, yields the highest predicted
        Jain's fairness index  (sum x)^2 / (n * sum x^2)  over the expected
        latencies x of all tasks.  Executing a candidate's layer delays every
        other task of this VM by that layer's execution time, which is what
        makes the candidates differ."""
        now = self.end_time
        waiting = [t for t in vm.task_list
                   if not t.completed and t.end_time <= now and self._is_waiting_for_virtual_machine(t)]
        if not waiting:
            return None
        if len(waiting) == 1:
            return waiting[0]

        tasks_on_vm = [t for t in vm.task_list if not t.completed]
        on_vm_ids = {id(t) for t in tasks_on_vm}

        # latencies of the tasks that are not on this VM do not depend on the choice
        fixed_sum = 0.0
        fixed_sum_sq = 0.0
        for t in self.simulator.task_list:
            if id(t) in on_vm_ids:
                continue
            x = self._estimated_latency(t, vm.flops)
            fixed_sum += x
            fixed_sum_sq += x * x
        n = len(self.simulator.task_list)

        def predicted_jains_fairness_index(candidate):
            vm_end_after = max(candidate.end_time, vm.end_time) + candidate.get_current_flops_size() / vm.flops
            sum_x = fixed_sum
            sum_x2 = fixed_sum_sq
            for t in tasks_on_vm:
                if t is candidate:
                    remaining_after = t.get_remaining_flops_size() - t.get_current_flops_size()
                    x = vm_end_after + remaining_after / vm.flops - t.start_time
                else:
                    x = max(t.end_time, vm_end_after) + t.get_remaining_flops_size() / vm.flops - t.start_time
                sum_x += x
                sum_x2 += x * x
            if sum_x2 <= 0:
                return 1.0
            return (sum_x ** 2) / (n * sum_x2)

        candidates = waiting
        if self.LAYER_BUDGET is not None:
            # MLHOM's order: continue the task in progress, else the first that arrived
            in_progress = [t for t in waiting if t.execution_time > 0]
            baseline = in_progress[0] if in_progress else waiting[0]
            longest_allowed = (1 + self.LAYER_BUDGET) * baseline.get_current_flops_size()
            candidates = [t for t in waiting if t.get_current_flops_size() <= longest_allowed]

        # ties are broken in favour of the task that has waited the longest
        return max(candidates, key=lambda t: (predicted_jains_fairness_index(t), t.get_latency_time()))

    # ---------------------------------------------------------------- cloud
    def response_to_in_cloud_state(self):
        # identical to MinimumLatencyHeuristicTask: the cloud executes the whole task
        cloud = self.next_sub_task_parameter_dict["cloud"]
        if self.end_time < cloud.end_time:
            self.end_time = cloud.end_time
            self.simulator.add_log({"log_type": TaskLogType.WAIT_FOR_CLOUD,
                                    "end_time": self.end_time}, self)
        else:
            cloud.execute_task(self)
