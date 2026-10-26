from enum import Enum
from system_model.base_task import TaskLogType


class VirtualMachineLogType(Enum):
    EXECUTE_TASK = "Execute Task in Virtual Machine"
    EXECUTE_TASK_LAYER = "Execute Task Layer"
    STOP_TASK = "Stop Task in Virtual Machine"


class VirtualMachine:
    """A virtual machine of an edge computing node.

    Time model
    ----------
    ``end_time`` is the simulated time at which the VM becomes free.  Work is
    appended to the VM's timeline: a task (or a layer of a task) starts at
    ``max(task.end_time, vm.end_time)`` - i.e. when both the task has arrived
    and the VM is free - and the VM's ``end_time`` is advanced by the
    execution time  FLOPs / flops.

    Task bookkeeping
    ----------------
    ``task_list`` holds every task that was offloaded to this VM (finished
    tasks stay in the list).  Each task keeps its own state:
    ``current_layer`` (the next layer to execute), ``end_time`` (the time it
    is available again), ``execution_time`` (time executed so far),
    ``remaining_time``, ``completed`` and ``next_sub_task`` (the handler the
    simulation calls when the task's turn comes).

    Two ways of serving the tasks (chosen by the simulation loop)
    -------------------------------------------------------------
    * ``handle_task(task)``       - execute all layers of that task in one go
                                    (MLHOM and the other baseline methods).
    * ``handle_task_layer()``     - execute ONE layer: the VM itself picks,
                                    from its task list, the waiting task
                                    with the maximum expected latency and
                                    runs its current layer (LFHOM).  Serving
                                    the task that would otherwise finish last
                                    reduces the largest latencies and so
                                    improves the latency fairness (Jain's
                                    index) of the whole system.
    """

    def __init__(self):
        self.name = ""
        self.flops = 0
        self.edge_computing_node = None
        self.task_list = []
        self.end_time = 0
        self.booked = False
        self.executed_flops = 0
        self.direct_computation_unit_list = []

        self.log_list = []

    # ------------------------------------------------------------------
    # Whole-task execution (MLHOM, HOM, ...)
    # ------------------------------------------------------------------
    def handle_task(self, task):
        """Execute ALL remaining layers of `task` on this VM, one layer after
        the other, with no other task interleaved.

        The first layer starts when both the task has arrived and the VM is
        free (``max(task.end_time, vm.end_time)``); every following layer
        starts when the previous one ends.  After the call the task is
        completed, its ``end_time`` is its completion time and the VM's
        ``end_time`` has advanced by the whole execution time of the task.

        Using ``execute_task_layer`` for every layer (instead of
        ``execute_task``) keeps the per-layer bookkeeping - ``current_layer``,
        ``execution_time``, ``remaining_time``, logs - identical to the
        layer-by-layer execution used by ``handle_task_layer``, so the two
        methods can be compared on equal terms."""
        if task.end_time < self.end_time:
            # the VM is busy: the task waits until the VM is free
            task.simulator.add_log({"log_type": TaskLogType.WAIT_FOR_VIRTUAL_MACHINE,
                                    "end_time": self.end_time}, task)
        while not task.completed:
            self.execute_task_layer(task)

    def execute_task(self, task):
        """Execute the whole task as one block (original implementation, kept
        for the task classes that call it directly).  Unlike ``handle_task``
        it does not advance ``current_layer``."""
        start_from = max(task.end_time, self.end_time)
        self.end_time = start_from
        task.end_time = self.end_time
        task.execution_time = task.get_remaining_flops_size() / self.flops
        self.executed_flops += task.get_remaining_flops_size()
        end_to = self.end_time + task.execution_time
        self.end_time = end_to
        task.end_time = self.end_time
        task.completed = True
        self.booked = False

        task.simulator.add_log({"log_type": VirtualMachineLogType.EXECUTE_TASK,
                                "task": task,
                                "start_time": start_from,
                                "end_time": end_to}, self)
        task.simulator.add_log({"log_type": VirtualMachineLogType.EXECUTE_TASK,
                                "virtual_machine": self,
                                "start_time": start_from,
                                "end_time": end_to}, task)

    # ------------------------------------------------------------------
    # Layer-by-layer execution
    # ------------------------------------------------------------------
    def execute_layer(self):
        """Execute the current layer of the unfinished task on this VM that
        has the highest latency so far (``end_time - start_time``).
        Original implementation, used by the load-balancing methods."""
        if len([task for task in self.task_list if not task.completed]) > 0:
            task = [task for task in self.task_list if not task.completed][0]
            for exiting_task in self.task_list:
                if not exiting_task.completed and exiting_task.get_latency_time() > task.get_latency_time():
                    task = exiting_task

            self.end_time = max(task.end_time, self.end_time)
            start_from = self.end_time
            self.end_time += task.get_current_flops_size() / self.flops
            task.execution_time += task.get_current_flops_size() / self.flops
            self.executed_flops += task.get_current_flops_size()
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
                task.next_sub_task = task.response_to_in_virtual_machine_state

            task.simulator.add_log({"log_type": VirtualMachineLogType.EXECUTE_TASK_LAYER,
                                    "task": task,
                                    "start_time": start_from,
                                    "end_time": end_to,
                                    "layer": current_layer}, self)
            task.simulator.add_log({"log_type": VirtualMachineLogType.EXECUTE_TASK_LAYER,
                                    "virtual_machine": self,
                                    "start_time": start_from,
                                    "end_time": end_to,
                                    "layer": current_layer}, task)

    def execute_task_layer(self, task):
        """Execute the CURRENT layer of `task` on this VM and update the
        task so that its next layer is the one to execute afterwards.

        Timing
            start = max(task.end_time, vm.end_time)   (task arrived, VM free)
            end   = start + current_layer.flops_size / vm.flops
            vm.end_time = end          (the VM is busy until then)
            task.end_time = end        (the task is available again then)

        Task properties updated
            execution_time  += layer execution time
            remaining_time   = time still needed for the layers after this one
            current_layer    -> next layer of the task  (move_to_next_layer)
            next_sub_task    -> response_to_in_virtual_machine_state, so the
                                task returns to this VM's queue for its next
                                layer; on the LAST layer the task is marked
                                completed and next_sub_task is set to None.

        VM properties updated
            executed_flops  += layer FLOPs   (used by the load-balancing metric)
            booked           = False

        Two log entries are written (one from the VM's point of view, one
        from the task's), which is why a layer appears twice in the
        simulator's global log list."""
        self.end_time = max(task.end_time, self.end_time)
        start_from = self.end_time
        self.end_time += task.get_current_flops_size() / self.flops
        task.execution_time += task.get_current_flops_size() / self.flops
        self.executed_flops += task.get_current_flops_size()
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
            task.next_sub_task = task.response_to_in_virtual_machine_state

        task.simulator.add_log({"log_type": VirtualMachineLogType.EXECUTE_TASK_LAYER,
                                "task": task,
                                "start_time": start_from,
                                "end_time": end_to,
                                "layer": current_layer}, self)
        task.simulator.add_log({"log_type": VirtualMachineLogType.EXECUTE_TASK_LAYER,
                                "virtual_machine": self,
                                "start_time": start_from,
                                "end_time": end_to,
                                "layer": current_layer}, task)

    def get_free_time(self):
        """Time at which the VM would be free if it executed all the remaining
        work of the tasks in its list (used by the offloading decision)."""
        free_time = self.end_time
        for task in self.task_list:
            free_time += task.get_remaining_flops_size() / self.flops
        return free_time

    @staticmethod
    def _estimated_latency_elsewhere(task, default_flops):
        """Expected final latency of a task that is NOT on this VM.
        Uses the task's own assigned resource (VM or cloud) when available,
        otherwise falls back to default_flops as a rough estimate."""
        if task.completed:
            return task.get_latency_time()
        unit = task.next_sub_task_parameter_dict.get("virtual_machine") or \
            task.next_sub_task_parameter_dict.get("cloud")
        if unit is None:
            return task.end_time + task.get_remaining_flops_size() / default_flops - task.start_time
        return max(task.end_time, unit.end_time) + task.get_remaining_flops_size() / unit.flops - task.start_time

    def insert_task_by_fairness(self, task):
        """Insert `task` into task_list at the position that gives the highest
        predicted Jain's fairness index over all task latencies in the simulation.

        For each candidate position the method assumes the VM will process its
        queue in list order (FIFO), with each task running all remaining FLOPs
        in one block.  Tasks outside this VM are estimated with their own
        resource's end_time.  The position with the highest predicted JFI wins.

        Falls back to a plain append when the simulator is not yet attached to
        the task or the task_list is empty."""
        simulator = getattr(task, 'simulator', None)
        if simulator is None or not self.task_list:
            self.task_list.append(task)
            return

        n = len(simulator.task_list)
        if n == 0:
            self.task_list.append(task)
            return

        # Fixed part: expected latencies of tasks NOT on this VM
        on_vm_ids = {id(t) for t in self.task_list}
        on_vm_ids.add(id(task))
        fixed_sum = 0.0
        fixed_sum_sq = 0.0
        for t in simulator.task_list:
            if id(t) in on_vm_ids:
                continue
            x = self._estimated_latency_elsewhere(t, self.flops)
            fixed_sum += x
            fixed_sum_sq += x * x

        best_pos = len(self.task_list)   # default: append
        best_jfi = -1.0

        for pos in range(len(self.task_list) + 1):
            # build the hypothetical FIFO queue with task inserted at pos
            ordered = list(self.task_list)
            ordered.insert(pos, task)

            sum_x = fixed_sum
            sum_x2 = fixed_sum_sq
            cursor = self.end_time
            for t in ordered:
                start = max(t.end_time, cursor)
                finish = start + t.get_remaining_flops_size() / self.flops
                cursor = finish
                latency = finish - t.start_time
                sum_x += latency
                sum_x2 += latency * latency

            if sum_x2 > 0:
                jfi = (sum_x ** 2) / (n * sum_x2)
                if jfi > best_jfi:
                    best_jfi = jfi
                    best_pos = pos

        self.task_list.insert(best_pos, task)

    # ------------------------------------------------------------------
    # Latency-fair layer execution (LFHOM)
    # ------------------------------------------------------------------
    def handle_task_layer(self):
        """Execute ONE layer on this VM: the current layer of the waiting task
        with the maximum expected latency.  The VM picks the task itself.

        Called by the simulation loop when the next event of the simulation
        is a task waiting on this VM.  No task is passed in: the VM works from
        its own ``task_list``.

        1. Current time.  The tasks on this VM that are waiting for it are
           ``pending``.  The loop calls this method only when the earliest
           event of the whole simulation is one of them, so the current time
           is  now = min(pending end_time).
        2. VM busy (``now < vm.end_time``).  Every pending task that has
           already arrived waits until the VM is free: its ``end_time`` is
           moved to ``vm.end_time`` and nothing is executed.  Their events
           fire again at that moment.
        3. VM free.  Among the tasks that have arrived (``get_waiting_tasks``)
           the VM computes for each one the latency it would end with if it
           ran to completion here (``expected_latency``) and executes the
           current layer of the task with the MAXIMUM expected latency
           (ties: the task that has waited longest).  ``execute_task_layer``
           updates that task's properties and sets its next layer.
        4. Only one layer is executed per call.  The tasks that were not
           chosen keep their event time, so the loop calls this method again
           as soon as the VM is free and the choice is re-evaluated with the
           updated state.

        Executing first the task that is expected to finish last shortens the
        largest latencies, which raises the latency fairness (Jain's index)
        of the system - at the price of a higher mean latency, because short
        tasks wait behind long ones.

        `simulator` is optional and only used to write the wait log entries;
        when omitted the tasks' own simulator reference is used."""
        # 1. the tasks waiting for this VM and the current time
        pending = [t for t in self.task_list
                   if not t.completed
                   and getattr(t.next_sub_task, "__name__", "") == "response_to_in_virtual_machine_state"]
        if not pending:
            return
        now = min(t.end_time for t in pending)

        # 2. the VM is busy: the tasks that have arrived wait until it is free
        if now < self.end_time:
            for t in pending:
                if t.end_time < self.end_time:
                    t.end_time = self.end_time
                    t.simulator.add_log({"log_type": TaskLogType.WAIT_FOR_VIRTUAL_MACHINE,
                                        "end_time": t.end_time}, t)
            return

        # 3. the VM is free: select the arrived task with the maximum expected
        #    latency and execute its current layer (this also sets its next layer)
        waiting = self.get_waiting_tasks(now)
        selected = max(waiting, key=lambda t: (self.expected_latency(t), t.get_latency_time()))
        self.execute_task_layer(selected)

    def get_waiting_tasks(self, now):
        """The tasks that are waiting for this VM at time `now`: on this VM,
        not completed, already arrived (``end_time <= now``) and in the
        'waiting for the virtual machine' state.  Tasks still in transit to
        this VM are not included."""
        return [t for t in self.task_list
                if not t.completed and t.end_time <= now
                and getattr(t.next_sub_task, "__name__", "") == "response_to_in_virtual_machine_state"]

    def expected_latency(self, task):
        """Latency the task would end with if it ran to completion on this VM
        starting as soon as it and the VM are available:
            max(task.end_time, vm.end_time) + remaining FLOPs / vm.flops - task.start_time"""
        return max(task.end_time, self.end_time) + task.get_remaining_flops_size() / self.flops - task.start_time

    def __str__(self):
        return self.name
