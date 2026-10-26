"""CoEdge benchmark (offloading strategy 7).

Hu, L., Sun, G. and Ren, Y. (2020) 'CoEdge: Exploiting the Edge-Cloud
Collaboration for Faster Deep Learning', IEEE Access 8: 100533-41.

What CoEdge does (Section IV of the paper)
------------------------------------------
For one deep learning task, CoEdge decides *once, when the task arrives*,
how the layers of the DNN are split into consecutive segments and which
node executes each segment (the latency-minimum allocation, LMA, problem):

* Src is an ordered sequence of nodes.  It starts as <e_r, c>, where e_r is
  the edge node that received the input and c is the cloud.  Segments are
  allocated to the nodes of Src in order (Theorem 2: no "cut-in"
  allocations), so the data always flow e_r -> ... -> c.
* For a given Src, the latency-minimum partition is found by the recursion
  of Eq. (4): the first node of S executes layers L_i..L_p, the intermediate
  output of L_p is sent to the next node (time theta_p / omega), and the
  remaining layers are allocated recursively to S \\ pi(S).
* Greedy growth (Algorithm 1): every node not yet in Src is tried "right
  ahead of the cloud"; the one giving the smallest latency delta* is added
  if delta* < delta_min.  The loop stops when no node improves the latency.

Adaptation to the simulator of this thesis
------------------------------------------
* A node e of the paper is an edge computing node (ECN).  When an ECN is
  used, its segment runs on the virtual machine of that ECN that becomes
  available first; the cloud is the node c.
* d_i^j (time for node e_i to run segment L_j) is FLOPs(L_j) / flops(e_i)
  as in the paper, plus the time until the node is available, so that CoEdge
  sees the load of the node (the paper plans for a single task; in a
  multi-task system this load term is what makes the allocation sensible and
  it is the property of CoEdge used for positioning in Section 2.7).
* omega_ij, the best available bandwidth, is the ECN-ECN rate divided by the
  number of hops of the shortest path (the same gamma/h used by RRHOM), the
  intra-ECN rate between VMs of the same ECN, and the ECN-cloud rate delta
  towards the cloud.  The cloud never sends data back (it is the last node).
* The planned work is reserved on every node of the plan as soon as the
  plan is made, so tasks planned later see it in the load of those nodes
  (this makes the benchmark stronger, not weaker).
* The plan is computed once and executed without revision: segment by
  segment, each segment without interruption on its node, the intermediate
  activation being transferred between consecutive nodes.  CoEdge has no
  run-time relocation and no intra-VM fairness rule.

Because the order of Src is fixed, Eq. (4) is solved exactly with a forward
dynamic programme over the layer boundary p (O(|Src| * m) with prefix
minima), which yields the same optimum as the recursion.
"""
from system_model.base_task import BaseTask, TaskLogType
from system_model.shortest_path_heuristic_task import ShortestPathHeuristicTask
from system_model.task_utility import TaskUtility

INF = float("inf")


class CoEdgeTask(ShortestPathHeuristicTask):
    def __init__(self, mobile_device):
        BaseTask.__init__(self, mobile_device)
        self.plan = []              # list of (unit, first_layer_index, last_layer_index)
        self.segment_index = 0
        self.planned_latency = None

    # ------------------------------------------------------------ helpers
    def _layer_index(self):
        return self.layer_list.index(self.current_layer)

    def remaining_flops_on(self, unit):
        """FLOPs this task still has to execute on `unit` (for the load that
        other CoEdge tasks see when they plan)."""
        if self.completed:
            return 0
        now = self._layer_index()
        total = 0
        for u, first, last in self.plan[self.segment_index:]:
            if u is unit:
                for k in range(max(first, now), last + 1):
                    total += self.layer_list[k].flops_size
        return total

    @staticmethod
    def unit_available_time(unit):
        """Time at which `unit` has finished the work already committed to it."""
        t = unit.end_time
        for task in unit.task_list:
            if task.completed:
                continue
            if isinstance(task, CoEdgeTask):
                t += task.remaining_flops_on(unit) / unit.flops
            else:
                t += task.get_remaining_flops_size() / unit.flops
        return t

    @staticmethod
    def _bandwidth(a, b, cloud):
        """omega between two nodes (ECNs or the cloud)."""
        if b is cloud:
            return a.cloud_network.speed
        if a is b:
            return INF
        hops = max(1, len(a.shortest_path[b]) - 1)
        link = a.edge_computing_node_network[a.edge_computing_node_list[0]].speed
        return link / hops

    # --------------------------------------------------------- Eq. (4) DP
    def _best_partition(self, src, arrival, node_info):
        """Latency-minimum allocation of all layers over the ordered sequence
        `src` (first = e_r, last = cloud).  Returns (finish_time, plan)."""
        layers = self.layer_list
        m = len(layers)
        prefix = [0.0] * m                  # prefix[q] = FLOPs of layers 1..q (layer 0 = input, 0 FLOPs)
        for q in range(1, m):
            prefix[q] = prefix[q - 1] + layers[q].flops_size
        out = [layer.data_size for layer in layers]   # theta_p^out

        cloud = src[-1]
        # node 0 = e_r: the input is already there.  F[q] = finish time with layers 0..q done
        unit0, avail0, flops0 = node_info[src[0]]
        start0 = max(arrival, avail0)
        F = [start0 + prefix[q] / flops0 for q in range(m)]
        F[0] = arrival                      # e_r may execute no layer (pure relay)
        back = [[(None, q) for q in range(m)]]   # back[k][q] = (p, q): node k ran layers p+1..q
        back[0] = [(0, q) for q in range(m)]
        prev_node = src[0]
        best_end, best_k = F[m - 1], 0

        for k in range(1, len(src)):
            node = src[k]
            unit, avail, flops = node_info[node]
            bw = self._bandwidth(prev_node, node, cloud)
            newF = [INF] * m
            bk = [(None, None)] * m
            run_min, run_arg = INF, None
            for q in range(m):
                # candidate p = q-1 ... prefix minimum over p < q (node must execute >= 1 layer)
                p = q - 1
                if p >= 0 and F[p] < INF:
                    a = max(F[p] + out[p] / bw, avail) - prefix[p] / flops
                    if a < run_min:
                        run_min, run_arg = a, p
                if run_arg is not None:
                    newF[q] = run_min + prefix[q] / flops
                    bk[q] = (run_arg, q)
            F = newF
            back.append(bk)
            prev_node = node
            if F[m - 1] < best_end:
                best_end, best_k = F[m - 1], k

        # rebuild the segments of the best ending node
        plan = []
        q = m - 1
        for k in range(best_k, -1, -1):
            p, qq = back[k][q]
            if k == 0:
                first = 0
                if q >= 1:
                    plan.append((node_info[src[0]][0], 0, q))
                    # when e_r executes nothing the input is forwarded (no segment)
                break
            plan.append((node_info[src[k]][0], p + 1, q))
            q = p
        plan.reverse()
        if not plan or plan[0][1] != 0:
            # layer 0 (the model input, 0 FLOPs) travels with the first segment
            u, first, last = plan[0]
            plan[0] = (u, 0, last)
        return best_end, plan

    def _plan(self, ecn):
        sim = self.simulator
        cloud = ecn.cloud
        arrival = self.end_time
        # node information: the VM of each ECN that is available first, and the cloud
        node_info = {}
        for e in sim.edge_computing_node_list:
            vm = min(e.virtual_machine_list, key=self.unit_available_time)
            node_info[e] = (vm, self.unit_available_time(vm), vm.flops)
        node_info[cloud] = (cloud, self.unit_available_time(cloud), cloud.flops)

        # Algorithm 1
        src = [ecn, cloud]
        delta_min, best_plan = self._best_partition(src, arrival, node_info)
        candidates = [e for e in sim.edge_computing_node_list if e is not ecn]
        while candidates:
            best = None
            for e in candidates:
                trial = src[:-1] + [e, cloud]           # Src (+) e: right ahead of the cloud
                delta, plan = self._best_partition(trial, arrival, node_info)
                if best is None or delta < best[0]:
                    best = (delta, plan, e, trial)
            if best[0] < delta_min:
                delta_min, best_plan, e_star, src = best[0], best[1], best[2], best[3]
                candidates.remove(e_star)
            else:
                break
        self.plan = best_plan
        self.planned_latency = delta_min - self.start_time
        self.segment_index = 0
        # reserve the planned work on every node of the plan, so that tasks
        # planned later see it in the load of those nodes
        for unit, _, _ in self.plan:
            if self not in unit.task_list:
                unit.task_list.append(self)

    # ------------------------------------------------------ state handlers
    def response_to_in_edge_computing_node(self):
        ecn = self.next_sub_task_parameter_dict["edge_computing_node"]
        self._plan(ecn)
        unit = self.plan[0][0]
        if unit is ecn.cloud:
            self.end_time += self.get_current_data_size() / ecn.cloud_network.speed
            self._go_to_cloud(unit)
        else:
            if unit.edge_computing_node is not ecn:
                self.end_time += self.get_current_data_size() / self._bandwidth(ecn, unit.edge_computing_node, ecn.cloud)
            self._go_to_virtual_machine(unit)

    def _go_to_virtual_machine(self, vm):
        if self not in vm.task_list:
            vm.task_list.append(self)
        self.next_sub_task_parameter_dict = {"virtual_machine": vm,
                                             "edge_computing_node": vm.edge_computing_node}
        self.next_sub_task = self.response_to_in_virtual_machine_state

    def _go_to_cloud(self, cloud):
        if self not in cloud.task_list:
            cloud.task_list.append(self)
        self.next_sub_task_parameter_dict = {"cloud": cloud}
        self.next_sub_task = self.response_to_in_cloud_state

    def response_to_in_virtual_machine_state(self):
        vm = self.next_sub_task_parameter_dict["virtual_machine"]
        if self.end_time < vm.end_time:
            self.end_time = vm.end_time
            self.simulator.add_log({"log_type": TaskLogType.WAIT_FOR_VIRTUAL_MACHINE,
                                    "end_time": self.end_time}, self)
            return
        # execute the whole segment without interruption
        _, first, last = self.plan[self.segment_index]
        while not self.completed and self._layer_index() <= last:
            vm.execute_task_layer(self)
        if self.completed:
            return
        # hand the intermediate activation to the next node of the plan
        self.segment_index += 1
        nxt = self.plan[self.segment_index][0]
        cloud = self.simulator.cloud
        src_ecn = vm.edge_computing_node
        data = self.layer_list[last].data_size
        if nxt is cloud:
            self.end_time += data / src_ecn.cloud_network.speed
            self._go_to_cloud(cloud)
        else:
            bw = self._bandwidth(src_ecn, nxt.edge_computing_node, cloud)
            if bw != INF:
                self.end_time += data / bw
            self._go_to_virtual_machine(nxt)

    def response_to_in_cloud_state(self):
        cloud = self.next_sub_task_parameter_dict["cloud"]
        if self.end_time < cloud.end_time:
            self.end_time = cloud.end_time
            self.simulator.add_log({"log_type": TaskLogType.WAIT_FOR_CLOUD,
                                    "end_time": self.end_time}, self)
        else:
            cloud.execute_task(self)      # the cloud is always the last node: run the rest
