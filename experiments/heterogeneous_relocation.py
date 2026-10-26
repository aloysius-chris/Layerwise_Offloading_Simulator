"""Supplementary experiment: RRHOM with the relocation test of Algorithm 2 actually evaluated, on identical
and on heterogeneous VM capacities.  The team's simulator is imported unchanged; only two things are added
here, and only for the methods that need them:

* heterogeneous capacities: after the system is built, the VMs of every ECN get 1, 1.5, 2, 2.5 and 3 GFLOP/s
  (mean 2 GFLOP/s, so the total edge capacity equals the homogeneous setting);
* RRHOM-R (relocation enabled): a VM executes the task it has started one layer at a time without
  interleaving other tasks, and after every layer it applies the net-gain test of Eq. (5):
      stay  = C_res / P_v
      move  = R_u + s_k / B_{v,u} + C_res / P_u      for every other VM u and the cloud
      relocate iff  min move < stay - eps_r,  eps_r = 0.01 s (the threshold of the team's code)
  R_u is the backlog of u at the time of the test (remaining work of its unfinished tasks plus the rest
  of its current busy period).

Methods: MLHOM (strategy 4); RRHOM-L (strategy 6 executed layer by layer with the started task keeping its VM,
relocation test switched off); RRHOM-R (the same with the relocation test).  RRHOM-R minus RRHOM-L isolates the
effect of relocation.  Output: one CSV row per run."""
import contextlib, csv, io, os, sys, time
from multiprocessing import Pool

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heterogeneous_raw.csv")
EPS = 0.01
HETERO = [1.0e9, 1.5e9, 2.0e9, 2.5e9, 3.0e9]
STRATEGY = {"MLHOM": 4, "RRHOM": 6, "RRHOM-L": 6, "RRHOM-R": 6}
FIELDS = ["setting", "x", "method", "seed", "latency", "weighted_latency", "jain", "weighted_jfi", "utilization",
          "load_variance", "relocations", "relocations_to_vm", "relocations_to_cloud", "data_moved_mb", "seconds"]


def _setup():
    sys.path[:0] = [CODE, os.path.join(CODE, "results")]
    os.chdir(CODE)


def run_case(args):
    setting, md, seed, method = args
    _setup()
    import settings
    from simulator import Simulation
    from system_model.virtual_machine import VirtualMachine
    from system_model.base_task import TaskLogType

    stats = {"vm": 0, "cloud": 0, "bytes": 0.0}
    original = VirtualMachine.handle_task

    def handle_task_with_relocation(self, task):
        owner = getattr(self, "owner", None)
        if owner is not None and owner is not task and not owner.completed:
            # the VM is executing another task: wait just after its next layer boundary
            task.end_time = max(task.end_time, self.end_time) + 1e-9
            return
        self.owner = task
        self.execute_task_layer(task)
        if task.completed:
            self.owner = None
            return
        if method != "RRHOM-R":
            return
        now = task.end_time
        remaining = task.get_remaining_flops_size()
        data = task.get_current_data_size()
        stay = remaining / self.flops
        best, best_time = None, stay
        for unit, rate in self.direct_computation_unit_list:
            backlog = max(unit.end_time - now, 0.0) + sum(
                t.get_remaining_flops_size() for t in unit.task_list if not t.completed) / unit.flops
            move = backlog + data / rate + remaining / unit.flops
            if move < best_time:
                best, best_time = (unit, rate), move
        if best is None or best_time >= stay - EPS:
            return
        unit, rate = best
        self.owner = None
        task.end_time += data / rate
        self.task_list.remove(task)
        unit.task_list.append(task)
        task.simulator.add_log({"log_type": TaskLogType.RELOCATE_TO_COMPUTATION_UNIT,
                                "current_virtual_machine": self, "computation_unit": unit,
                                "end_time": task.end_time}, task)
        stats["bytes"] += data
        if unit is task.simulator.cloud:
            stats["cloud"] += 1
            task.next_sub_task_parameter_dict = {"cloud": unit}
            task.next_sub_task = task.response_to_in_cloud_state
        else:
            stats["vm"] += 1
            task.next_sub_task_parameter_dict = {"virtual_machine": unit,
                                                 "edge_computing_node": unit.edge_computing_node}
            task.next_sub_task = task.response_to_in_virtual_machine_state

    VirtualMachine.handle_task = handle_task_with_relocation if method in ("RRHOM-L", "RRHOM-R") else original
    vm_count = 5
    sim = Simulation(layer_list=settings.algo.layer_list, vm_flops=settings.vm_flops,
                     cloud_flops=settings.cloud_flops, vm_count=vm_count, ecn_count=settings.ecn_count,
                     du_count=settings.du_count, md_count=md)
    if setting == "heterogeneous":
        for ecn in sim.edge_computing_node_list:
            for vm, f in zip(ecn.virtual_machine_list, HETERO):
                vm.flops = f
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        sim.start(offloading_strategy=STRATEGY[method], random_state=seed)
    VirtualMachine.handle_task = original
    reloc = sum(1 for t in sim.task_list for log in t.log_list
                if isinstance(log, dict) and log.get("log_type") == TaskLogType.RELOCATE_TO_COMPUTATION_UNIT)
    return {"setting": setting, "x": md * settings.ecn_count * settings.du_count, "method": method, "seed": seed,
            "latency": sim.get_latency(), "weighted_latency": sim.get_weighted_latency(),
            "jain": sim.get_jains_fairness_index(), "weighted_jfi": sim.get_weighted_jains_fairness_index(),
            "utilization": sim.get_resources_utilization(), "load_variance": sim.get_load_balancing(),
            "relocations": reloc, "relocations_to_vm": stats["vm"], "relocations_to_cloud": stats["cloud"],
            "data_moved_mb": stats["bytes"] / 2 ** 20, "seconds": round(time.time() - t0, 1)}


def main():
    mode = sys.argv[1]
    if mode == "test":
        for case in [("homogeneous", 6, 1, "RRHOM"), ("homogeneous", 6, 1, "RRHOM-R"), ("heterogeneous", 6, 1, "RRHOM-R")]:
            print(run_case(case), flush=True)
        return
    cases = []
    for seed in range(1, 31):
        for md in (2, 4, 6, 8, 10):
            for method in ("MLHOM", "RRHOM-L", "RRHOM-R"):
                cases.append(("heterogeneous", md, seed, method))
            cases.append(("homogeneous", md, seed, "RRHOM-R"))
    done = set()
    if os.path.exists(OUT):
        with open(OUT, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["setting"], str(int(r["x"]) // 50), r["seed"], r["method"]))
    todo = [c for c in cases if (c[0], str(c[1]), str(c[2]), c[3]) not in done]
    # longest runs first so the pool finishes evenly
    todo.sort(key=lambda c: (c[3] == "MLHOM", -c[1]))
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f, Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 7) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap_unordered(run_case, todo), 1):
            w.writerow(row)
            f.flush()
            print(i, "/", len(todo), row["setting"], row["x"], row["method"], row["seed"], row["seconds"], flush=True)


if __name__ == "__main__":
    main()
