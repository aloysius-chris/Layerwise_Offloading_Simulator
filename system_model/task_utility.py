class TaskUtility:
    @staticmethod
    def sort_virtual_machine_list_in_edge_computing_node_by_end_time(ecn):
        vm_list = []
        for vm in ecn.virtual_machine_list:
            if not vm.booked:
                vm_list.append(vm)
        vm_list.sort(key=lambda x: x.end_time)
        return vm_list

    @staticmethod
    def sort_free_virtual_machine_list_in_edge_computing_node_by_end_time(ecn, time):
        vm_list = []
        for vm in ecn.virtual_machine_list:
            if not vm.booked and vm.end_time <= time:
                vm_list.append(vm)
        vm_list.sort(key=lambda x: x.end_time)
        return vm_list

    @staticmethod
    def sort_free_virtual_machine_list_by_distance_from_edge_computing_node(simulator, source_ecn, time):
        vm_list = []
        for ecn in simulator.edge_computing_node_list:
            for vm in ecn.virtual_machine_list:
                if not vm.booked and vm.end_time <= time:
                    vm_list.append(vm)
        vm_list.sort(key=lambda x: len(x.edge_computing_node.shortest_path[source_ecn]))
        return vm_list

    @staticmethod
    def sort_virtual_machine_by_end_time_and_transfer_time(simulator, current_ecn, task):
        vm_list = []
        for ecn in simulator.edge_computing_node_list:
            for vm in ecn.virtual_machine_list:
                vm_list.append(vm)
        vm_list.sort(key=lambda x: x.get_free_time() +
                                   TaskUtility.get_time_to_transfer_task_to_virtual_machine(task, x) +
                                    task.get_remaining_flops_size() / x.flops
                     )
        return vm_list

    @staticmethod
    def get_time_to_transfer_task_to_virtual_machine(task, vm):
        ecn = task.next_sub_task_parameter_dict["edge_computing_node"]
        distance = len(ecn.shortest_path[vm.edge_computing_node])
        if vm in ecn.virtual_machine_list:
            return 0
        else:
            return task.get_current_data_size() * distance / \
                ecn.edge_computing_node_network[ecn.edge_computing_node_list[0]].speed

    @staticmethod
    def get_time_to_transfer_task_to_cloud(task):
        ecn = task.next_sub_task_parameter_dict["edge_computing_node"]
        return task.get_current_data_size() / ecn.cloud_network.speed

    @staticmethod
    def transfer_task_to_virtual_machine(task, vm):
        vm.locked = True
        task.next_sub_task_parameter_dict = {"virtual_machine": vm}
        task.next_sub_task = task.response_to_in_virtual_machine_state
        vm.task_list.append(task)

    @staticmethod
    def transfer_task_to_cloud(task, cloud):
        cloud.locked = True
        task.next_sub_task_parameter_dict = {"cloud": cloud}
        task.next_sub_task = task.response_to_in_cloud_state
        cloud.task_list.append(task)

    @staticmethod
    def transfer_task_to_edge_computing_node_virtual_machine(task, current_ecn, vm):
        vm.booked = True
        task.next_sub_task_parameter_dict = {
            "current_edge_computing_node": current_ecn,
            "path": current_ecn.shortest_path[vm.edge_computing_node],
            "target_edge_computing_node": vm.edge_computing_node,
            "virtual_machine": vm}
        task.next_sub_task = task.response_in_transit_to_edge_computing_node
        task.in_transit = True

    @staticmethod
    def transfer_task_to_distribution_unit(task):
        task.mobile_device.distribution_unit_network.transfer(task.mobile_device, task)
        task.next_sub_task = task.response_to_in_distribution_unit_state

    @staticmethod
    def transfer_task_to_edge_computing(task):
        task.mobile_device.distribution_unit.edge_computing_node_network.transfer(
            task.mobile_device.distribution_unit, task)
        task.next_sub_task_parameter_dict = {
            "edge_computing_node":
                task.mobile_device.distribution_unit.edge_computing_node}
        task.next_sub_task = task.response_to_in_edge_computing_node
        task.next_sub_task_parameter_dict["edge_computing_node"].task_replication_list.append((task, task.end_time))

    @staticmethod
    def transfer_task_between_edge_computing_nodes(task, from_ecn, to_ecn):
        from_ecn.edge_computing_node_network[to_ecn].transfer(from_ecn, task)
        to_ecn.task_replication_list.append((task, task.end_time))
