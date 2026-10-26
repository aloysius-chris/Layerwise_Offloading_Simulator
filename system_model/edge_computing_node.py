class EdgeComputingNode:
    def __init__(self):
        self.name = ""
        self.virtual_machine_list = []
        self.distribution_unit_list = []
        self.edge_computing_node_network = {}
        self.edge_computing_node_list = []
        self.shortest_path = {}
        self.cloud_network = None
        self.cloud = None
        self.task_replication_list = []

        self.log_list = []

    def __str__(self):
        return self.name
