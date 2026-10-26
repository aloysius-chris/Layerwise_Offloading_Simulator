class DistributionUnit:
    def __init__(self):
        self.name = ""
        self.mobile_device_list = []
        self.edge_computing_node_network = None
        self.edge_computing_node = None

        self.log_list = []

    def __str__(self):
        return self.name
