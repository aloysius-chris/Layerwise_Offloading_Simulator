class MobileDevice:
    def __init__(self):
        self.name = ""
        self.distribution_unit_network = None
        self.distribution_unit = None

        self.log_list = []

    def __str__(self):
        return self.name
