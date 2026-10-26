class Layer:
    def __init__(self, data_size=0, flops_size=0):
        self.name = None
        self.data_size = data_size
        self.flops_size = flops_size

    def __str__(self):
        return self.name
