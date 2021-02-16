class Connection:
    def __init__(self, weight, neuron):
        self.former_weight = None
        self.weight = weight
        self.neuron = neuron
        self.received_value = None

    def free():
        pass

    def propagate(self):
        self.neuron.value += self.weight * self.received_value
