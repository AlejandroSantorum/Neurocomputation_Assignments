class Connection:
    def __init__(self, weight, neuron):
        self.former_weight = None
        self.weight = weight
        self.neuron = neuron
        self.received_value = None

    def update_weight(self, term):
        self.former_weight = self.weight
        self.weight = self.weight + term

    def any_weight_update(self):
        if self.former_weight == self.weight:
            return False
        return True

    def propagate(self):
        self.neuron.value += self.weight * self.received_value
