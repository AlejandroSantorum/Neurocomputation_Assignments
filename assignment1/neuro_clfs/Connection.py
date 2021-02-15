class Connection:
    def __init__(weight, neuron):
        self.former_weight = None
        self.weight = weight
        self.neuron = neuron
        self.received_value = None

    def Free():
        pass

    def Propagate():
        self.neuron.value += self.weight * self.received_value