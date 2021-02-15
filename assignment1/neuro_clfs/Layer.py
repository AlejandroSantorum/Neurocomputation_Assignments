class Layer:
    def __init__():
        self.neurons = []

    def Free():
        pass

    def Initialise():
        pass

    def Add(neuron):
        self.neurons.append(neuron)

    def ConnectLayer(layer, weight_mode):
        pass

    def ConnectNeuron(neuron, weight_mode):
        pass

    def Trigger():
        for neuron in self.neurons:
            neuron.Trigger()

    def Propagate():
        for neuron in self.neurons:
            neuron.Propagate()