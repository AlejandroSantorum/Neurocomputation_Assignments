from enum import Enum

class Layer:

    class WeightMode(Enum):
        ZeroWeight = 0
        OneWeight = 1
        PerceptronWeight = 2

    def __init__(self):
        self.neurons = []

    def free():
        pass

    def initialise():
        pass

    def add(self, neuron):
        self.neurons.append(neuron)

    def connectLayer(self, layer, weight_mode):
        for neuron in layer.neurons:
            self.connectNeuron(neuron, weight_mode)

    def connectNeuron(self, neuron, weight_mode):
        if weight_mode is Layer.WeightMode.ZeroWeight:
            for orig_neuron in self.neurons:
                orig_neuron.connect(neuron, 0)

        elif weight_mode is Layer.WeightMode.OneWeight:
            for orig_neuron in self.neurons:
                orig_neuron.connect(neuron, 1)

        elif weight_mode is Layer.WeightMode.PerceptronWeight:
            pass

    def trigger(self):
        for neuron in self.neurons:
            neuron.trigger()

    def propagate(self):
        for neuron in self.neurons:
            neuron.propagate()
