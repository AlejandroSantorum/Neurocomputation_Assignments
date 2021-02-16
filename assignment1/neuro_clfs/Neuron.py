from enum import Enum
from neuro_clfs.Connection import Connection


class Neuron:

    class Type(Enum):
        Direct = 0
        McCulloch = 1
        Bias = 2
        BipolarSigmoid = 3
        CustomSigmoid = 4

    def __init__(self, threshold, type, active_output=None, inactive_output=None):
        self.threshold = threshold
        self.type = type
        self.active_output = active_output
        self.inactive_output = inactive_output
        self.connections = []
        self.f_x = 0
        self.value = 0

    def free():
        pass

    def initialise(self, value):
        self.value = value

    def connect(self, neuron, weight):
        self.connections.append(Connection(weight, neuron))

    def trigger(self):
        if self.type is Neuron.Type.Direct:
            self.f_x = self.value
        elif self.type is Neuron.Type.Bias:
            self.f_x = 1.0
        elif self.type is Neuron.Type.McCulloch:
            self.f_x = self.active_output if self.value >= self.threshold else self.inactive_output
        # else
        for connection in self.connections:
            connection.received_value = self.f_x

    def propagate(self):
        for connection in self.connections:
            connection.propagate()
