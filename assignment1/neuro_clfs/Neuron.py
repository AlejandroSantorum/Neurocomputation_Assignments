class Neuron:

    Types = enum('Direct', 'McCulloch', 'Bias', 'BipolarSigmoid', 'CustomSigmoid')

    def __init__(self, threshold, type, active_output = None, inactive_output = None):
        self.threshold = threshold
        self.type = type
        self.active_output = active_output
        self.inactive_output = inactive_output
        self.connections = []
        self.f_x = 0
        self.value = 0

    def free():
        pass

    def initialise(value):
        self.value = value

    def connect(neuron, weight):
        self.connections.append(Connection(neuron, weight))

    def trigger():
        if self.type is Types.Direct:
            self.f_x = self.value
        else if self.type is Types.Bias:
            self.f_x = 1.0
        else if self.type is Types.McCulloch:
            self.f_x = self.active_output if self.value >= self.threshold else self.inactive_output
        # else
        for conection in self.connections:
            connection.received_value = self.f_x

    def propagate():
        for connection in self.connections:
            connection.propagate()