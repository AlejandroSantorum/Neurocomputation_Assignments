class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def free():
        pass

    def initialise():
        pass

    def add(self, layer):
        self.layers.append(layer)

    def trigger(self):
        for layer in self.layers:
            layer.trigger()

    def propagate(self):
        for layer in reversed(self.layers):
            layer.propagate()
