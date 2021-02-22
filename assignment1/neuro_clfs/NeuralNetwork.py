class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def free():
        pass

    def initialise():
        for layer in self.layers:
            layer.initialise()

    def any_weight_update(self):
        for layer in self.layers:
            if layer.any_weight_update():
                return True
        return False

    def add(self, layer):
        self.layers.append(layer)

    def trigger(self):
        for layer in self.layers:
            layer.trigger()

    def propagate(self):
        for layer in self.layers:
            layer.propagate()

    def get_output(self):
        output_vals = []
        for neuron in self.layers[-1].neurons:
            output_vals.append(neuron.f_x)
        return output_vals

    def print_nn(self):
        for (i, layer) in enumerate(self.layers):
            print("\tLayer", i)
            layer.print_layer()
