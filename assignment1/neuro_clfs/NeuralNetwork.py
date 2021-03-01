'''

    Authors:
        · Alejandro Santorum Varela - alejandro.santorum@estudiante.uam.es
        · Sergio Galán Martín - sergio.galanm@estudiante.uam.es

    File: NeuralNetwork.py
    Date: Feb. 20, 2021
    Project: Assignment 1 - Neurocomputation [EPS-UAM]

    Description: This file contains the implementation of class NeuralNetwork, that
        is a set of neurons and connections in order to approximate a general
        function, determined by its weights and input parameters.

'''


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
