######################################################################################
#
#   Authors:
#       · Alejandro Santorum Varela - alejandro.santorum@estudiante.uam.es
#       · Sergio Galán Martín - sergio.galanm@estudiante.uam.es
#       
#   File: Layer.py
#   Date: Feb. 20, 2021
#   Project: Assignment 1 - Neurocomputation [EPS-UAM]
#
#   Description: This file contains the implementation of class Layer, which goal
#       is to group several neurons allocated at the same depth of a neural
#       network.
#
######################################################################################

from enum import Enum
import random

class Layer:

    class WeightMode(Enum):
        ZeroWeight = 0
        AdalineWeight = 1

    def __init__(self):
        self.neurons = []

    def free():
        pass

    def initialise():
        for neuron in self.neurons:
            neuron.initialise()

    def any_weight_update(self):
        for neuron in self.neurons:
            if neuron.any_weight_update():
                return True
        return False

    def add(self, neuron):
        self.neurons.append(neuron)

    def connectLayer(self, layer, weight_mode):
        for neuron in layer.neurons:
            self.connectNeuron(neuron, weight_mode)

    def connectNeuron(self, neuron, weight_mode):
        if weight_mode is Layer.WeightMode.ZeroWeight:
            for orig_neuron in self.neurons:
                orig_neuron.connect(neuron, 0)

        elif weight_mode is Layer.WeightMode.AdalineWeight:
            for orig_neuron in self.neurons:
                orig_neuron.connect(neuron, random.uniform(-0.5, 0.5))

    def trigger(self):
        for neuron in self.neurons:
            neuron.trigger()

    def propagate(self):
        for neuron in self.neurons:
            neuron.propagate()

    def print_layer(self):
        for (i, neuron) in enumerate(self.neurons):
            print("\t\tNeuron", i)
            neuron.print_neuron()
