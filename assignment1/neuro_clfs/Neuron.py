'''

    Authors:
        · Alejandro Santorum Varela - alejandro.santorum@estudiante.uam.es
        · Sergio Galán Martín - sergio.galanm@estudiante.uam.es

    File: Neuron.py
    Date: Feb. 20, 2021
    Project: Assignment 1 - Neurocomputation [EPS-UAM]

    Description: This file contains the implementation of class Neuron, the logic
        units of a neural network.

'''


from enum import Enum
from neuro_clfs.Connection import Connection


class Neuron:

    class Type(Enum):
        Direct = 0
        McCulloch = 1
        Bias = 2
        BipolarSigmoid = 3
        CustomSigmoid = 4
        Perceptron = 5

    def __init__(self, type, threshold=0.5, active_output=None, inactive_output=None):
        self.type = type
        self.threshold = threshold
        self.active_output = active_output
        self.inactive_output = inactive_output
        self.connections = []
        self.f_x = 0
        self.value = 0

    def free():
        pass

    def initialise(self, value):
        self.value = value

    def any_weight_update(self):
        for connection in self.connections:
            if connection.any_weight_update():
                return True
        return False

    def connect(self, neuron, weight):
        self.connections.append(Connection(weight, neuron))

    def trigger(self):
        if self.type is Neuron.Type.Direct:
            self.f_x = self.value
        elif self.type is Neuron.Type.Bias:
            self.f_x = 1.0
        elif self.type is Neuron.Type.BipolarSigmoid:
            self.f_x = 1 if self.value >= 0 else -1
        elif self.type is Neuron.Type.McCulloch:
            self.f_x = self.active_output if self.value >= self.threshold else self.inactive_output
        elif self.type is Neuron.Type.Perceptron:
            if self.value > self.threshold:
                self.f_x = self.active_output
            elif self.value < -self.threshold:
                self.f_x = self.inactive_output
            else:
                self.f_x = 0

        self.value = 0
        for connection in self.connections:
            connection.received_value = self.f_x

    def propagate(self):
        for connection in self.connections:
            connection.propagate()

    def print_neuron(self):
        print("\t\t\tValue:", self.value)
        print("\t\t\tF_x:", self.f_x)
        for (i,conn) in enumerate(self.connections):
            print("\t\t\tConnection", i)
            conn.print_connection()
