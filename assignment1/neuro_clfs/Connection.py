######################################################################################
#
#   Authors:
#       · Alejandro Santorum Varela - alejandro.santorum@estudiante.uam.es
#       · Sergio Galán Martín - sergio.galanm@estudiante.uam.es
#       
#   File: Connection.py
#   Date: Feb. 20, 2021
#   Project: Assignment 1 - Neurocomputation [EPS-UAM]
#
#   Description: This file contains the implementation of class Connection, which
#       goal is to transfer information (weights, activations, etc.) through
#       different types of neural networks. 
#
######################################################################################


class Connection:
    def __init__(self, weight, neuron):
        self.former_weight = None
        self.weight = weight
        self.neuron = neuron
        self.received_value = None

    def update_weight(self, term):
        #print("[Antes] Former: ", self.former_weight, "Weight:" , self.weight)
        self.former_weight = self.weight
        self.weight = self.weight + term
        #print("[Despues] Former: ", self.former_weight, "Weight:" , self.weight)

    def any_weight_update(self):
        if self.former_weight == self.weight:
            return False
        return True

    def propagate(self):
        self.neuron.value += self.weight * self.received_value

    def print_connection(self):
        print("\t\t\t\tFormer_weight:", self.former_weight)
        print("\t\t\t\tWeight:", self.weight)