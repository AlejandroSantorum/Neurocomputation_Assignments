'''

    Authors:
        · Alejandro Santorum Varela - alejandro.santorum@estudiante.uam.es
        · Sergio Galán Martín - sergio.galanm@estudiante.uam.es

    File: adaline.py
    Date: Feb. 26, 2021
    Project: Assignment 1 - Neurocomputation [EPS-UAM]

    Description: This file contains the implementation of the adaline learning
        procedure for a neural network of 1 layer.

'''

import sys
from neuro_clfs.NeuralNetwork import NeuralNetwork
from neuro_clfs.Layer import Layer
from neuro_clfs.Neuron import Neuron
from read_data_utils import parse_read_mode


DEBUG_FLAG = False

DEFAULT_ALPHA = 0.1
DEFAULT_TOL = 0.001
# adaline.py read_mode file1 [file2] alpha tol
def read_input_params():
    # Reading train/test sets depending on given read mode
    read_mode, sets = parse_read_mode()

    # Reading learning rate alpha and perceptron tolerance (if specified)
    if (read_mode == 1 or read_mode == 3) and len(sys.argv) >= 5:
        alpha = float(sys.argv[4])
        if alpha >= 1:
            print("Warning: alpha might be too large to converge")
        if len(sys.argv) == 6:
            tol = float(sys.argv[5])

    elif read_mode == 2 and len(sys.argv) >= 4:
        alpha = float(sys.argv[3])
        if len(sys.argv) == 5:
            tol = float(sys.argv[4])
        else:
            tol = DEFAULT_TOL

    else: # default value
        alpha = DEFAULT_ALPHA
        tol = DEFAULT_TOL

    return read_mode, sets, alpha, tol



def build_perceptron_nn(n_inputs):
    # Building neural network with given data
    nn = NeuralNetwork()
    input_layer = Layer()
    output_layer = Layer()

    # Creating neurons of input layer
    input_layer.add(Neuron(Neuron.Type.Bias))
    for i in range(n_inputs):
        input_layer.add(Neuron(Neuron.Type.Direct))

    output_layer.add(Neuron(Neuron.Type.BipolarSigmoid))
    # Layer.WeightMode.AdalineWeight = Layer.WeightMode.RandomWeight
    input_layer.connectLayer(output_layer, Layer.WeightMode.AdalineWeight)

    nn.add(input_layer)
    nn.add(output_layer)

    return nn



def train_perceptron_nn(nn, sets, alpha, tol):
    (s, t, _, _) = sets

    n_train = len(s)

    input_layer = nn.layers[0]
    output_layer = nn.layers[1]

    while True:
        updates = [0]*len(input_layer.neurons)
        for i in range(n_train):

            # Step 3: init input layer values
            for (j, neuron) in enumerate(input_layer.neurons[1:]):
                neuron.initialise(s[i][j])

            #nn.print_nn()

            # Step 4: calculate output neuron response
            nn.trigger()
            nn.propagate()
            # Step 5: update weights
            y_in = output_layer.neurons[0].value
            #print("Output value (y_in):", y_in, "t_i", t[i][0])
            # updating w_i
            for (j, neuron) in enumerate(input_layer.neurons[1:]):
                update_value = alpha*(t[i][0] - y_in)*s[i][j]
                print("j-esimo term:", update_value)
                neuron.connections[0].update_weight(update_value)
                updates[j] += update_value
                # if abs(update_value) > max_update:
                #     max_update = abs(update_value)
            # updating b
            update_value = alpha*(t[i][0] - y_in)
            #print("b term:", update_value)
            input_layer.neurons[0].connections[0].update_weight(update_value)
            updates[-1] += update_value
            # if abs(update_value) > max_update:
            #     max_update = abs(update_value)

            #print("Max_update", max_update)
        max_update = max(map(abs,updates))
        print(max_update)
        if max_update < tol:
            break
    return nn


def test_perceptron_nn(nn, sets):
    (_, _, s, t) = sets

    n_test = len(s)
    input_layer = nn.layers[0]

    for i in range(n_test):
        # init input layer values
        for (j, neuron) in enumerate(input_layer.neurons[1:]):
            neuron.initialise(s[i][j])
        # calculate output neuron response
        nn.trigger()
        nn.propagate()
        nn.trigger()

        print(s[i], end='')
        print(t[i], end='')
        print(nn.get_output())



if __name__ == '__main__':
    read_mode, sets, alpha, tol = read_input_params()

    nn = build_perceptron_nn(len(sets[0][0]))

    nn = train_perceptron_nn(nn, sets, alpha, tol)

    test_perceptron_nn(nn, sets)


# THINGS TO DO:
    # 1) Vectorize code
    # 2) ¿Number of iterations instead of tolerance?
    # 3) ¿Print loss per epoch?
