######################################################################################
#
#   Authors:
#       · Alejandro Santorum Varela - alejandro.santorum@estudiante.uam.es
#       · Sergio Galán Martín - sergio.galanm@estudiante.uam.es
#       
#   File: adaline.py
#   Date: Feb. 25, 2021
#   Project: Assignment 1 - Neurocomputation [EPS-UAM]
#
#   Description: This file contains the implementation of the perceptron learning
#       procedure for a neural network of 1 layer.
#
######################################################################################


import sys
from neuro_clfs.NeuralNetwork import NeuralNetwork
from neuro_clfs.Layer import Layer
from neuro_clfs.Neuron import Neuron
from read_data_utils import parse_read_mode


DEFAULT_ALPHA = 1.0
DEFAULT_TH = 0.0
# perceptron.py read_mode file1 [file2] alpha threshold
def read_input_params():
    # Reading train/test sets depending on given read mode
    read_mode, sets = parse_read_mode()

    # Reading learning rate alpha and perceptron threshold (if specified)
    if (read_mode == 1 or read_mode == 3) and len(sys.argv) >= 5:
        alpha = float(sys.argv[4])
        if alpha <= 0 or alpha > 1:
            print("Error: alpha must be in (0, 1]")
            exit()
        if len(sys.argv) == 6:
            threshold = float(sys.argv[5])

    elif read_mode == 2 and len(sys.argv) >= 4:
        alpha = float(sys.argv[3])
        if len(sys.argv) == 5:
            threshold = float(sys.argv[4])
        else:
            threshold = DEFAULT_TH

    else: # default value
        alpha = DEFAULT_ALPHA
        threshold = DEFAULT_TH

    return read_mode, sets, alpha, threshold



def build_perceptron_nn(n_inputs, threshold):
    # Building neural network with given data
    nn = NeuralNetwork()
    input_layer = Layer()
    output_layer = Layer()

    # Creating neurons of input layer
    input_layer.add(Neuron(Neuron.Type.Bias))
    for i in range(n_inputs):
        input_layer.add(Neuron(Neuron.Type.Direct))

    output_layer.add(Neuron(Neuron.Type.Perceptron, threshold=threshold, active_output=1, inactive_output=-1))
    # Layer.WeightMode.PerceptronWeight = Layer.WeightMode.ZeroWeight
    input_layer.connectLayer(output_layer, Layer.WeightMode.ZeroWeight)

    nn.add(input_layer)
    nn.add(output_layer)

    return nn


def train_perceptron_nn(nn, sets, alpha):
    (s, t, _, _) = sets

    n_train = len(s)

    input_layer = nn.layers[0]
    output_layer = nn.layers[1]

    update_flag = True
    while update_flag:

        update_flag = False

        for i in range(n_train):
            # Step 3: init input layer values
            for (j, neuron) in enumerate(input_layer.neurons[1:]):
                neuron.initialise(s[i][j])

            #nn.print_nn()
            # Step 4: calculate output neuron response
            nn.trigger()
            nn.propagate()
            #y_in = output_layer.neurons[0].value
            nn.trigger()
            # Step 5: update weights (if needed)
            #print("Output value (y_in):", y_in, "Output y:", output_layer.neurons[0].f_x, "t_i", t[i][0])
            if output_layer.neurons[0].f_x != t[i][0]:
                # updating w_i
                for (j, neuron) in enumerate(input_layer.neurons[1:]):
                    #print("j-esimo term:", alpha*t[i][0]*s[i][j])
                    neuron.connections[0].update_weight(alpha*t[i][0]*s[i][j])
                # updating b
                #print("b term:", alpha*t[i][0])
                input_layer.neurons[0].connections[0].update_weight(alpha*t[i][0])
            else:
                for neuron in input_layer.neurons:
                    neuron.connections[0].update_weight(0) # term = 0

            if nn.any_weight_update():
                update_flag = True

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
    read_mode, sets, alpha, threshold = read_input_params()

    nn = build_perceptron_nn(len(sets[0][0]), threshold)

    nn = train_perceptron_nn(nn, sets, alpha)

    test_perceptron_nn(nn, sets)
