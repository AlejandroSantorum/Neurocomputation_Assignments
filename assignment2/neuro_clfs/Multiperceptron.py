'''

    Authors:
        · Alejandro Santorum Varela - alejandro.santorum@estudiante.uam.es
        · Sergio Galán Martín - sergio.galanm@estudiante.uam.es

    File: Multiperceptron.py
    Date: Mar. 10, 2021
    Project: Assignment 2 - Neurocomputation [EPS-UAM]

    Description: This file contains the implementation of class Multiperceptron, whose
        goal is to execute forward propagation and backward propagation in order to
        learn the best weights. It inherits NNClassifier class.

'''

from neuro_clfs.NNClassifier import NNClassifier
from neuro_clfs.NeuralNetwork import NeuralNetwork
from neuro_clfs.Layer import Layer
from neuro_clfs.Neuron import Neuron

import numpy as np


def bipolar_sigmoid(x):
    return 2/(1+np.exp(-x)) - 1

def bipolar_sigmoid_derivative(x):
    return 0.5 * (1 + bipolar_sigmoid(x)) * (1 - bipolar_sigmoid(x))



class Multiperceptron(NNClassifier):
    '''
        Implementation of class Multiperceptron, whose goal is to execute multilayer perceptron
        learning algorithm. It inherits NNClassifier class.
    '''

    def __init__(self, n_inputs, n_outputs, hidden_layer_sizes=None, alpha=1.0,\
                 activation='bipolar', n_epochs=100, verbose=False):
        '''
            Constructor of a new object 'Multiperceptron'.

            :param n_inputs: Number of inputs of the neural network
            :param n_outputs: Number of outputs of the neural network
            :param hidden_layer_sizes: Tuple indicating the neurons of the hidden layers.
                The length of this tuple is the number of hidden layers.
                The i-th element represents the number of neurons in the i-th hidden layer,
                    NOT COUNTING THE BIAS NEURON.
            :param alpha: (Optional) Learning parameter alpha. Default=1.0
            :param activation: (Optional) Activation function of the neurons.
                activation='bipolar': it uses bipolar sigmoid function.
            :param verbose: (Optional) If set to True, feedback of each epoch is printed.
            :param n_epochs: (Optional) Number of epochs. Default=100
            :return: None
        '''
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.activation = activation
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.epoch_errors = []
        # building neural network with given data
        self.nn = NeuralNetwork()

        input_layer = Layer()
        input_layer.add(Neuron(Neuron.Type.Bias))
        input_layer.add(Neuron(Neuron.Type.Direct))
        input_layer.add(Neuron(Neuron.Type.Direct))

        hid_layer = Layer()
        hid_layer.add(Neuron(Neuron.Type.Bias))
        hid_layer.add(Neuron(Neuron.Type.BipolarSigmoid))
        hid_layer.add(Neuron(Neuron.Type.BipolarSigmoid))

        input_layer.neurons[0].connect(hid_layer.neurons[0], 0)
        input_layer.neurons[0].connect(hid_layer.neurons[1], 2)
        input_layer.neurons[0].connect(hid_layer.neurons[2], -0.1)
        input_layer.neurons[1].connect(hid_layer.neurons[0], 0)
        input_layer.neurons[1].connect(hid_layer.neurons[1], -2)
        input_layer.neurons[1].connect(hid_layer.neurons[2], 4.3)
        input_layer.neurons[2].connect(hid_layer.neurons[0], 0)
        input_layer.neurons[2].connect(hid_layer.neurons[1], 9.2)
        input_layer.neurons[2].connect(hid_layer.neurons[2], 8.8)

        output_layer = Layer()
        output_layer.add(Neuron(Neuron.Type.BipolarSigmoid))

        hid_layer.neurons[0].connect(output_layer.neurons[0], -0.8)
        hid_layer.neurons[1].connect(output_layer.neurons[0], -4.5)
        hid_layer.neurons[2].connect(output_layer.neurons[0], 5.3)

        self.nn.add(input_layer)
        self.nn.add(hid_layer)
        self.nn.add(output_layer)

        '''
        # creating input layer and its neurons
        input_layer = Layer()
        input_layer.add(Neuron(Neuron.Type.Bias))
        for i in range(n_inputs):
            input_layer.add(Neuron(Neuron.Type.Direct))

        prev_layer = input_layer
        # creating hidden layers and its neurons
        hidden_layers = []
        for i in range(len(self.hidden_layer_sizes)):
            layer = Layer()
            layer.add(Neuron(Neuron.Type.Bias))
            for j in range(self.hidden_layer_sizes[i]):
                # TODO: if statement if several activations are implemented
                if activation == 'bipolar':
                    layer.add(Neuron(Neuron.Type.BipolarSigmoid))
            # weight mode AdalineWeight := random initialization between a small interval
            prev_layer.connectLayer(layer, Layer.WeightMode.AdalineWeight)
            hidden_layers.append(layer)
            prev_layer = layer

        # creating output layer and its neurons
        output_layer = Layer()
        for i in range(n_outputs):
            output_layer.add(Neuron(Neuron.Type.BipolarSigmoid))
        prev_layer.connectLayer(output_layer, Layer.WeightMode.AdalineWeight)

        # adding layers to neural network
        self.nn.add(input_layer)
        for hid_layer in hidden_layers:
            self.nn.add(hid_layer)
        self.nn.add(output_layer)
        '''


    def _forward_propagation(self):
        self.former_values = []
        self.former_activations = []
        # input layer trigger
        self.nn.trigger()
        '''
        temp_former_activations = []
        for neuron in self.nn.layers[0].neurons:
            temp_former_activations.append(neuron.value)
        self.former_activations.append(temp_former_activations)
        '''
        temp_former_activations = []
        for neuron in self.nn.layers[0].neurons:
            temp_former_activations.append(neuron.f_x)
        self.former_activations.append(temp_former_activations)

        # rest of the network propagation
        for layer in self.nn.layers[1:]:
            self.nn.propagate()
            temp_former_values = []
            for neuron in layer.neurons:
                temp_former_values.append(neuron.value)

            self.nn.trigger()
            temp_former_activations = []
            for neuron in layer.neurons:
                temp_former_activations.append(neuron.f_x)

            self.former_values.append(temp_former_values)
            self.former_activations.append(temp_former_activations)


    def _backward_propagation(self, ytrain_array):
        self.Deltas = []
        # getting last array of predictions (each output neuron activation)
        predictions = self.former_activations[-1]
        prev_layer = self.former_activations[-2]

        # getting last array of values (each output neuron value)
        pred_values = self.former_values[-1]
        deltas_prev_layer = []
        Delta_layer = []
        for k,neuron in enumerate(self.nn.layers[-1].neurons):
            if self.activation == 'bipolar':
                delta_k = (ytrain_array[k]-predictions[k]) * bipolar_sigmoid_derivative(pred_values[k])
                deltas_prev_layer.append(delta_k)
                for act in prev_layer:
                    Delta_layer.append(self.alpha * delta_k * act)
        self.Deltas.append(Delta_layer)

        for (i,layer) in enumerate(reversed(self.nn.layers[1:-1])):
            #pred_values = reversed(self.former_values)[i+1]
            pred_values = self.former_values[::-1][i+1]

            #prev_layer = reversed(self.former_activations)[i+2]
            prev_layer = self.former_activations[::-1][i+2]

            deltas_prev_layer_new = []
            Delta_layer = []
            for (j,neuron) in enumerate(layer.neurons):
                delta_in_j = 0
                for (k, connection) in enumerate(neuron.connections):
                    delta_in_j += connection.weight * deltas_prev_layer[k]
                delta_j = delta_in_j * bipolar_sigmoid_derivative(pred_values[j])
                deltas_prev_layer_new.append(delta_j)
                for act in prev_layer:
                    Delta_layer.append(self.alpha * delta_j * act)
            deltas_prev_layer = deltas_prev_layer_new
            self.Deltas.append(Delta_layer)


    def _update_nn_weights(self):
        current_layer = self.nn.layers[-2]
        next_layer = self.nn.layers[-1]

        for j,Delta in enumerate(self.Deltas):
            while Delta != []:
                for neuron in current_layer.neurons:
                    for i in range(len(next_layer.neurons)):
                        aux = Delta.pop(0)
                        neuron.connections[i].update_weight(aux)

            if j == len(self.Deltas) - 1:
                break
            next_layer = current_layer
            current_layer = self.nn.layers[-3-j]



    def train(self, xtrain, ytrain):
        '''
            Input parameters and return explained in parent class.

            It trains the Multiperceptron object accordingly to perceptron algorithm.
        '''
        n_train = len(xtrain)

        # getting input and output layers
        input_layer = self.nn.layers[0]
        output_layer = self.nn.layers[1]

        # training loop
        for k in range(self.n_epochs):
            # setting flag to False before every epoch
            update_flag = False
            if self.verbose:
                print("Epoch", k)

            # an epoch trains over all examples
            for i in range(n_train):
                # nit input layer values
                for (j, neuron) in enumerate(input_layer.neurons[1:]):
                    neuron.initialise(xtrain[i][j])

                # calculate neurons values
                self._forward_propagation()

                # backpropagate gradient
                self._backward_propagation(ytrain[i])

                # update weights
                self._update_nn_weights()

                self.nn.print_nn()
                exit()

            self.epoch_errors.append(self.error(ytrain, self.predict(xtrain), metric='mse'))


    def predict(self, xtest):
        '''
            Input parameters and return explained in parent class.

            It predicts the network output accordingly its hyperparameters and fitting parameters.
        '''
        n_test = len(xtest)
        input_layer = self.nn.layers[0]

        ytest = []

        for i in range(n_test):
            # init input layer values
            for (j, neuron) in enumerate(input_layer.neurons[1:]):
                neuron.initialise(xtest[i][j])
            # calculate output neuron response
            # self._forward_propagation()

            self.nn.trigger()
            for layer in self.nn.layers[1:]:
                self.nn.propagate()
                self.nn.trigger()

            outputs = self.nn.get_output()

            if len(outputs) == 1:
                if outputs[0] > 0:
                    ytest.append([1])
                else:
                    ytest.append([-1])
            else:
                ret = [-1]*len(outputs)
                max_idx = outputs.index(max(outputs))
                ret[max_idx] = 1
                ytest.append(ret)

        return ytest
