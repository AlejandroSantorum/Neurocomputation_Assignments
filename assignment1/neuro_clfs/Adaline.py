from .NNClassifier import NNClassifier
from .NeuralNetwork import NeuralNetwork
from .Layer import Layer
from .Neuron import Neuron
import numpy as np

class Adaline(NNClassifier):

    def __init__(self, n_inputs, n_outputs, alpha=0.1, tol=0.01, verbose=False):
        self.alpha = alpha
        self.tol = tol
        self.verbose = verbose
        self.epoch_errors = []

        # building neural network with given data
        self.nn = NeuralNetwork()
        input_layer = Layer()
        output_layer = Layer()

        # creating neurons of input layer
        input_layer.add(Neuron(Neuron.Type.Bias))
        for i in range(n_inputs):
            input_layer.add(Neuron(Neuron.Type.Direct))

        for i in range(n_outputs):
            output_layer.add(Neuron(Neuron.Type.BipolarSigmoid))

        # WeightMode AdalineWeight = WeightMode RandomWeight
        input_layer.connectLayer(output_layer, Layer.WeightMode.AdalineWeight)

        self.nn.add(input_layer)
        self.nn.add(output_layer)


    def train(self, xtrain, ytrain):
        n_train = len(xtrain)
        n_outputs = len(ytrain[0])

        # getting input and output layers
        input_layer = self.nn.layers[0]
        output_layer = self.nn.layers[1]

        n_epoch = 0
        while True:
            updates = np.zeros((n_outputs, len(input_layer.neurons)))
            for i in range(n_train):

                # init input layer values
                for (j, neuron) in enumerate(input_layer.neurons[1:]):
                    neuron.initialise(xtrain[i][j])

                # calculate output neuron response
                self.nn.trigger()
                self.nn.propagate()
                # update weights
                for (j, neuron_out) in enumerate(output_layer.neurons):
                    y_in = neuron_out.value
                    # updating w_i
                    for (k, neuron_in) in enumerate(input_layer.neurons[1:]):
                        update_value = self.alpha*(ytrain[i][j] - y_in)*xtrain[i][k]
                        neuron_in.connections[j].update_weight(update_value)
                        updates[j][k] += update_value
                    # updating b
                    update_value = self.alpha*(ytrain[i][j] - y_in)
                    input_layer.neurons[0].connections[j].update_weight(update_value)
                    updates[j][-1] += update_value

                # y_in = output_layer.neurons[0].value
                # # updating w_i
                # for (j, neuron) in enumerate(input_layer.neurons[1:]):
                #     update_value = self.alpha*(ytrain[i][0] - y_in)*xtrain[i][j]
                #     neuron.connections[0].update_weight(update_value)
                #     updates[j] += update_value
                # # updating b
                # update_value = self.alpha*(ytrain[i][0] - y_in)
                # input_layer.neurons[0].connections[0].update_weight(update_value)
                # updates[-1] += update_value

            max_update = np.amax(np.absolute(updates))

            self.epoch_errors.append(self.error(ytrain, self.predict(xtrain), metric='mse'))

            if self.verbose:
                n_epoch += 1
                print("Epoch", n_epoch, " --> Maximum weight diff:", max_update)

            if max_update < self.tol:
                break


    def predict(self, xtest):
        n_test = len(xtest)
        input_layer = self.nn.layers[0]

        ytest = []

        for i in range(n_test):
            # init input layer values
            for (j, neuron) in enumerate(input_layer.neurons[1:]):
                neuron.initialise(xtest[i][j])
            # calculate output neuron response
            self.nn.trigger()
            self.nn.propagate()
            self.nn.trigger()

            ytest.append(self.nn.get_output())

        return ytest
