from .NNClassifier import NNClassifier
from .NeuralNetwork import NeuralNetwork
from .Layer import Layer
from .Neuron import Neuron

class Perceptron(NNClassifier):

    def __init__(self, n_inputs, n_outputs, threshold=0.1, alpha=1.0, verbose=False, max_epoch=20):
        self.threshold = threshold
        self.alpha = alpha
        self.verbose = verbose
        self.max_epoch = max_epoch

        # building neural network with given data
        self.nn = NeuralNetwork()
        input_layer = Layer()
        output_layer = Layer()

        # creating neurons of input layer
        input_layer.add(Neuron(Neuron.Type.Bias))
        for i in range(n_inputs):
            input_layer.add(Neuron(Neuron.Type.Direct))

        for i in range(n_outputs):
            output_layer.add(Neuron(Neuron.Type.Perceptron, threshold=threshold, active_output=1, inactive_output=-1))
        # WeightMode PerceptronWeight = WeightMode ZeroWeight
        input_layer.connectLayer(output_layer, Layer.WeightMode.ZeroWeight)

        self.nn.add(input_layer)
        self.nn.add(output_layer)


    def train(self, xtrain, ytrain):
        n_train = len(xtrain)

        # getting input and output layers
        input_layer = self.nn.layers[0]
        output_layer = self.nn.layers[1]

        # training loop: update_flag is True if any nn weight is updated
        update_flag = True
        n_epoch = 0
        while update_flag and n_epoch < self.max_epoch:
            # setting flag to False before every epoch
            update_flag = False
            n_epoch += 1
            if self.verbose:
                print("Epoch", n_epoch)

            # an epoch trains over all examples
            for i in range(n_train):
                # nit input layer values
                for (j, neuron) in enumerate(input_layer.neurons[1:]):
                    neuron.initialise(xtrain[i][j])

                # calculate output neuron response
                self.nn.trigger()
                self.nn.propagate()
                self.nn.trigger()
                # update weights (if needed)
                for (j, neuron_out) in enumerate(output_layer.neurons):
                    if neuron_out.f_x != ytrain[i][j]:
                        # updating w_i
                        for (k, neuron_in) in enumerate(input_layer.neurons[1:]):
                            neuron_in.connections[j].update_weight(self.alpha*ytrain[i][j]*xtrain[i][k])
                        # updating b
                        input_layer.neurons[0].connections[j].update_weight(self.alpha*ytrain[i][j])

                    else:
                        for neuron_in in input_layer.neurons:
                            neuron_in.connections[j].update_weight(0) # term = 0

                # if output_layer.neurons[0].f_x != ytrain[i][0]:
                #     # updating w_i
                #     for (k, neuron) in enumerate(input_layer.neurons[1:]):
                #         neuron.connections[0].update_weight(self.alpha*ytrain[i][0]*xtrain[i][k])
                #     # updating b
                #     input_layer.neurons[0].connections[0].update_weight(self.alpha*ytrain[i][0])
                # else:
                #     for neuron in input_layer.neurons:
                #         neuron.connections[0].update_weight(0) # term = 0

                # checking if any former weight is different than current weight
                if self.nn.any_weight_update():
                    update_flag = True


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
