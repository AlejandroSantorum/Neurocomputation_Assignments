import sys
from neuro_clfs.NeuralNetwork import NeuralNetwork
from neuro_clfs.Layer import Layer
from neuro_clfs.Neuron import Neuron


def build_nn_ex1():
    # Building neural network with given data
    nn = NeuralNetwork()
    input_layer = Layer()
    hidden_layer = Layer()
    output_layer = Layer()

    input_layer.add(Neuron(0.5, Neuron.Type.Direct))
    input_layer.add(Neuron(0.5, Neuron.Type.Direct))
    input_layer.add(Neuron(0.5, Neuron.Type.Direct))

    hidden_layer.add(Neuron(2, Neuron.Type.McCulloch, 1, 0))
    hidden_layer.add(Neuron(2, Neuron.Type.McCulloch, 1, 0))
    hidden_layer.add(Neuron(2, Neuron.Type.McCulloch, 1, 0))

    output_layer.add(Neuron(1, Neuron.Type.McCulloch, active_output=1, inactive_output=0))

    input_layer.connectLayer(hidden_layer, Layer.WeightMode.OneWeight)
    hidden_layer.connectLayer(output_layer, Layer.WeightMode.OneWeight)

    nn.add(input_layer)
    nn.add(hidden_layer)
    nn.add(output_layer)

    return nn



if __name__ == "__main__":

    # Reading data file
    if len(sys.argv) == 2:
        file = open(sys.argv[1])
        file_lines = file.readlines()
    else:
        # Error: No data file provided
        print("Please, provide data file name")
        exit()

    nn = build_nn_ex1()

    print('x1  x2  x3  a12  a13  a23  y')
    for line in file_lines:
        values = line.split()

        i = 0
        for neuron in nn.layers[0].neurons:
            neuron.initialise(int(values[i]))
            i += 1

        nn.trigger()
        nn.propagate()

        for layer in nn.layers:
            for neuron in layer.neurons:
                print(neuron.value, " ", end='')
        print("")
