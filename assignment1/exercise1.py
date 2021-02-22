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

    x1 = Neuron(Neuron.Type.Direct)
    x2 = Neuron(Neuron.Type.Direct)
    x3 = Neuron(Neuron.Type.Direct)
    input_layer.add(x1)
    input_layer.add(x2)
    input_layer.add(x3)
    h1 = Neuron(Neuron.Type.McCulloch, threshold=2, 1, 0)
    h2 = Neuron(Neuron.Type.McCulloch, threshold=2, 1, 0)
    h3 = Neuron(Neuron.Type.McCulloch, threshold=2, 1, 0)
    hidden_layer.add(h1)
    hidden_layer.add(h2)
    hidden_layer.add(h3)
    o1 = Neuron(Neuron.Type.McCulloch, threshold=1, 1, 0)
    output_layer.add(o1)

    # a12
    x1.connect(h1, 1)
    x2.connect(h1, 1)
    x3.connect(h1, 0)
    # a13
    x1.connect(h2, 1)
    x2.connect(h2, 0)
    x3.connect(h2, 1)
    # a23
    x1.connect(h3, 0)
    x2.connect(h3, 1)
    x3.connect(h3, 1)

    h1.connect(o1, 1)
    h2.connect(o1, 1)
    h3.connect(o1, 1)

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
                print(neuron.f_x, "  ", end='')
        print("")

## TODO:
#   1) Pretty printer
#   2) Hacer las preguntas
#   3) Cabeceras de ficheros, funciones y un par de comentarios (pydoc)

## Preguntas:
#   1) ¿Hay que mostrar el estado de la capa oculta/salida despues de que se acaben los ejemplos? -> Sí, poniendo entradas "basura"
#   2) ¿Podríamos simplificar la red neuronal para que se obtiviese la salida correcta en t+1? -> No
#   3) ¿Es una OR de tres entradas o el circuito de la figura 2? -> El de la figura
#   4) ¿Es necesario utilizar gitlab o podemos usar github? -> Github
