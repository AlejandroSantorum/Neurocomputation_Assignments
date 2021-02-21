import sys
from neuro_clfs.NeuralNetwork import NeuralNetwork
from neuro_clfs.Layer import Layer
from neuro_clfs.Neuron import Neuron
from read_data_utils import parse_read_mode




def read_input_params():
    # Reading train/test sets depending on given read mode
    read_mode, sets = parse_read_mode()

    # Reading learning rate alpha (if specified)
    if (read_mode == 1 or read_mode == 3) and len(sys.argv) == 5:
        alpha = float(sys.argv[4])

    elif read_mode == 2 and len(sys.argv) == 4:
        alpha = float(sys.argv[3])

    else: # default value
        alpha = 1.0

    return read_mode, sets, alpha



if __name__ == '__main__':
    read_mode, sets, alpha = read_input_params()
    print(read_mode, sets, alpha)


# ¿Cómo actualizar los pesos dinámicamente?