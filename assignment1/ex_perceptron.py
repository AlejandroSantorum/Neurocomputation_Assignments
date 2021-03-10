'''
    Authors:
        · Alejandro Santorum Varela - alejandro.santorum@estudiante.uam.es
        · Sergio Galán Martín - sergio.galanm@estudiante.uam.es

    File: ex_perceptron.py
    Date: Feb. 28, 2021
    Project: Assignment 1 - Neurocomputation [EPS-UAM]

    Description: This file contains the script for perceptron exercise

'''


import sys
from neuro_clfs.Perceptron import Perceptron
from read_data_utils import parse_read_mode
from tabulate import tabulate


DEFAULT_ALPHA = 1.0
DEFAULT_TH = 0.0
DEFAULT_EPOCH = 20
# ex_perceptron.py read_mode file1 [file2/percentage] alpha threshold
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

    # TODO: get max_epoch as parameter

    else: # default value
        alpha = DEFAULT_ALPHA
        threshold = DEFAULT_TH
        max_epoch = DEFAULT_EPOCH

    return read_mode, sets, alpha, threshold, max_epoch




def main(sets, alpha, threshold, max_epoch):
    xtrain, ytrain, xtest, ytest = sets

    n_inputs = len(xtrain[0])
    n_outputs = len(ytrain[0])

    perc_nn = Perceptron(n_inputs, n_outputs, threshold, alpha, False, max_epoch)

    perc_nn.train(xtrain, ytrain)
    ypred = perc_nn.predict(xtest)

    results = []
    headers = ["x1", "x2", "t (target)", "y (predicted)"]
    for i in range(len(ytest)):
        results.append([str(xtest[i][0]), str(xtest[i][1]), str(ytest[i][0]), str(ypred[i][0])])

    print(tabulate(results, headers=headers, tablefmt="pretty"))

    print("MSE Loss:", perc_nn.error(ytest, ypred, metric='mse'))

    weight_list = []
    input_layer = perc_nn.nn.layers[0]
    for neuron in input_layer.neurons:
        for conn in neuron.connections:
            weight_list.append(conn.weight)

    b = weight_list[0]
    weight_list = weight_list[1:]
    print("Decision boundary: ", end='')
    for (i, w) in enumerate(weight_list):
        print(str(round(w, 2)) + "·X"+str(i+1) + " + ", end='')
    print(str(round(b, 2)) + " = 0")




if __name__ == '__main__':
    read_mode, sets, alpha, threshold, max_epoch = read_input_params()
    main(sets, alpha, threshold, max_epoch)
