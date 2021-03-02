'''

    Authors:
        · Alejandro Santorum Varela - alejandro.santorum@estudiante.uam.es
        · Sergio Galán Martín - sergio.galanm@estudiante.uam.es

    File: ex_adaline.py
    Date: Feb. 28, 2021
    Project: Assignment 1 - Neurocomputation [EPS-UAM]

    Description: This file contains the script for adaline exercise

'''

import sys
from neuro_clfs.Adaline import Adaline
from read_data_utils import parse_read_mode
from tabulate import tabulate


DEFAULT_ALPHA = 0.1
DEFAULT_TOL = 0.001
# ex_adaline.py read_mode file1 [file2] alpha tol
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


def main(read_mode, sets, alpha, tol):
    xtrain, ytrain, xtest, ytest = sets

    n_inputs = len(xtrain[0])

    adal_nn = Adaline(n_inputs, alpha, tol)

    adal_nn.train(xtrain, ytrain)
    ypred = adal_nn.predict(xtest)

    results = []
    headers = ["x1", "x2", "t (target)", "y (predicted)"]
    for i in range(len(ytest)):
        results.append([str(xtest[i][0]), str(xtest[i][1]), str(ytest[i][0]), str(ypred[i][0])])

    print(tabulate(results, headers=headers, tablefmt="pretty"))



if __name__ == '__main__':
    read_mode, sets, alpha, tol = read_input_params()
    main(read_mode, sets, alpha, tol)
