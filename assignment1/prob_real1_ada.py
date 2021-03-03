'''
    Authors:
        · Alejandro Santorum Varela - alejandro.santorum@estudiante.uam.es
        · Sergio Galán Martín - sergio.galanm@estudiante.uam.es

    File: prob_real1_ada.py
    Date: Mar. 03, 2021
    Project: Assignment 1 - Neurocomputation [EPS-UAM]

    Description: This file contains the script for real problem 1 adaline exercise

'''


import sys
import numpy as np
from neuro_clfs.Adaline import Adaline
from read_data_utils import parse_read_mode, read1
from tabulate import tabulate



FILE_PATH = "test_files/problema_real1.txt"

DEFAULT_ALPHA = 0.5
DEFAULT_TOL = 0.001
DEFAULT_NREPS = 10
DEFAULT_PERCENTAGE = 0.75

# prob_real1_perc.py [-hyper] [-a alpha] [-tol tolerance] [-nreps num_reps]
def read_input_params():
    alpha = DEFAULT_ALPHA
    tol = DEFAULT_TOL
    num_reps = DEFAULT_NREPS

    # reading input params alpha and/or tolerance (if specified)
    for (idx, parameter) in enumerate(sys.argv[1:]):
        if parameter == '-a':
            alpha = sys.argv[idx+2]
            if alpha <= 0 or alpha > 1:
                print("Error: alpha must be in (0, 1]")
                exit()
        if parameter == '-tol':
            tol = sys.argv[idx+2]
            if tol <= 0:
                print("Error: tolerance must be positive")
                exit()
        if parameter == '-nreps':
            num_reps = sys.argv[idx+2]
            if num_reps <= 0:
                print("Error: num_reps must be at least one")
                exit()

    return alpha, tol, num_reps




ALPHAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
TOLS = [0.1, 0.01, 0.001, 0.0001]

def val_hyperparams():
    headers = ["Thresholds \ Alphas"]
    for alpha in ALPHAS:
        headers.append(str(alpha))

    L_RES = []
    for tol in TOLS:
        L = [str(th)]
        for alpha in ALPHAS:
            mse, std = exec_real1(alpha, tol, DEFAULT_NREPS)
            L.append(str(mse)+' +- '+str(std))
    L_RES.append(L)
    print(tabulate(L_RES, headers=headers, tablefmt="grid"))



def exec_real1(alpha, tol, num_reps):
    mse_list = []
    for i in range(num_reps):
        # reading training and test sets
        sets = read1(FILE_PATH, DEFAULT_PERCENTAGE)
        xtrain, ytrain, xtest, ytest = sets
        # TODO: Coger una sola columna objetivo (target)
        n_inputs = len(xtrain[0])

        ada_nn = Adaline(n_inputs, alpha=alpha, tol=tol)

        ada_nn.train(xtrain, ytrain)
        ypred = ada_nn.predict(xtest)
        mse = ada_nn.error(ytest, ypred, metric='mse')
        mse_list.append(mse)

    mse_list = np.asarray(mse_list)
    return mse_list.mean(), mse_list.std()



if __name__ == '__main__':
    # hyperparameter validation
    if '-hyper' in sys.argv:
        val_hyperparams()
    # executing with specified parameters
    else:
        alpha, tol, num_reps = read_input_params()
        mse, std = exec_real1(alpha, tol, num_reps)
        print("Mean Squared Error:", mse, "+-", std)