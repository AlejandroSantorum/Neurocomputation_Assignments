
from abc import ABCMeta,abstractmethod
import numpy as np


class NNClassifier:

    # abstract class
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, xtrain, ytrain):
        pass

    @abstractmethod
    def predict(self, xtest):
        pass

    def score(self, xtest, ytest):
        ypred = self.predict(xtest)
        return sum(ytest == ypred)/len(ypred)

    def error(self, ytrue, ypred, metric='acc'):
        # Accuracy
        if metric == 'acc':
            n_test = len(ytrue)
            hit_rate = 0
            for i in range(n_test):
                if np.array_equal(ytrue[i], ypred[i]):
                    hit_rate += 1
            hit_rate /= n_test
            # miss rate is the complement of hit rate
            return 1-hit_rate
        # Mean Squared Error
        if metric == 'mse': # TODO: review
            n_test = len(ytrue)
            n_outputs = len(ytrue[0])
            S = [0]*n_outputs
            for j in range(n_outputs):
                for i in range(n_test):
                    S[j] += (ytrue[i][j] - ypred[i][j])**2
            S = np.asarray(S)
            return (1/n_test)*np.sum(S)
