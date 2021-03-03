
from abc import ABCMeta,abstractmethod


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
            hit_rate = sum(ytrue == ypred)/len(ypred)
            # miss rate is the complement of hit rate
            return 1-hit_rate
        # Mean Squared Error
        if metric == 'mse':
            n_test = len(ytrue)
            s = 0
            for i in range(n_test):
                s += (ytrue[i] - ypred[i])**2
            return (1/n_test)*s
