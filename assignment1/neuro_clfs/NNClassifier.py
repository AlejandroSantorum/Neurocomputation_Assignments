
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

    def error(self, ytrue, ypred):
        hit_rate = sum(ytrue == ypred)/len(ypred)
        return 1-hit_rate # miss rate is the complement of hit rate
