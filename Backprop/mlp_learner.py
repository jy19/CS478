from toolkitPython.supervised_learner import SupervisedLearner

import numpy as np

class Layer:
    def __init__(self):
        self.weights = []
        self.prev_weights = []

    def sigmoid_func(self, x, deriv=False):
        if deriv:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def update_weights(self, delta_weights):
        self.weights = self.weights + delta_weights

    def calc_error(self):
        pass


class NeuralNetLearner(SupervisedLearner):
    def __init__(self, lr=0.1, momentum=0, epochs=10):
        self.layers = []
        self.lr = lr
        self.momentum = momentum
        self.features = []
        self.labels = []
        self.MAX_EPOCHS = epochs

    def train(self, features, labels):
        self.features = features
        self.labels = labels

    def predict(self, features, labels):
        pass

    def feed_forward(self):
        """
        feed forward to next layers
        :return:
        """

    def back_propagate(self):
        pass


