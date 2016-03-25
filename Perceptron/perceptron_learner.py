from __future__ import print_function
from toolkitPython.supervised_learner import SupervisedLearner

import random

class PerceptronLearner(SupervisedLearner):
    def __init__(self):
        self.threshold = 0
        self.lr = 0.1
        self.bias = 0
        self.weights = []

    def perceptron_algo(self, features):
        for feature in features:
            if self.activation_function(feature):
                # correct
                pass
            else:
                # incorrect
                # adjust weight
                pass

    def activation_function(self, curr_features):
        activation = 0
        weights = len(self.weights)
        for i in xrange(weights):
            activation += self.weights[i] * curr_features[i]
        if activation > self.threshold:
            return 1
        return 0

    def adjust_weight(self, curr_features):
        # dw = (target - actual) * lr * weight
        pass

    def train(self, features, labels):
        # init
        # set all weights to small +/- random numbers
        # train for T iterations or until all outputs correct

        # init weights
        num_weights = features.cols
        for i in xrange(num_weights):
            self.weights.append(random.randrange(-1, 1))

        # while there are errors:
        # run perceptron
        # epoch+=1

        features.print()
        print(features.rows)
        print(features.cols)
        # labels.print()  # the names

    def predict(self, features, labels):
        pass
