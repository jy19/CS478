from __future__ import print_function
from toolkitPython.supervised_learner import SupervisedLearner

import random
import numpy as np
import matplotlib.pyplot as plt

class PerceptronLearner(SupervisedLearner):
    def __init__(self, max_epochs=5000):
        self.threshold = 0
        self.lr = 0.1
        self.bias = 0
        self.weights = []
        self.MAX_EPOCHS = max_epochs

    def plot_scatter(self, features, labels):
        figure = plt.figure()
        ax1 = figure.add_subplot(111)
        x1, x2, y1, y2 = [], [], [], []
        for x in xrange(labels.rows):
            if labels.row(x)[0] == 0.0:
                x1.append(features.row(x)[0])
                y1.append(features.row(x)[1])
            else:
                x2.append(features.row(x)[0])
                y2.append(features.row(x)[1])
        ax1.scatter(x1, y1, c='b', marker='s', label='class 0')
        ax1.scatter(x2, y2, c='r', marker='o', label='class 1')

        x = np.arange(-0.6, 0.6, 0.1)
        y = []
        m, fx, b = self.weights[0], self.weights[1], self.weights[2]
        for point in x:
            y.append(((-m * point) + b) / fx)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='upper right')
        print('weights: ', self.weights)
        plt.plot(x, y)
        plt.show()
        # plt.savefig('Perceptron/notLinearlySeparable.png')

    def run_perceptron(self, feature, label):
        predicted = self.activation_function(feature)
        delta_weights = []
        target = label[0]
        if predicted != target:
            # not expected output
            # adjust weights
            for x in xrange(len(feature)):
                curr_dw = self.adjust_weight(target, predicted, feature[x])
                self.weights[x] += curr_dw
                delta_weights.append(curr_dw)
            bias_delta = self.adjust_weight(target, predicted, self.bias)
            self.weights[-1] += bias_delta
            delta_weights.append(bias_delta)
        else:
            delta_weights = np.zeros(len(feature) + 1)
        return delta_weights

    def activation_function(self, curr_feature):
        activation = 0
        num_weights = len(curr_feature)
        for i in xrange(num_weights):
            activation += self.weights[i] * curr_feature[i]
        activation += self.bias * self.weights[-1]  # account for the bias
        if activation > self.threshold:
            return 1
        return 0

    def adjust_weight(self, target, predicted, curr_weight):
        # dw = (target - predicted) * lr * weight
        delta_weight = (target - predicted) * self.lr * curr_weight
        return delta_weight

    def train(self, features, labels):
        num_labels = labels.rows
        # init
        # set all weights to small +/- random numbers
        # train for T iterations or until all outputs correct

        # init weights
        num_weights = features.cols
        for i in xrange(num_weights):
            self.weights.append(random.randrange(-1, 1))
        self.weights.append(0)  # append a weight for the bias

        not_changed = 0  # keep track of how many loops there are no improvements
        epochs = 0
        while not_changed < 3 and epochs < self.MAX_EPOCHS:
            curr_epoch_weights = np.zeros(num_weights + 1)
            for x in xrange(num_labels):
                delta_weights = self.run_perceptron(features.row(x), labels.row(x))
                curr_epoch_weights += delta_weights

            if curr_epoch_weights.any():  # if there are any changes in weights
                not_changed = 0
            else:
                not_changed += 1
            epochs += 1
            features.shuffle(labels)  # shuffle the data
        print("weights:", self.weights)
        print("Epochs taken to train:", epochs)
        # self.plot_scatter(features, labels)

    def predict(self, features, labels):
        result = self.activation_function(features)
        labels.append(result)
