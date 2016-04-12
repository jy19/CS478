from toolkitPython.supervised_learner import SupervisedLearner

import numpy as np

def sigmoid_func(x, deriv=False):
        if deriv:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

class Layer:
    def __init__(self, weights):
        self.weights = weights
        self.prev_weights = []
        self.bias = 1
        self.input_data = None
        self.output_data = None

    def update_weights(self, delta_weights):
        self.prev_weights = self.weights
        self.weights = self.weights + delta_weights

    def update_input(self, input_data):
        self.input_data = input_data

    def calc_error(self):
        pass

    def change_weights(self, errors):
        pass

class NeuralNetLearner(SupervisedLearner):
    def __init__(self, hidden_nodes, lr=0.1, momentum=0, epochs=10):
        self.output_layer = None
        self.hidden_layers = []
        self.lr = lr
        self.momentum = momentum
        self.hidden_nodes = hidden_nodes  # number of nodes in the hidden layer
        self.features = []
        self.labels = []
        self.MAX_EPOCHS = epochs
        self.curr_acc = 0
        self.curr_best_params = []

    def train(self, features, labels):
        self.features = features
        self.labels = labels

        num_rows = features.rows
        epochs = 0
        weights = 2*np.random.random((features.cols, 1)) - 1  # initialize random weights with mean 0
        hidden_weights = 2*np.random.random((self.hidden_nodes, 1)) - 1

        self.output_layer = Layer(weights)
        self.hidden_layers.append(Layer(hidden_weights))
        error = 100
        no_improvement = 0
        while no_improvement < 3 and epochs < self.MAX_EPOCHS:
            # shuffle data every epoch
            for i in xrange(num_rows):
                self.feed_forward(features.row(i))
            # calc error
            # if error doesn't change much, no_improvement += 1 else no_improvement = 0
            epochs += 1

    def predict(self, features, labels):
        pass

    def feed_forward(self, input):
        # feed forward to next layer
        for hidden_layer in self.hidden_layers:
            # update inputs to hidden layer
            # do delta rule on inputs
            # get outputs
            pass
        # update output layer inputs from hidden layer's outputs
        # do delta rule on inputs
        # get outputs based on delta rule results

    def back_propagate(self):
        # get output layer's errors
        # get hidden layers' errors baseon the output layer's errors
        # adjust weights
        pass
