from toolkitPython.supervised_learner import SupervisedLearner
from toolkitPython.matrix import Matrix

import numpy as np


def sigmoid_func(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


class Layer:
    def __init__(self, weights, is_hidden=False):
        self.weights = weights
        self.delta_weights = np.zeros(len(weights))
        self.bias = 1
        self.lr = 0.1
        self.momentum = 0
        self.input_data = None
        self.output_data = None
        self.target = None
        self.is_hidden = is_hidden

    def update_weights(self, delta_weights):
        self.delta_weights = delta_weights
        self.weights = self.weights + delta_weights

    def update_input(self, input_data):
        self.input_data = input_data

    def layer_predict(self):
        # predict the outputs given input
        data = self.input_data + [self.bias]
        # print data, self.weights.T
        non_adjusted_out = np.dot(data, self.weights.T)
        predicted_output = sigmoid_func(non_adjusted_out)
        self.output_data = predicted_output

    def calc_error(self, output_errors=None, output_weights=None):
        if self.is_hidden:
            error = output_errors.dot(output_weights.T)
        else:
            error = self.target - self.output_data
        return error

    def calc_delta(self, errors):
        data = self.input_data + [self.bias]
        delta_weights = []
        for i in xrange(len(data)):
            curr_delta = (self.lr * errors[i] * data) + (self.momentum * self.delta_weights[i])
            delta_weights.append(curr_delta)
        return delta_weights


class NeuralNetLearner(SupervisedLearner):
    def __init__(self, hidden_nodes, lr=0.1, momentum=0, epochs=10):
        self.output_layer = None
        self.hidden_layers = []
        self.lr = lr
        self.momentum = momentum
        self.hidden_nodes = hidden_nodes  # number of nodes in the hidden layer
        self.MAX_EPOCHS = epochs
        self.do_dropout = False
        self.dropout_percent = 0.2

    def train(self, features, labels):

        num_rows = features.rows
        num_cols = features.cols
        epochs = 0

        # create a validation set and training set for stopping criteria
        train_percent = 0.8
        training_features = Matrix(features, 0, 0, int(num_rows * train_percent), num_cols)
        training_labels = Matrix(labels, 0, 0, int(num_rows * train_percent), num_cols)
        validation_features = Matrix(features, int(num_rows * train_percent), 0,
                                     int(num_rows * (1 - train_percent)), num_cols)
        validation_labels = Matrix(labels, int(num_rows * train_percent), 0, int(num_rows * (1 - train_percent)),
                                   num_cols)

        # initialize random weights with mean 0, plus bias weight
        weights = 2 * np.random.random((len(training_features.data) + 1, 1)) - 1
        hidden_weights = 2 * np.random.random((self.hidden_nodes + 1, len(training_features.data)+1)) - 1

        self.output_layer = Layer(weights)
        self.hidden_layers.append(Layer(hidden_weights, True))
        error = 100
        no_improvement = 0
        while no_improvement < 3 and epochs < self.MAX_EPOCHS:
            # shuffle data every epoch
            for i in xrange(num_rows):
                self.output_layer.target = training_labels.row(i)[0]
                self.feed_forward(training_features.row(i))
                self.back_propagate()
            curr_error = self.measure_accuracy(validation_features, validation_labels)
            if abs(curr_error - error) < 0.001:
                no_improvement += 1
            else:
                no_improvement = 0
            error = curr_error
            epochs += 1
        print "epochs taken: ", epochs
        print "validation set error: ", error

    def predict(self, features, labels):
        self.feed_forward(features)
        labels.append(np.argmax(self.output_layer.output_data))

    def feed_forward(self, input_data):
        # feed forward to next layer
        curr_input_data = input_data
        for hidden_layer in self.hidden_layers:  # feed through hidden layers
            hidden_layer.update_input(curr_input_data)
            hidden_layer.layer_predict()
            if self.do_dropout:
                hidden_layer.output_data = self.dropout(hidden_layer.output_data)
            curr_input_data = hidden_layer.output_data
        # print "forward, predicted hidden: ", self.hidden_layer[0].output_data
        self.output_layer.update_input(self.hidden_layers[-1].output_data)
        self.output_layer.layer_predict()
        # print "forward, predicted output: ", self.output_layer.output_data
        if self.do_dropout:
            self.output_layer.output_data = self.dropout(self.output_layer.output_data)

    def back_propagate(self):
        output_errors = self.output_layer.calc_error()
        curr_hidden_errors = output_errors
        curr_weights = self.output_layer.weights
        for hidden_layer in reversed(self.hidden_layers):
            hidden_errors = hidden_layer.calc_error(curr_hidden_errors, curr_weights)
            curr_hidden_errors = hidden_errors
            curr_weights = hidden_layer.weights
            curr_deltas = hidden_layer.calc_delta(curr_hidden_errors)
            hidden_layer.update_weghts(curr_deltas)

        output_delta = self.output_layer.calc_delta(output_errors)
        self.output_layer.update_weights(output_delta)
        # print "output layer new weights: ", self.output_layer.weights
        # get output layer's errors
        # get hidden layers' errors based on the output layer's errors
        # adjust weights by calculating delta weights of each layer based on prev layer's output

    def dropout(self, output_data):
        # randomly set a node to 0
        output_data *= np.random.binomial([np.ones((len(self.output_layer.output_data), self.hidden_nodes))],
                                          1 - self.dropout_percent)[0] * (1.0 / (1 - self.dropout_percent))
        # print "drop out output: ", output_data
        return output_data
