from toolkitPython.supervised_learner import SupervisedLearner
from toolkitPython.matrix import Matrix
import math
import copy


def find_item(li, item):
    return [ind for ind in xrange(len(li)) if item in li[ind]][0]


class Node:
    def __init__(self, partition_data, output_data, input_attrs):
        self.partition_data = partition_data
        self.output_data = output_data
        self.attributes = input_attrs
        self.chosen_attr = -1  # attribute to split on, init to -1 so don't get NoneType errors
        self.children = {}
        self.label = None


class DecisionTreeLearner(SupervisedLearner):
    def __init__(self):
        self.root_node = None
        self.UNKNOWN = 999.0
        self.MISSING = float("infinity")
        self.curr_acc = 0
        self.prune = False
        self.handle_real = False

    def modify_features_real(self, features):
        # modify real features so that they can be processed by ID3
        # import numpy as np
        col_bins = []
        for i in xrange(features.cols):
            col_bin = self.bin_values(features.col(i))
            col_bins.append(col_bin)
        # col_bins = np.array(col_bins)
        for i in xrange(features.rows):
            # features.data[i] = [np.argwhere(col_bins[j] == features.data[i][j])[0][0] for j in xrange(features.cols-1)]
            features.data[i] = [find_item(col_bins[j], features.data[i][j]) for j in xrange(features.cols)]
        # print "modified features", features.data
        return features

    def train(self, features, labels):
        # print "orig features:", features.data
        # handle real values
        if self.handle_real:
            features = self.modify_features_real(features)
        # print "features: ", features.data
        # replace missing values
        for i in xrange(features.rows):
            for j in xrange(features.rows):
                features.data[i] = [self.UNKNOWN if x == features.MISSING else x for x in features.data[i]]

        if self.prune:
            # divide features further into training and validation for pruning
            train_percent = 0.8
            curr_rows = features.rows
            curr_cols = features.cols
            training_features = Matrix(features, 0, 0, int(curr_rows * train_percent), curr_cols)
            training_labels = Matrix(labels, 0, 0, int(curr_rows * train_percent), curr_cols)
            validation_features = Matrix(features, int(curr_rows * train_percent), 0,
                                         int(curr_rows * (1 - train_percent)), curr_cols)
            validation_labels = Matrix(labels, int(curr_rows * train_percent), 0, int(curr_rows * (1 - train_percent)),
                                       curr_cols)
            input_data = training_features.data
            label_data = [label[0] for label in training_labels.data]
        else:
            input_data = features.data
            label_data = [label[0] for label in labels.data]

        # self.root_node = Node(features.data, [label[0] for label in labels.data], features.attr_names)
        self.root_node = Node(input_data, label_data, features.attr_names)
        self.create_tree()
        nodes, depth = self.get_tree_info(self.root_node)
        print "unpruned tree: deepest, nodes: ", depth, nodes
        # self.print_tree(self.root_node)

        if self.prune:
            self.curr_acc = self.measure_accuracy(validation_features, validation_labels)
            self.prune_tree(validation_features, validation_labels)

    def predict(self, features, labels):
        prediction = self.r_predict(self.root_node, features)
        labels.append(prediction)

    def r_predict(self, node, features):
        if not node.label:
            value = features[node.chosen_attr]
            if value == self.MISSING:
                try:
                    next_node = node.children[self.UNKNOWN]
                except KeyError:
                    return node.label if node.label is not None else 1
            else:
                try:
                    next_node = node.children[value]
                except KeyError:
                    return node.label if node.label is not None else 1
            prediction = self.r_predict(next_node, features)
        else:
            return node.label
        return prediction

    def get_attr_data(self, partition, attr_index):
        # get the data under the given attr index
        data = []
        for row in partition:
            if row[attr_index] not in data:
                data.append(row[attr_index])
        return data

    def get_partition_data(self, data, attr_index, value):
        partition = []
        for row in data:
            if row[attr_index] == value:
                new_row = []
                # add value if it's not in col
                for i in xrange(len(row)):
                    if i != attr_index:
                        new_row.append(row[i])
                partition.append(new_row)
        return partition

    def find_most_common(self, data):
        return max(set(data), key=data.count)

    def create_tree(self):
        self.r_create_tree(self.root_node, 0)

    def r_create_tree(self, curr_node, curr_depth):
        # print "curr depth:", curr_depth
        if len(set(curr_node.output_data)) == 1:
            # print '1 class left in output'
            # 1 class left, finish
            curr_node.label = curr_node.output_data[0]
            return
        elif not curr_node.attributes or not curr_node.partition_data:
            # print 'no attributes or data'
            # no attributes left or empty, label with most common class of parent
            curr_node.label = self.find_most_common(curr_node.output_data)
            return

        num_attrs = len(curr_node.attributes)
        # output_entropy = self.calc_entropy(curr_node.output_data)
        output_col_obj = self.get_col_outputs(curr_node.output_data)
        output_entropy = self.calc_entropy(output_col_obj)
        # print "output_entropy:", output_entropy
        info_gains = []
        # print num_attrs
        for i in xrange(num_attrs):
            # attr_col = np.array(curr_node.partition_data)[:, i]
            # print "curr node partition data", curr_node.partition_data
            # print "index: ", i
            attr_col = [row[i] for row in curr_node.partition_data]
            # print 'attr col: ', attr_col
            col_data = self.get_col_outputs(attr_col)
            # self.calc_info_gain(col_data, curr_node)
            curr_entropy = self.calc_info_gain(col_data, curr_node)
            info_gain = output_entropy - curr_entropy
            info_gains.append(info_gain)
        # print "info_gain array:", info_gains
        # best_attr_index = np.argmax(info_gains)
        best_attr_index = info_gains.index(max(info_gains))
        curr_node.chosen_attr = best_attr_index
        # print 'curr best attr index:', best_attr_index

        # create child nodes for each output in best attr field
        # best_attr_col = np.array(curr_node.partition_data)[:, best_attr_index]
        best_attr_col = [row[best_attr_index] for row in curr_node.partition_data]
        best_attr_outputs = self.get_col_outputs(best_attr_col)
        # print 'best attr outputs: ', best_attr_outputs
        for k, v in best_attr_outputs.iteritems():
            child_partition_data = [curr_node.partition_data[x] for x in v]
            child_output_data = [curr_node.output_data[x] for x in v]
            child_attributes = curr_node.attributes[:best_attr_index] + curr_node.attributes[best_attr_index + 1:]
            child_node = Node(child_partition_data, child_output_data, child_attributes)
            curr_node.children[k] = child_node
            self.r_create_tree(child_node, curr_depth + 1)

    def get_col_outputs(self, column):
        # get pos outputs of a column, and where they occur
        col_outputs = {}
        col_len = len(column)
        for i in xrange(col_len):
            try:
                col_outputs[column[i]].append(i)
            except KeyError:
                col_outputs[column[i]] = [i]
        return col_outputs

    def calc_info_gain(self, col_outputs, node):
        info_gain = 0
        output_total = sum([len(v) for v in col_outputs.itervalues()])
        for k, v in col_outputs.iteritems():
            label_attr = self.get_col_outputs([node.output_data[x] for x in v])
            info_gain += (len(v) / float(output_total)) * self.calc_entropy(label_attr)
        return info_gain

    def calc_entropy(self, col_outputs):
        entropy = 0
        output_total = sum([len(v) for v in col_outputs.itervalues()])
        for k, v in col_outputs.iteritems():
            x = len(v) / float(output_total)
            entropy += -x * math.log(x, 2)
        return entropy

    def prune_tree(self, validation_features, validation_labels):
        # reduce error pruning (for fully trained trees)
        # for each non leaf node, test accuracy on validation set for
        # modified tree where subtree of this node is removed and node is assigned majority class
        # keep pruned tree that does best on validation set and at least as well as original tree
        # repeat until no pruned tree does as well as original tree
        self.r_prune_tree(self.root_node, validation_features, validation_labels)
        nodes, depth = self.get_tree_info(self.root_node)
        print "Pruned tree, depth, nodes:", depth, nodes

    def r_prune_tree(self, curr_node, validation_features, validation_labels):
        if curr_node != self.root_node:  # don't prune the root node..
            if curr_node.chosen_attr != -1:  # not split on any attributes, is non-leaf node
                backup = copy.deepcopy(curr_node)
                # remove subtree and assign majority class
                curr_node.chosen_attr = -1
                curr_node.label = self.find_most_common(curr_node.output_data)
                curr_node.children = {}

                pruned_acc = self.measure_accuracy(validation_features, validation_labels)
                if pruned_acc < self.curr_acc:  # validation is not better, put subtree back
                    curr_node = backup

        # try pruning all children nodes
        for k, v in curr_node.children.iteritems():
            curr_node.children[k] = self.r_prune_tree(v, validation_features, validation_labels)

        return curr_node

    def get_tree_info(self, curr_node, depth=1, nodes=1):
        if curr_node.label:  # curr node is leaf node
            return nodes, depth

        depths = [depth]
        for k, v in curr_node.children.iteritems():
            nodes, curr_depth = self.get_tree_info(v, depth + 1, nodes + 1)
            depths.append(curr_depth)

        return nodes, max(depths)

    def bin_values(self, data, bins=3):
        # experiment to handle real value inut
        # assumes data normalized
        data = sorted(data)
        k, m = len(data) / bins, len(data) % bins
        return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(bins)]

    def print_tree(self, node):
        self.print_node(node, 0)

    def print_node(self, node, depth):
        if node.chosen_attr != -1:
            print "--" * depth, "Splits, On attribute: ", node.attributes[node.chosen_attr]
        else:
            print "--" * depth, "Stop split, label: ", node.label

        if node.children:
            print "--" * depth, "children: "
            for child in node.children.values():
                self.print_node(child, depth + 1)
