from toolkitPython.supervised_learner import SupervisedLearner

import numpy as np
import math

class Node:
    def __init__(self, partition_data, output_data, attrs, target_attrs):
        self.partition_data = partition_data
        self.output_data = output_data
        self.attributes = attrs
        self.target_attrs = target_attrs
        self.chosen_attr = None  # attribute to split on
        self.children = {}
        self.label = None

class DecisionTreeLearner(SupervisedLearner):
    def __init__(self):
        self.features = None
        self.labels = None
        self.root_node = None

    def train(self, features, labels):
        # replace missing values with 1
        # for i in xrange(features.rows):
        #     features.data[i] = [1 if x == features.MISSING else x for x in features.data[i]]
        self.features = features
        self.labels = labels
        self.root_node = Node(features.data, labels.data, features.attr_names, labels.attr_names)
        self.create_tree()

    def predict(self, features, labels):
        pass

    def find_most_common(self):
        pass

    def calc_entropy(self, data_subset, attr_index):
        num_rows = len(data_subset)
        value_counts = {}
        entropy = 0
        for i in xrange(num_rows):
            try:
                value_counts[data_subset[i][attr_index]] += 1
            except KeyError:
                value_counts[data_subset[i][attr_index]] = 1

        for count in value_counts.values():
            entropy += (-count/num_rows) * math.log(count/num_rows, 2)

        return entropy

    def calc_info_gain(self, data, attributes):
        num_attrs = len(attributes)
        num_rows = len(data)
        info_gains = []
        for attr_index in xrange(num_attrs):
            curr_entropy = self.calc_entropy(data, attr_index)
            value_counts = {}
            partition_entropy = 0
            for i in xrange(num_rows):
                try:
                    value_counts[data[i][attr_index]] += 1
                except KeyError:
                    value_counts[data[i][attr_index]] = 1
            for k in value_counts.keys():
                val_prob = value_counts[k] / sum(value_counts.values())
                subset = [entry for entry in data if entry[attr_index] == k]
                partition_entropy += val_prob * self.calc_entropy(subset, attr_index)

            info_gain = curr_entropy - partition_entropy
            info_gains.append(info_gain)
        return info_gains

    def create_tree(self):
        pass

    def r_create_tree(self, curr_node, curr_depth):
        curr_depth += 1

        if len(set(curr_node.output_data)) == 1:
            # 1 class left, finish
            curr_node.label = curr_node.output_data[0]
            return
        elif not curr_node.attributes:
            # empty, label with most common class of parent
            return
        info_gains = self.calc_info_gain(curr_node.partition_data, curr_node.attributes)
        curr_node.chosen_attr = np.argmax(info_gains)
        # calculate info gain of attributes in data set
        # split data set into subsets by max info gain, create node for each partition
        # for each partition:
            # if 1 class or if stop criteria: end
            # elif class > 1, and attributes: calc info gain of all remaining attributes
                # else when no attri: end and label with most common class of parent
            # if empty: label with most common class of parent

