from toolkitPython.supervised_learner import SupervisedLearner

import numpy as np
import math


class Node:
    def __init__(self, partition_data, output_data, input_attrs):
        self.partition_data = partition_data
        self.output_data = output_data
        self.attributes = input_attrs
        self.chosen_attr = None  # attribute to split on
        self.children = {}
        self.label = None


class DecisionTreeLearner(SupervisedLearner):
    def __init__(self):
        self.features = None
        self.labels = None
        self.root_node = None
        self.UNKNOWN = 999.0
        self.MISSING = float("infinity")

    def train(self, features, labels):
        # replace missing values
        for i in xrange(features.rows):
            features.data[i] = [self.UNKNOWN if x == features.MISSING else x for x in features.data[i]]
        self.features = features
        self.labels = labels
        self.root_node = Node(features.data, labels.data, features.attr_names)
        self.create_tree()

    def predict(self, features, labels):
        prediction = self.r_predict(self.root_node, features.data[0])
        labels.append(prediction)

    def r_predict(self, node, features):
        if not node.label:
            value = features[node.chosen_attr]
            if value == self.MISSING:
                try:
                    next_node = node.children[self.UNKNOWN]
                except KeyError:
                    return node.label or 1
            else:
                try:
                    next_node = node.children[value]
                except KeyError:
                    return node.label or 1
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
            entropy += (-count / num_rows) * math.log(count / num_rows, 2)

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
        self.r_create_tree(self.root_node)

    def r_create_tree(self, curr_node):
        # curr_depth += 1

        if len(set(curr_node.output_data)) == 1:
            # 1 class left, finish
            curr_node.label = curr_node.output_data[0]
            return
        elif not curr_node.attributes or not curr_node.partition_data:
            # no attributes left or empty, label with most common class of parent
            curr_node.label = self.find_most_common(curr_node.target_data)
            return

        info_gains = self.calc_info_gain(curr_node.partition_data, curr_node.attributes)
        best_attr_index = np.argmax(info_gains)
        curr_node.chosen_attr = best_attr_index

        # todo create child nodes for each output in best attr field
        for value in self.get_attr_data(curr_node.partition_data, best_attr_index):
            child_partition_data = self.get_partition_data(curr_node.partition_data, best_attr_index, value)
            child_output_data = [curr_node.output_data[i] for i in xrange(len(curr_node.partition_data))]
            child_attributes = curr_node.attributes[:best_attr_index] + curr_node.attributes[best_attr_index + 1:]
            child_node = Node(child_partition_data, child_output_data, child_attributes)
            curr_node.children[value] = child_node
            self.r_create_tree(child_node)

            # calculate info gain of attributes in data set
            # split data set into subsets by max info gain, create node for each partition
            # for each partition:
            # if 1 class or if stop criteria: end
            # elif class > 1, and attributes: recurse back to calc info gain of all remaining attributes
            # else when no attri: end and label with most common class of parent
            # if empty: label with most common class of parent
