from toolkitPython.supervised_learner import SupervisedLearner

# import numpy as np
import math


class Node:
    def __init__(self, partition_data, output_data, input_attrs):
        self.partition_data = partition_data
        self.output_data = output_data
        self.attributes = input_attrs
        self.chosen_attr = -1  # attribute to split on
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
        self.root_node = Node(features.data, [label[0] for label in labels.data], features.attr_names)
        self.create_tree()
        # self.print_tree(self.root_node)

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

    # def calc_entropy(self, data_subset, col_data=None):
    #     num_rows = len(data_subset)
    #     value_counts = {}
    #     entropy = 0
    #     for i in xrange(num_rows):
    #         try:
    #             value_counts[data_subset[i]] += 1
    #         except KeyError:
    #             value_counts[data_subset[i]] = 1
    #     for count in value_counts.values():
    #         x = float(count) / num_rows
    #         entropy += -x * math.log(x, 2)
    #
    #     return entropy

    # def calc_info_gain(self, data, output_data):
    #     num_rows = len(data)
    #     value_counts = {}
    #     partition_entropy = 0
    #     for i in xrange(num_rows):
    #         try:
    #             value_counts[data[i]] += 1
    #         except KeyError:
    #             value_counts[data[i]] = 1
    #     for k in value_counts.keys():
    #         val_prob = float(value_counts[k]) / sum(value_counts.values())
    #         subset = [entry for entry in data if entry == k]
    #         partition_entropy += val_prob * self.calc_entropy(subset)
    #
    #     return partition_entropy

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

            # calculate info gain of attributes in data set
            # split data set into subsets by max info gain, create node for each partition
            # for each partition:
            # if 1 class or if stop criteria: end
            # elif class > 1, and attributes: recurse back to calc info gain of all remaining attributes
            # else when no attri: end and label with most common class of parent
            # if empty: label with most common class of

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

    def prune_tree(self):
        # reduce error pruning (for fully trained trees)
        # for each non leaf node, test accuracy on validation set for
        # modified tree where subtree of this node is removed and node is assigned majority class
        # keep pruned tree that does best on validation set and at least as well as original tree
        # repeat until no pruned tree does as well as original tree
        pass

    def bin_values(self):
        # experiment to handle real value input
        pass

    def print_tree(self, node):
        pass
