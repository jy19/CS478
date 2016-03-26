from __future__ import print_function
from toolkitPython.matrix import Matrix
from toolkitPython.supervised_learner import SupervisedLearner

import time
import sys
import heapq
from collections import Counter
import numpy as np

class InstanceBasedLearner(SupervisedLearner):
    def __init__(self, k, distance_weighting, regression):
        self.k = k
        self.distance_weighting = distance_weighting
        self.regression = regression
        self.features = None
        self.labels = None

    def get_nearest_neighbors_np(self, test_instance):
        # calculate distances faster bc python for loops are slow and np uses C loops
        test_instance_np = np.asarray(test_instance)
        train_instances = np.asarray(self.features.data)
        distances = np.sum((train_instances - test_instance_np)**2, axis=1)
        # print('distances:', distances)

        # get k smallest elements
        dist_indices = np.argpartition(distances, self.k)[:self.k]
        neighbors = []
        # print('neighbors:')
        for index in dist_indices:
            # print('curr neighbor:', self.features.row(index))
            curr_tup = (distances[index], self.labels.row(index)[0])
            neighbors.append(curr_tup)

        return neighbors

    def get_nearest_neighbors(self, test_instance):
        k = self.k
        # distances = [(0, None)]*k
        distances = []
        num_training = self.features.rows
        num_features = self.features.cols
        # print('test instance:', test_instance)
        for x in range(num_training):
            curr_distance = calc_distance(num_features, test_instance, self.features.row(x))
            # print('train instance:', self.features.row(x))
            # print('curr distance: ', curr_distance)
            if self.distance_weighting:
                # curr_distance = 1 / (curr_distance * curr_distance)
                curr_distance = 1 / curr_distance
            else:
                curr_distance *= -1  # multiply distances by -1 for max heap.. kinda hack-ish
            curr_tup = (curr_distance, self.labels.row(x)[0])
            if len(distances) < k:
                heapq.heappush(distances, curr_tup)
            else:
                largest_dist = heapq.heappop(distances)
                if largest_dist[0] < curr_distance:
                    heapq.heappush(distances, curr_tup)
                else:
                    heapq.heappush(distances, largest_dist)
            # heapq.heappush(distances, curr_tup)

        return distances

    # def knn(self, test_instances, labels):
    #     # print('knn, test instances:', test_instances)
    #     num_tests = test_instances.rows
    #     for x in xrange(num_tests):
    #         neighbors = self.get_nearest_neighbors(test_instances.row(x))
    #         curr_class = determine_class(neighbors)
    #         labels.append(curr_class)

    def train(self, features, labels):
        # store feature vectors and class labels
        self.features = features
        self.labels = labels
        print('k:', self.k)
        print('distance weight:', self.distance_weighting)
        print('regression:', self.regression)

    def predict(self, features, labels):
        # print('test instance:', features)
        # neighbors = self.get_nearest_neighbors(features)
        neighbors = self.get_nearest_neighbors_np(features)
        # print('neighbors: ', neighbors)
        if self.regression:
            if self.distance_weighting:
                pass
            else:
                prediction = avg_mean_regression(neighbors)
        else:
            if self.distance_weighting:
                prediction = determine_class(neighbors, True)
            else:
                prediction = determine_class(neighbors, False)
        labels.append(prediction)


def calc_distance(num_features, test_instance, train_instance):
        ttl_distance = 0
        for x in range(num_features):
            diff = train_instance[x] - test_instance[x]
            ttl_distance += (diff * diff)
        return ttl_distance

def avg_mean_regression(neighbors):
        return sum(neighbor[0] for neighbor in neighbors) / len(neighbors)

def determine_class(neighbors, distance_weighting):
    if distance_weighting:
        from operator import itemgetter
        # vote weighted by distance
        votes = {}
        for neighbor in neighbors:
            try:
                votes[neighbor[1]] += (1 / neighbor[0])
            except KeyError:
                votes[neighbor[1]] = (1 / neighbor[0])
        sorted_votes = sorted(votes.iteritems(), key=itemgetter(1), reverse=True)
        return sorted_votes[0][0]
    else:
        votes = Counter(neighbor[1] for neighbor in neighbors)
        return votes.most_common(1)[0][0]

def normalize(data):
    data = np.array(data)
    data_norm = (data - data.min(0)) / data.ptp(0)
    return data_norm

def main():
    try:
        execname, train_fn, test_fn = sys.argv
    except ValueError:
        execname = sys.argv[0]
        print('usage: {0} train_fn test_fn'.format(execname))
        sys.exit(-1)

    regression = True
    weighting = False
    data = Matrix()
    data.load_arff(train_fn)
    data.normalize()
    # data.data = normalize(data.data)
    test_data = Matrix(arff=test_fn)
    test_data.normalize()
    # test_data.data = normalize(test_data.data)

    print("Test set name: {}".format(test_fn))
    print("Number of test instances: {}".format(test_data.rows))
    features = Matrix(data, 0, 0, data.rows, data.cols-1)
    labels = Matrix(data, 0, data.cols-1, data.rows, 1)

    test_features = Matrix(test_data, 0, 0, test_data.rows, test_data.cols-1)
    test_labels = Matrix(test_data, 0, test_data.cols-1, test_data.rows, 1)
    confusion = Matrix()

    accuracies = []
    for k in xrange(1, 10, 2):
        learner = InstanceBasedLearner(k, weighting, regression)
        start_time = time.time()
        learner.train(features, labels)
        elapsed_time = time.time() - start_time
        print("Time to train (in seconds): {}".format(elapsed_time))

        test_accuracy = learner.measure_accuracy(test_features, test_labels, confusion)
        accuracies.append((k, test_accuracy))
        elapsed_time = time.time() - start_time
        print("Test set accuracy: {}".format(test_accuracy))
        print("Time took to predict (in seconds): {}".format(elapsed_time))

    print('\n'.join('{0}: {1}'.format(tup[0], tup[1]) for tup in accuracies))

if __name__ == "__main__":
    main()
