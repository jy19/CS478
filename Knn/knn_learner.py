from __future__ import print_function
from toolkitPython.supervised_learner import SupervisedLearner

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

        return distances

    def reduce_train_set(self, data):
        data.shuffle()
        data.data = data.data[:2000]
        return data

    def train(self, features, labels):
        # store feature vectors and class labels
        self.features = features
        self.labels = labels
        for i in xrange(self.features.rows):
            self.features.data[i] = [1 if x == self.features.MISSING else x for x in self.features.data[i]]
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
                prediction = avg_mean_regression(neighbors, True)
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

def avg_mean_regression(neighbors, weighting=False):
    if weighting:
        weighted_distances = [(1/neighbor[0]) for neighbor in neighbors]
        weighted_output = 0
        for i in xrange(len(neighbors)):
            weighted_output += (weighted_distances[i] * neighbors[i][1])
        return weighted_output / sum(weighted_distances)
    else:
        return sum(neighbor[1] for neighbor in neighbors) / len(neighbors)

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
