from __future__ import print_function
# from toolkitPython.matrix import Matrix
from toolkitPython.matrix import Matrix
from toolkitPython.supervised_learner import SupervisedLearner

import time
import sys
# import math
# from operator import itemgetter
import heapq
from collections import Counter
# from scipy.spatial import distance as scp_distance

class InstanceBasedLearner(SupervisedLearner):
    def __init__(self, k, distance_weighting, regression):
        self.k = k
        self.distance_weighting = distance_weighting
        self.regression = regression
        self.features = None
        self.labels = None

    def get_nearest_neighbors(self, test_instance):
        k = self.k
        # distances = [(0, None)]*k
        distances = []
        num_training = self.features.rows
        num_features = self.features.cols
        print('test instance:', test_instance)
        for x in range(num_training):
            curr_distance = calc_distance(num_features, test_instance, self.features.row(x))
            print('train instance:', self.features.row(x))
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
        # distances.sort()
        # neighbors = []
        # for x in xrange(self.k):
        #     # neighbors.append(distances[x])
        #     neighbors.append(heapq.heappop(distances))
        # print(neighbors)
        # return neighbors

    # def get_nearest_neighbors2(self, test_instance):
    #     k = self.k
    #     distances = [(sys.maxint, None)] * (k+1)
    #     num_training = self.features.rows
    #     print('test_instance: ', test_instance)
    #     for x in xrange(num_training):
    #         print('curr feature: ', self.features.row(x))
    #         curr_distance = scp_distance.euclidean(self.features.row(x), test_instance)
    #         if self.distance_weighting:
    #             curr_distance = 1 / (curr_distance * curr_distance)
    #         curr_tup = (curr_distance, self.labels.row(x)[0])
    #         if curr_tup[0] < distances[0][0]:
    #             distances[0] = curr_tup
    #             for i in xrange(k):
    #                 if distances[i][0] >= distances[i+1][0]:
    #                     break
    #                 distances[i], distances[i+1] = distances[i+1], distances[i]

        # return distances[1:]
        # neighbors = []
        # for x in xrange(k):
        #     neighbors.append(heapq.heappop(distances))
        # return neighbors

    def knn(self, test_instances, labels):
        # print('knn, test instances:', test_instances)
        num_tests = test_instances.rows
        for x in xrange(num_tests):
            neighbors = self.get_nearest_neighbors(test_instances.row(x))
            curr_class = determine_class(neighbors)
            labels.append(curr_class)

    def train(self, features, labels):
        # store feature vectors and class labels
        # features.print()  # data
        # labels.print()  # classification
        # print('training, feature rows: ', features.rows)  # num rows of data
        # print('training, feature cols: ', features.cols)  # num of features
        # print('training, labels rows:', labels.rows)
        self.features = features
        self.labels = labels
        print('k:', self.k)
        print('distance weight:', self.distance_weighting)
        print('regression:', self.regression)

    def predict(self, features, labels):
        # print('predict features: ', features)
        neighbors = self.get_nearest_neighbors(features)
        # neighbors = self.get_nearest_neighbors2(features)
        # print('neighbors: ', neighbors)
        if self.regression:
            prediction = avg_mean_regression(neighbors)
        else:
            prediction = determine_class(neighbors)
        # print('prediction: ', prediction)
        labels.append(prediction)
        # self.knn(features, labels)


def calc_distance(num_features, test_instance, train_instance):
        ttl_distance = 0
        for x in range(num_features):
            diff = train_instance[x] - test_instance[x]
            # ttl_distance += pow((train_instance[x] - test_instance[x]), 2)
            ttl_distance += (diff * diff)
        # return math.sqrt(ttl_distance)
        return ttl_distance

# def calc_distance_scp(training_set, test_instance):
#     distances = []
#     num_training = training_set.rows
#     for x in num_training:
#         curr_distance = scp_distance.euclidean(training_set.row(x), test_instance)
#         distances.append(curr_distance)
#     return distances
def avg_mean_regression(neighbors):
        return sum(neighbor[0]*-1 for neighbor in neighbors) / len(neighbors)

def determine_class(neighbors):
        # class_count = {}  # dictionary to keep track of class and number of neighbors that voted that
        # for neighbor in neighbors:
        #     # print('determining class: ', neighbor[2])
        #     # curr_class = neighbor[2]
        #     curr_class = neighbor[1]
        #     try:
        #         class_count[curr_class] += 1
        #     except KeyError:
        #         class_count[curr_class] = 1
        # sorted_classes = sorted(class_count.iteritems(), key=itemgetter(1), reverse=True)
        # return sorted_classes[0][0]
        votes = Counter(neighbor[1] for neighbor in neighbors)
        return votes.most_common(1)[0][0]

def main():
    try:
        execname, train_fn, test_fn, weighting, regression = sys.argv
    except ValueError:
        execname = sys.argv[0]
        print('usage: {0} train_fn test_fn weighting regression'.format(execname))
        sys.exit(-1)

    data = Matrix()
    data.load_arff(train_fn)
    test_data = Matrix(arff=test_fn)
    test_data.normalize()

    print("Test set name: {}".format(test_fn))
    print("Number of test instances: {}".format(test_data.rows))
    features = Matrix(data, 0, 0, data.rows, data.cols-1)
    labels = Matrix(data, 0, data.cols-1, data.rows, 1)

    test_features = Matrix(test_data, 0, 0, test_data.rows, test_data.cols-1)
    test_labels = Matrix(test_data, 0, test_data.cols-1, test_data.rows, 1)
    confusion = Matrix()

    # for k in xrange(1, 16, 2):
    for k in xrange(1, 2):
        learner = InstanceBasedLearner(k, bool(weighting), bool(regression))
        start_time = time.time()
        learner.train(features, labels)
        elapsed_time = time.time() - start_time
        print("Time to train (in seconds): {}".format(elapsed_time))

        test_accuracy = learner.measure_accuracy(test_features, test_labels, confusion)
        elapsed_time = time.time() - start_time
        print("Test set accuracy: {}".format(test_accuracy))
        print("Time took to predict (in seconds): {}".format(elapsed_time))

if __name__ == "__main__":
    main()
