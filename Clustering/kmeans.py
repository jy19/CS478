from toolkitPython.matrix import Matrix, mode

import numpy as np
import sys
import random

class KMeans:
    def __init__(self, k, features):
        self.k = k
        self.matrix = features

    # get random centroids to start
    # keep iterating and changing centroids until no more change
    def run_kmeans(self):
        iterations = 0
        # centroids = self.random_init_centroid()
        centroids = self.matrix.data[:5]  # for testing purposes
        prev_centroids = []
        while not self.has_converged(prev_centroids, centroids):
            print '==========================centroids============================='
            print centroids
            print 'iteration:', iterations
            prev_centroids = centroids
            clusters = self.assign_centroids(prev_centroids)
            centroids = self.update_centroids(clusters)
            iterations += 1

    def calc_distance(self, centroid, curr_point):
        num_features = self.matrix.cols
        distance = 0
        for x in xrange(1, num_features):  # ignore first feature since that's id
            if self.matrix.value_count(x) == 0:  # feature is continuous
                if centroid[x] == self.matrix.MISSING or curr_point[x] == self.matrix.MISSING:
                    curr_dist = 1
                else:
                    curr_dist = centroid[x] - curr_point[x]
            else:  # feature is nominal
                if centroid[x] == self.matrix.MISSING or curr_point[x] == self.matrix.MISSING:
                    curr_dist = 1
                elif centroid[x] == curr_point[x]:
                    curr_dist = 0
                else:
                    curr_dist = 1
            distance += (curr_dist * curr_dist)
        return distance

    def assign_centroids(self, centroids):
        clusters = [[] for x in range(self.k)]
        data = self.matrix.data
        cluster_index = 0
        min_dist = sys.maxint
        for point in data:
            # calc distance between point to all centroids
            # get min and assign point to that centroid
            for i in xrange(self.k):
                curr_distance = self.calc_distance(centroids[i], point)
                if curr_distance < min_dist:
                    cluster_index = i
                    min_dist = curr_distance
            clusters[cluster_index].append(point)
            min_dist = sys.maxint
        return clusters

    def update_centroids(self, clusters):
        num_features = self.matrix.cols
        updated_centroids = []
        for cluster in clusters:
            curr_centroid = []
            for x in xrange(num_features):
                curr_cols = [row[x] for row in cluster]
                # use masked array to ignore missing values
                ma = np.ma.masked_equal(curr_cols, self.matrix.MISSING).compressed()
                if self.matrix.value_count(x) == 0:  # col is continuous
                    curr_value = np.mean(ma)
                else:  # col is nominal
                    mode_val, freq = mode(ma)
                    curr_value = mode_val[0]
                curr_centroid.append(curr_value)
            updated_centroids.append(curr_centroid)
        return updated_centroids

    def has_converged(self, prev_centroids, centroids):
        # stopping criteria: if centroids are not changing
        # possibly also add a max iterations stopping criteria
        return sorted(prev_centroids) == sorted(centroids)

    def random_init_centroid(self):
        return random.sample(self.matrix.data, self.k)


def main():
    try:
        execname, arff_fn, normalize = sys.argv
    except ValueError:
        execname = sys.argv[0]
        print('usage: {0} arff_file normalize(1|0)'.format(execname))
        sys.exit(-1)

    normalize = int(normalize)
    data = Matrix()
    data.load_arff(arff_fn)
    print 'normalize?', normalize
    if normalize == 1:
        data.normalize()

    k = 5
    kmeans = KMeans(k, data)
    kmeans.run_kmeans()


if __name__ == '__main__':
    main()
