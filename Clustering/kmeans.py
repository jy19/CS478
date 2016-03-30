from toolkitPython.matrix import Matrix, mode

import numpy as np
import sys
import random


class KMeans:
    def __init__(self, k, features, random_init, use_label):
        self.k = k
        self.matrix = features
        self.distances = []
        self.random_init = random_init
        self.use_label = use_label

    def run_kmeans(self):
        iterations = 0
        if self.random_init:
            centroids = self.random_init_centroid()
        else:
            centroids = self.matrix.data[:self.k]
        prev_centroids = []
        print 'num cols: ', self.matrix.cols
        clusters = []
        while not self.has_converged(prev_centroids, centroids):
            print 'iterations:', iterations + 1
            self.distances = []
            prev_centroids = centroids
            clusters = self.assign_centroids(prev_centroids)
            centroids = self.update_centroids(clusters)
            sse = sum(self.distances)
            print 'cluster lengths: ', ' '.join(str(len(x)) for x in clusters)
            print 'sse:', sse
            iterations += 1

        for i in xrange(len(centroids)):
            centroid_str = ''
            for j in xrange(len(centroids[i])):
                if self.matrix.value_count(j) == 0:
                    centroid_str += str(centroids[i][j]) + ','
                else:
                    centroid_str += self.matrix.attr_value(j, int(centroids[i][j])) + ', '
            print 'centroid {0}: {1}'.format(i, centroid_str)
        self.plot_silhouette(clusters)

    def calc_distance(self, centroid, curr_point):
        num_features = self.matrix.cols
        if not self.use_label:
            num_features -= 1
        distance = 0
        for x in xrange(num_features):
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
                # print curr_distance
                if curr_distance < min_dist:
                    cluster_index = i
                    min_dist = curr_distance
            clusters[cluster_index].append(point)
            # print point[0], ':', cluster_index
            self.distances.append(min_dist)
            min_dist = sys.maxint
        return clusters

    def update_centroids(self, clusters):
        num_features = self.matrix.cols
        updated_centroids = []
        for cluster in clusters:
            curr_centroid = []
            for x in xrange(num_features):
                curr_cols = [row[x] for row in cluster]
                # print 'curr cols:', curr_cols
                # use masked array to ignore missing values
                ma = np.ma.masked_equal(curr_cols, self.matrix.MISSING).compressed()
                if ma.size == 0:
                    curr_centroid.append(self.matrix.MISSING)
                    continue
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

    def plot_silhouette(self, clusters):
        # http://stackoverflow.com/a/6725320/5004049
        import matplotlib.pyplot as plt
        from matplotlib import cm

        X = []
        for cluster in clusters:
            X.extend([x for x in cluster])

        X = np.asarray(X)
        labels = []
        for x in xrange(len(clusters)):
            labels.extend([x for y in clusters[x]])

        labels = np.asarray(labels)
        # s = silhouette_score(X, labels)
        # print s
        s = silhouette(X, labels)

        order = np.lexsort((-s, labels))
        indices = [np.flatnonzero(labels[order] == k) for k in range(self.k)]
        ytick = [(np.max(ind) + np.min(ind)) / 2 for ind in indices]
        ytickLabels = ["%d" % x for x in range(self.k)]
        cmap = cm.jet(np.linspace(0, 1, self.k)).tolist()
        clr = [cmap[i] for i in labels[order]]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.barh(range(X.shape[0]), s[order], height=1.0,
                edgecolor='none', color=clr)
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.yticks(ytick, ytickLabels)
        plt.xlabel('Silhouette Value')
        plt.ylabel('Cluster')
        plt.savefig('abalone-silhouette-{0}.png'.format(self.k))


def silhouette(X, cIDX):
    """
    Computes the silhouette score for each instance of a clustered dataset,
    which is defined as:
        s(i) = (b(i)-a(i)) / max{a(i),b(i)}
    with:
        -1 <= s(i) <= 1

    Args:
        X    : A M-by-N array of M observations in N dimensions
        cIDX : array of len M containing cluster indices (starting from zero)

    Returns:
        s    : silhouette value of each observation
    """
    from scipy.spatial.distance import pdist, squareform

    N = X.shape[0]  # number of instances
    K = len(np.unique(cIDX))  # number of clusters

    # compute pairwise distance matrix
    D = squareform(pdist(X))

    # indices belonging to each cluster
    kIndices = [np.flatnonzero(cIDX == k) for k in range(K)]

    # compute a,b,s for each instance
    a = np.zeros(N)
    b = np.zeros(N)
    for i in range(N):
        # instances in same cluster other than instance itself
        a[i] = np.mean([D[i][ind] for ind in kIndices[cIDX[i]] if ind != i])
        # instances in other clusters, one cluster at a time
        b[i] = np.min([np.mean(D[i][ind])
                       for k, ind in enumerate(kIndices) if cIDX[i] != k])
    s = (b - a) / np.maximum(a, b)

    return s


def main():
    try:
        execname, arff_fn, k, normalize, random_init, use_label = sys.argv
    except ValueError:
        execname = sys.argv[0]
        print('usage: {0} arff_file k normalize(1|0) random_init(1|0) use_label(1|0)'.format(execname))
        sys.exit(-1)

    normalize = int(normalize)
    data = Matrix()
    data.load_arff(arff_fn)
    if normalize == 1:
        data.normalize()
    random_init = int(random_init)
    use_label = int(use_label)
    kmeans = KMeans(int(k), data, random_init, use_label)
    kmeans.run_kmeans()


if __name__ == '__main__':
    main()
