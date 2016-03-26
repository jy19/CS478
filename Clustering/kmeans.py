from toolkitPython.matrix import Matrix

import sys

class KMeans:
    def __init__(self, k, data):
        self.k = k
        self.data = data

    # get random centroids to start
    # keep iterating and changing centroids until no more change
    def run_kmeans(self):
        # centroid = randomly initialize centroid
        # prev_centroid = None
        # while changes()
            # prev_centroid = centroid
            # assign labels to data based on centoids
            # assign centroids based on new data labels
        pass

    def changes(self, prev_centroids, centroids):
        # stopping criteria: if centroids are not changing
        # or maybe if given a max num of iterations, it's exceeded?
        return prev_centroids == centroids


def main():
    try:
        execname, arff_fn, normalize = sys.argv
    except ValueError:
        execname = sys.argv[0]
        print('usage: {0} arff_file normalize(1|0)'.format(execname))
        sys.exit(-1)

    data = Matrix()
    data.load_arff(arff_fn)
    if normalize:
        data.normalize()

    k = 2
    kmeans = KMeans(k, data)


if __name__ == '__main__':
    main()
