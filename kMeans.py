import numpy
import csv
import logging
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.cm as cm
import random
import time
logging.basicConfig(filename="test.log", level=logging.DEBUG)
style.use('ggplot')


class kMeans:
    '''Initialize K-Means parameters @k - k,@r - number of iterations,@t - tolerance'''

    def __init__(self, k=3, r=10, t=0.0001):
        self.k = k
        self.r = r
        self.t = t
        self.rss = numpy.empty((r, 1))

    '''Cluster data into k clurster.@data - (n_rows,n_features)'''

    def clusterData(self, data):
        self.centroids = {}

        for c in range(self.k):
            randomPoint = random.randrange(1, len(data))
            self.centroids[c] = data[randomPoint]
        # end

        for r in range(self.r):
            self.clusters = {}

            for c in range(self.k):
                self.clusters[c] = []
            # end

            for d in data:
                distances = [numpy.linalg.norm(
                    d - self.centroids[clusterMean]) for clusterMean in self.centroids]
                assignment = distances.index(min(distances))
                self.clusters[assignment].append(d)
            # end
            oldCentroids = dict(self.centroids)
            for c in self.clusters:
                self.centroids[c] = numpy.average(self.clusters[c], axis=0)
            # end
                converged = True

            for centroid in self.centroids:
                old = oldCentroids[centroid]
                current = self.centroids[centroid]
                if numpy.sum((current-old)/old*100) > self.t:
                    converged = False
                # end
            if converged:
                break
            # end
            self.rss[r, 0] = sum(((x - self.centroids[c])**2).sum()
                                 for x, c in zip(data, self.clusters))
        # end
    # end


'''Starting point'''


def main():
    random.seed(time.time())
    print("Loading data set...")
    data = numpy.loadtxt('GMM_dataset.txt', dtype=float)
    km = kMeans(k=5, r=10, t=0.001)
    km.clusterData(data)
    print("K Means Centroids - ", km.centroids)
    print("K Means Sum Of Squared Error - ", km.rss)
    # Plotting starts here
    plt.subplot(121)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title("K Means Clustering With K = "+str(km.k))
    colors = iter(cm.gist_rainbow(numpy.linspace(0, 1, len(km.clusters))))
    for centroid in km.centroids:
        plt.scatter(km.centroids[centroid][0],
                    km.centroids[centroid][1], color="k", s=100, marker="*")

    for classification in km.clusters:
        color = next(colors)
        for features in km.clusters[classification]:
            plt.scatter(features[0], features[1],
                        color=color, s=10, marker="o")
        # end
    # end
    plt.subplot(122)
    plt.plot(km.rss)
    plt.title("Sum Of Squared Error vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Sum Of Squared Error")
    plt.show()


if __name__ == "__main__":
    main()
