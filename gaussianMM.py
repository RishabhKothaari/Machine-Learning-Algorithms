import numpy
from numpy import linalg as lg
from numpy import seterr
import csv
import logging
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.cm as cm
import random
import time
import kMeans
from scipy.stats import multivariate_normal
from math import log
from matplotlib.patches import Ellipse
import matplotlib.ticker as ticker
logging.basicConfig(filename="test.log", level=logging.DEBUG)
numpy.set_printoptions(threshold=numpy.nan)
style.use('ggplot')

enableLog = True


def log(messages={}):
    global enableLog
    if(enableLog):
        for k in messages:
            logging.debug(k+"{}".format(messages[k]))
    # end
# end


class gMM:
    '''Initilialize GMM centroids,clusters and mixing co-efficients.@data - (n_rows,n_features),@r - runs,@centroids - (n_clusters[(n_features)])
    @clusters - n_clusters[clusterpoints] received from K means'''

    def __init__(self, data, r, centroids, clusters, t):
        self.r = r
        self.data = data
        self.t = t
        self.clusters = numpy.empty((len(clusters)), dtype=object)
        self.responsibilities = numpy.empty(
            (data.shape[0], len(clusters)), dtype=float)
        self.covs = numpy.empty((len(clusters)), dtype=object)
        self.priors = numpy.empty((len(clusters)), dtype=float)
        self.centroids = numpy.empty((len(clusters)), dtype=object)
        self.N = data.shape[0]
        self.assignments = numpy.empty((data.shape[0]), dtype=int)
        self.likelihoods = numpy.empty((self.r+1, 1))
        for c in clusters:
            self.centroids[c] = centroids[c]
            self.clusters[c] = numpy.ma.array(clusters[c], dtype=float)
            self.priors[c] = len(clusters[c])/self.N
            self.covs[c] = numpy.dot(numpy.array(
                clusters[c]-self.centroids[c]).T, numpy.array(clusters[c]-self.centroids[c]))
        # end
    # end

    '''Multivariate normal pdf.@d - (n_rows,n_features),@m - ([n_features]) mean of the cluster,@c - (n_cluster[(n_features,n_features)]) covariance of a cluster'''

    def pdf(self, d, m, c):
        p1 = 1/(((2*numpy.pi)**(len(m)/2))*(lg.det(c))**0.5)
        p2 = -0.5*((d-m).T.dot(lg.inv(c))).dot((d-m))
        return float(p1*numpy.exp(p2))

    '''Calculate loglikelihood of the Gaussian.@data - (n_rows,n_features),@p-(n_rows,n_clusters)-responsibilities of data point wrt a cluster'''

    def logLikeliHood(self, data, p):
        cluster = numpy.zeros((data.shape[0], 1))
        for i in range(0, data.shape[0]):
            # cluster[i] = j means that example i belongs in cluster j
            cluster[i, 0] = numpy.argmax(p[i, :])
            cluster = cluster.astype(int)
        # end
        result = 0
        for x in range(data.shape[0]):
            result = result + numpy.log(p[x, cluster[x, 0]])
        # end
        return result

    '''Maximization Step of EM called to maximize probabilities/responsibilities of each point belonging to a Gaussian
    @data - (n_rows,n_features),@responsibilities - (n_rows,n_clusters),@clusters(n_clusters[clusterpoints])
    returns - @priors,@means,@covs for each cluster'''

    def maximize(self, data, responsibilities, clusters):
        priors = self.priors
        nK = numpy.sum(responsibilities, axis=0)
        self.priors = priors/nK
        for c in range(clusters.shape[0]):
            resp = responsibilities[:, c]
            mean = numpy.sum((data*resp[:, numpy.newaxis]), axis=0)/nK[c]
            self.centroids[c] = mean
            diff = data - self.centroids[c]
            cov = numpy.dot((resp[:, numpy.newaxis]*diff).T, diff)/nK[c]
            self.covs[c] = cov
            # end
        # end
        return [self.priors, self.centroids, self.covs]

    '''Expectation Step of EM called to calculate the probabilities/responsibilities of each point belonging to a Gaussian
    @data - (n_rows,n_features),@clusters(n_clusters[clusterpoints]),@priors(n_clusters),@centroids(n_clusters),@covs(n_clusters(n_clusters,n_features))
    returns - responsibilities(n_rows,n_clusters)'''

    def expectation(self, data, clusters, priors, centroids, covs):
        for x in range(data.shape[0]):
            for c in range(clusters.shape[0]):
                self.responsibilities[x, c] = priors[c] * \
                    self.pdf(data[x, :], centroids[c], covs[c])
                # end
                self.assignments[x] = numpy.argmax(self.responsibilities[x])
            # end
        norm = numpy.sum(self.responsibilities, axis=1)
        self.responsibilities = self.responsibilities / \
            norm.reshape(norm.shape[0], 1)
        return self.responsibilities

    '''Fit the Gaussian by applying Gaussian Mixture EM algorithm'''

    def fit(self):
        clusters = self.clusters
        responsibilities = self.responsibilities
        centroids = self.centroids
        covs = self.covs
        priors = self.priors
        data = self.data
        assignments = self.assignments
        oldLoglikeliHood = -100000000000
        converged = False
        for r in range(self.r):
            print("step - ", r)
            responsibilities = self.expectation(
                data, clusters, priors, centroids, covs)
            [priors, centroids, covs] = self.maximize(
                data, responsibilities, clusters)
            loglikelihood = self.logLikeliHood(data, responsibilities)
            print("loglikelihood - ", loglikelihood)
            if abs(loglikelihood - oldLoglikeliHood) < self.t:
                converged = True
            # end
            oldLoglikeliHood = loglikelihood
            self.likelihoods[r, 0] = loglikelihood
            if converged:
                break
            # end
        # end
    # end

    '''Plot Ellipsoids for each of the Gaussian ditribution obtained after Gaussian Mixture EM algorithm
    @param data - (n_rows,n_features)'''

    def plotGaussian(self, data, **kwargs):
        clusters = self.clusters
        colors = iter(cm.gist_rainbow(numpy.linspace(0, 1, len(clusters))))
        centroids = self.centroids
        assignments = self.assignments
        covs = self.covs
        plt.subplot(121)
        ax = plt.gca()
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Gaussian Mixture Model With K = '+str(clusters.shape[0]))
        for c in range(clusters.shape[0]):
            color = next(colors)
            eigvals, eigvecs = lg.eigh(covs[c])
            eigvals = 3. * numpy.sqrt(2.) * numpy.sqrt(eigvals)
            u = eigvecs[0]/lg.norm(eigvecs[0])
            angle = numpy.arctan(u[1]/u[0])
            angle = 180. * angle/numpy.pi
            ellipse = Ellipse(xy=centroids[c], width=eigvals[0],
                              height=eigvals[1], angle=180.+angle, color=color, linewidth=0.5, alpha=0.5, **kwargs)
            ax.add_artist(ellipse)
            plt.scatter(data[assignments == c, 0],
                        data[assignments == c, 1], color=color, s=10, marker="o")
        # end
        for c in range(centroids.shape[0]):
            plt.scatter(centroids[c][0], centroids[c]
                        [1], color="k", s=50, marker="*")
        plt.subplot(122)
        plt.xlabel('Iterations')
        plt.ylabel('Negative Log likelihood')
        plt.title("Loglikeihood vs Number Of Iterations")
        plt.plot(self.likelihoods[1:])
        plt.show()


'''Compute true mean and covariance for checking with mean and covariance obtained after GMM'''


def computeTrueValues(data):
    cov = []
    mean = []
    cluster1 = data[0:500, :]
    cluster2 = data[500:1000, :]
    cluster3 = data[1000:1500, :]
    mean1 = numpy.mean(cluster1, axis=0)
    mean2 = numpy.mean(cluster2, axis=0)
    mean3 = numpy.mean(cluster3, axis=0)
    mean.extend((mean1, mean2, mean3))
    cov.extend((numpy.dot((cluster1-mean1).T, cluster1-mean1)/cluster1.shape[0],
                numpy.dot((cluster2-mean2).T, cluster2-mean2)/cluster2.shape[0], numpy.dot((cluster3-mean3).T, cluster3-mean3)/cluster3.shape[0]))
    return mean, cov


'''starting point'''


def main():
    data = numpy.loadtxt('GMM_dataset.txt')
    km = kMeans.kMeans(k=5, r=30, t=1e-03)
    km.clusterData(data)
    trueMean, trueCov = computeTrueValues(data)
    log({"true mean": trueMean})
    log({"true cov": trueCov})
    gmm = gMM(data, r=50, centroids=km.centroids,
              clusters=km.clusters, t=1e-03)
    gmm.fit()
    print("Gaussian Mixture Centroids - ", gmm.centroids)
    print("Gaussian Mixture Covariance - ", gmm.covs)
    gmm.plotGaussian(data)


if __name__ == "__main__":
    main()
