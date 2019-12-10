import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score,
                             normalized_mutual_info_score as NMI,
                             adjusted_rand_score as ARI)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        "-k",
        required=True,
        type=int,
        help="Number of clusters for K-Means and GMM")

    # Optional arguments
    parser.add_argument(
        "-v", "--visualize",
        default=False,
        action='store_true',
        help="Optional. Boolean. Visualize the prediction."
    )

    parser.add_argument(
        "--pca",
        default=False,
        action='store_true',
        help="Optional. Boolean. Use PCA for embedding higher dimensional data in 2D. Will use t-SNE as default."
             "Has no effect if data is already 2D."
    )

    return parser.parse_args()


def plot_clustering(X, labels, legend=False):
    for label in np.unique(labels):
        plt.scatter(X[labels == label][:, 0], X[labels == label][:, 1], s=10, alpha=0.5)

    if legend:
        plt.legend(np.unique(labels))

    plt.show()


def visualize(z, y):
    if z.shape[1] == 2:
        plot_clustering(z, y)
    elif args.pca:
        pca = PCA(2)
        plot_clustering(pca.fit_transform(z), y)
    else:
        plot_clustering(TSNE(n_components=2).fit_transform(z), y)


def accuracy(true_labels, pred_labels):
    """A simple accuracy metric. Makes a confusion matrix and takes then
    maximum column-wise value as being the "correct" prediction (assumes
    that the clusters are at least close to right).
    """
    k = len(np.unique(true_labels))

    assert len(np.unique(true_labels)) == len(np.unique(pred_labels)), \
        "There are {} true labels, but {} predicted labels".format(k, len(np.unique(pred_labels)))

    matrix = np.zeros((k, k))
    for (test, pred) in zip(true_labels, pred_labels):
        matrix[int(test), int(pred)] += 1

    maxes = np.amax(matrix, axis=1)
    return np.sum(maxes) / np.sum(matrix)


class ClusteringMethod:
    def __init__(self, clustering):
        self.clustering = clustering
        self.labels = None
        self.k = None
        self.silhouette_score = None
        self.accuracy = None
        self.ari = None
        self.nmi = None

    def fit(self, z):
        self.labels = self.clustering.fit_predict(z)
        self.k = len(np.unique(self.labels))

    def score(self, z, true_labels):
        assert self.labels is not None, "Cannot compute clustering scores before fitting the data."

        self.silhouette_score = silhouette_score(z, self.labels)
        self.nmi = NMI(true_labels, self.labels, average_method="geometric")  # Same average as original paper
        self.ari = ARI(true_labels, self.labels)

        true_k = len(np.unique(true_labels))
        if self.k == true_k:
            self.accuracy = accuracy(true_labels, self.labels)
        else:
            print("Fitted number of labels ({}) is not equal to given true number of labels ({}). Cannot "
                  "compute accuracy.".format(self.k, true_k))


if __name__ == "__main__":
    args = parse_args()

    methods = {
        "kmeans": ClusteringMethod(
            KMeans(
                n_clusters=args.k,
                init='k-means++',
                n_init=200,
                max_iter=300)),
        "gmm": ClusteringMethod(
            GaussianMixture(
                n_components=args.k,
                covariance_type='full',
                max_iter=100,
                n_init=10,
                init_params='kmeans')),
        "dbscan": ClusteringMethod(
            DBSCAN(
                eps=0.5,
                min_samples=10,
                metric='canberra',
                n_jobs=-1))
    }

    # TODO: change data import when single-cell data is available
    z = np.loadtxt("hwk3data/EMGaussian.train")
    y = np.loadtxt("hwk3data/HMMlabels.train")
    visualize(z, y)

    for name, method in methods.items():
        print(name)
        method.fit(z)
        method.score(z, y)

        print("\t Sihouette score =", method.silhouette_score)
        print("\t NMI =", method.nmi)
        print("\t ARI =", method.ari)

        if np.unique(method.labels)[0] == -1 and method.k == args.k + 1:
            method.accuracy = accuracy(method.labels[method.labels != -1], y[method.labels != -1])
            print("\t Accuracy =", method.accuracy, "(with", len(method.labels[method.labels == -1]),
                  "outliers out of", len(method.labels), "points ignored)")

            # Adjust accuracy for case where the number of clusters is correct, but some points are outliers
            # TODO: make sure this correction is OK
            method.accuracy -= np.sum(method.labels == -1) / len(method.labels)
            print("\t Accuracy =", method.accuracy, "(corrected)")
        elif method.accuracy is not None:
            print("\t Accuracy =", method.accuracy)

        if args.visualize:
            visualize(z, method.labels)
