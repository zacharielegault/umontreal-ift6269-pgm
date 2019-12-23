import argparse
from typing import Iterable, Union

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


def plot_clustering(
        X: Union[np.ndarray, Iterable],
        labels: Union[np.ndarray, Iterable],
        legend=False,
        s=10,
        alpha=0.5,
        save_path=None,
):
    if isinstance(X, np.ndarray):
        X = [X]
    elif isinstance(X, dict):
        X = X.values()
    assert all([isinstance(x, np.ndarray) for x in X]), "{}".format([type(x) for x in X])

    if isinstance(labels, np.ndarray):
        labels = [labels]
    elif isinstance(labels, dict):
        labels = labels.values()
    assert all([isinstance(labels_, np.ndarray) for labels_ in labels])

    fig, ax = plt.subplots(nrows=len(X), sharex=True, sharey=True)
    ax = [ax] if len(X) == 1 else ax

    for i, (x, labels_) in enumerate(zip(X, labels)):
        print(i)
        for label in np.unique(labels_):
            ax[i].scatter(x[labels_ == label][:, 0], x[labels_ == label][:, 1], s=s, alpha=alpha)
            ax[i].set_aspect('equal')

        if legend:
            box = ax[i].get_position()
            ax[i].legend(np.unique(labels), loc='center left', bbox_to_anchor=(1, 0.5))
            ax[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def visualize(z, y, legend=False):
    if z.shape[1] == 2:
        plot_clustering(z, y)
    elif args.pca:
        pca = PCA(2)
        plot_clustering(pca.fit_transform(z), y, legend=legend)
    else:
        plot_clustering(TSNE(n_components=2).fit_transform(z), y, legend=legend)


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
        #"dbscan": ClusteringMethod(
        #    DBSCAN(
        #        eps=0.5,
        #        min_samples=10,
        #        metric='canberra',
        #        n_jobs=-1))
    }

    path = "./data"
    z = np.load("validation_latent.npy")
    y = np.load(path + "/cortex_y_test.npy").astype(np.float32)
    names = ['astrocytes ependymal', 'endothelial mural', 'interneurons', 'microglia', 'oligodendrocytes',
             'pyramidal CA1', 'pyramidal SS']
    labels = y.astype(np.str)
    for i in range(7):
        labels[y == i] = names[i]

    visualize(z, labels, legend=True)

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
