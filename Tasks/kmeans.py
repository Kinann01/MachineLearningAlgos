#!/usr/bin/env python3
import argparse

import numpy as np

import sklearn.datasets

from enum import Enum

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--clusters", default=3, type=int, help="Number of clusters")
parser.add_argument("--examples", default=200, type=int, help="Number of examples")
parser.add_argument("--init", default="random", choices=["random", "kmeans++"], help="Initialization")
parser.add_argument("--iterations", default=20, type=int, help="Number of kmeans iterations to perfom")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.


def plot(args: argparse.Namespace, iteration: int,
         data: np.ndarray, centers: np.ndarray, clusters: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    if args.plot is not True:
        plt.gcf().get_axes() or plt.figure(figsize=(4*2, 5*6))
        plt.subplot(6, 2, 1 + len(plt.gcf().get_axes()))
    plt.title("KMeans Initialization" if not iteration else
              "KMeans After Iteration {}".format(iteration))
    plt.gca().set_aspect(1)
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=200, c="#ff0000")
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=50, c=range(args.clusters))
    plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

class KMeans:
    def __init__(self, data, generator):
        self._data = data
        self._generator = generator
        self.centers = None
        self.clusters = None
    
    def random_init(self, num_clusters):
        self.centers = self._data[self._generator.choice(len(self._data), size=num_clusters, replace=False)]
        
    
    def kmeans_plus_plus_init(self, num_clusters):
        center_idxs = []
        # randomly choose first center
        center_idxs.append(self._generator.randint(len(self._data)))
        # choose other centers
        pot_center_idxs = list(range(len(self._data)))
        for _ in range(num_clusters - 1):
            # Remove the last chosen center index from potential center indexes
            last_center_idx = np.where(pot_center_idxs == center_idxs[-1])[0]
            pot_center_idxs = np.delete(pot_center_idxs, last_center_idx)

            squared_distances = np.zeros((len(pot_center_idxs), len(center_idxs)))

            for j, center_idx in enumerate(center_idxs):
                diff = self._data[pot_center_idxs] - self._data[center_idx]
                squared_distances[:, j] = np.linalg.norm(diff, axis=1) ** 2

            min_squared_distances = np.min(squared_distances, axis=1)
            center_idxs.append(self._generator.choice(pot_center_idxs, p=min_squared_distances / np.sum(min_squared_distances)))

        self.centers = self._data[center_idxs]
    
    def perform_iter(self):
        distances = np.zeros((len(self._data), len(self.centers)))
        for i, center in enumerate(self.centers):
            diff = self._data - center
            distances[:, i] = np.linalg.norm(diff, axis=1)
        self.clusters = np.argmin(distances, axis=1)
        
        for c in range(len(self.centers)):
            self.centers[c] = np.mean(self._data[self.clusters == c], axis=0)
        
            

def main(args: argparse.Namespace) -> np.ndarray:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial dataset.
    data, target = sklearn.datasets.make_blobs(
        n_samples=args.examples, centers=args.clusters, n_features=2, random_state=args.seed)

    # TODO: Initialize `centers` to be
    # - if args.init == "random", K random data points, using the indices
    #   returned by
    #     generator.choice(len(data), size=args.clusters, replace=False)
    # - if args.init == "kmeans++", generate the first cluster by
    #     generator.randint(len(data))
    #   and then iteratively sample the rest of the clusters proportionally to
    #   the square of their distances to their closest cluster using
    #     generator.choice(unused_points_indices, p=square_distances / np.sum(square_distances))
    #   Use the `np.linalg.norm` to measure the distances.
    model = KMeans(data, generator)
    if args.init == "random":
        model.random_init(args.clusters)
    elif args.init == "kmeans++":
        model.kmeans_plus_plus_init(args.clusters)

    # centers = model.centers
    if args.plot:
        plot(args, 0, data, model.centers, clusters=None)

    # Run `args.iterations` of the K-Means algorithm.
    for iteration in range(args.iterations):
        # TODO: Perform a single iteration of the K-Means algorithm, storing
        # zero-based cluster assignment to `clusters`.
        model.perform_iter()

        if args.plot:
            plot(args, 1 + iteration, data, model.centers, model.clusters)

    return model.clusters


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    clusters = main(args)
    print("Cluster assignments:", clusters, sep="\n")
