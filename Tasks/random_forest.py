#!/usr/bin/env python3
import argparse

import numpy as np
from sklearn.metrics import accuracy_score
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bagging", default=False, action="store_true", help="Perform bagging")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--feature_subsampling", default=1.0, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=44, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.

def entropy(classes):
    bins = np.bincount(classes)
    bins = bins[np.nonzero(bins)]
    return -np.sum(bins * np.log(bins / len(classes)))

class Node:
    def __init__(self, instances, prediction):
        self.isLeaf = True
        self.instances = instances
        self.prediction = prediction
        self.feature = None
        self.value = None
        self.left = None
        self.right = None

    def split(self, feature, value, left, right):
        self.isLeaf = False
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right

    def canSplit(self, depth, maxDepth, criterion):
        return (maxDepth is None or depth < maxDepth) and (criterion is None or criterion != 0)

class DecisionTree:
    def __init__(self, featureSubSamplingMethod, dp, r, d, t):
        self.sample = featureSubSamplingMethod
        self.maxDepth = dp
        self.treeRoot = r
        self.data = d
        self.targets = t

    def go(self):
        self.splitWithRecursion(self.treeRoot, 0)

    def predictInput(self, data):
        predictions = np.zeros(len(data))
        for example in range(len(data)):
            node = self.treeRoot
            while not node.isLeaf:
                if data[example][node.feature] <= node.value:
                    node = node.left
                else:
                    node = node.right
            predictions[example] = node.prediction

        return predictions

    def splitWithRecursion(self, node, depth):
        if not node.canSplit(depth, self.maxDepth, entropy(self.targets[node.instances])):
            return
        
        feature, value, left, right, _ = self.bestSplit(node.instances)
        leftNode = Node(left, np.argmax(np.bincount(self.targets[left])))
        rightNode = Node(right, np.argmax(np.bincount(self.targets[right])))
        node.split(feature, value, leftNode, rightNode)
        # print(len(node.instances), len(leftNode.instances), len(rightNode.instances), entropy(self.targets[node.instances]))
        self.splitWithRecursion(leftNode, depth + 1)
        self.splitWithRecursion(rightNode, depth + 1)

    def bestSplit(self, instances):
        bestCriterion = int(1e9)
        features = self.sample(len(self.data[0]))
        for featureIdx in features:
            uniqueSortedValues = np.unique(self.data[instances, featureIdx])
            midpoints = (uniqueSortedValues[:-1] + uniqueSortedValues[1:]) / 2

            for value in midpoints:
                leftMask = self.data[instances, featureIdx] <= value
                rightMask = ~leftMask
                leftInstances = instances[leftMask]
                rightInstances = instances[rightMask]
                if len(leftInstances) > 0 and len(rightInstances) > 0:
                    leftTargets = self.targets[leftInstances]
                    rightTargets = self.targets[rightInstances]
                    currentCriterion = entropy(leftTargets) + entropy(rightTargets)

                    if currentCriterion < bestCriterion:
                        bestCriterion = currentCriterion
                        bestFeature = featureIdx
                        bestValue = value
                        bestLeftInstances = leftInstances
                        bestRightInstances = rightInstances

        return bestFeature, bestValue, bestLeftInstances, bestRightInstances, bestCriterion

class RandomForest:

    def __init__(self, trees, bagging, b, fs, d):
        self.numberOfTrees = trees
        self.baggingMethod = bagging
        self.bagging = b
        self.featureSubSamplingMethod = fs
        self.maxDepth = d
        self.allTrees = np.zeros(self.numberOfTrees, dtype=DecisionTree)

    def createTree(self, train_data, train_target):

        root = Node(np.arange(len(train_data)), np.argmax(np.bincount(train_target)))
        tree = DecisionTree(self.featureSubSamplingMethod, self.maxDepth, root, train_data, train_target)
        tree.go()
        return tree

    def trainForest(self, train_data, train_target):
        trees = self.createForest(train_data, train_target)
        return trees

    def createForest(self, train_data, train_target):
        trees = []
        for i in range(self.numberOfTrees):
            if self.bagging:
                indices = self.baggingMethod(train_data)
                tree = self.createTree(train_data[indices], train_target[indices])
            else:
                tree = self.createTree(train_data, train_target)
            trees.append(tree)
            self.allTrees[i] = tree

        return trees

    def predictForest(self, data):
        treePredictions = []
        for tree in self.allTrees:
            treePredictions.append(tree.predictInput(data))

        treePredictions = np.array(treePredictions)
        predictions = np.zeros(len(data))
        for i in range(len(data)):
            predictions[i] = np.argmax(np.bincount(treePredictions[:, i].astype(int))  )

        return predictions
            
def main(args: argparse.Namespace) -> tuple[float, float]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Create random generators.
    generator_feature_subsampling = np.random.RandomState(args.seed)
    def subsample_features(number_of_features: int) -> np.ndarray:
        return np.sort(generator_feature_subsampling.choice(
            number_of_features, size=int(args.feature_subsampling * number_of_features), replace=False))

    generator_bootstrapping = np.random.RandomState(args.seed)
    def bootstrap_dataset(train_data: np.ndarray) -> np.ndarray:
        return generator_bootstrapping.choice(len(train_data), size=len(train_data), replace=True)

    # TODO: Create a random forest on the training data.
    #
    # Use a simplified decision tree from the `decision_tree` assignment:
    # - use `entropy` as the criterion
    # - use `max_depth` constraint, to split a node only if:
    #   - its depth is less than `args.max_depth`
    #   - the criterion is not 0 (the corresponding instance targets are not the same)
    # When splitting nodes, proceed in the depth-first order, splitting all nodes
    # in the left subtree before the nodes in right subtree.


    # TODO: Finally, measure the training and testing accuracy.
    # Additionally, implement:
    # - feature subsampling: when searching for the best split, try only
    #   a subset of features. Notably, when splitting a node (i.e., when the
    #   splitting conditions [depth, criterion != 0] are satisfied), start by
    #   generating the subsampled features using
    #     subsample_features(number_of_features)
    #   returning the features that should be used during the best split search.
    #   The features are returned in ascending order, so when `feature_subsampling == 1`,
    #   the `np.arange(number_of_features)` is returned.
    #
    # - train a random forest consisting of `args.trees` decision trees
    #
    # - if `args.bagging` is set, before training each decision tree
    #   create a bootstrap sample of the training data by calling
    #     dataset_indices = bootstrap_dataset(train_data)
    #   and if `args.bagging` is not set, use the original training data.
    #
    # During prediction, use voting to find the most frequent class for a given
    # input, choosing the one with the smallest class number in case of a tie.
    
    numberOfTreesToTrain = args.trees
    maxDepth = args.max_depth
    bagging = args.bagging
    forest = RandomForest(numberOfTreesToTrain, bootstrap_dataset, bagging, subsample_features, maxDepth)
    forest.trainForest(train_data, train_target)
    trainPrediction = forest.predictForest(train_data)
    testPrediction = forest.predictForest(test_data)

    # TODO: Finally, measure the training and testing accuracy.

    test_accuracy = accuracy_score(test_target, testPrediction)
    train_accuracy = accuracy_score(train_target, trainPrediction)

    return 100 * train_accuracy, 100 * test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))