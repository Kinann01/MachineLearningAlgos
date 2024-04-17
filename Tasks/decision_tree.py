#!/usr/bin/env python3
import argparse
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from queue import PriorityQueue

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.

# Limits 
# 1) maximum depth of the tree - depth < args.max_depth
# 2) maximum number of leaf nodes - number of leaf nodes < args.max_leaves 
# 3) minimum examples required to split - len(self.instances) >= args.min_to_split
# 4) criterion value is not zero - criterion != 0


###
### 
class Node:
    def __init__(self, instances, prediction):
        self.isLeaf = True # is leaf node
        self.instances = instances # indices of instances
        self.prediction = prediction # most frequent class
        self.feature = None # feature index to split on (0,1,2,...)
        self.value = None # feature value to split on (average of two nearest unique feature values)
        self.left = None # left child
        self.right = None  # right child
        self.criterion = None # criterion value

    def __repr__(self) -> str:
        return "Node({}, {}, {}, {}, {}, {})".format(self.isLeaf, self.instances, self.prediction, self.feature, self.value, self.criterion)

    def split(self, feature, value, left, right, criterion):
        self.isLeaf = False 
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.criterion = criterion

    def canSplit(self, min_to_split, depth, maxDepth):
        return (len(self.instances) >= min_to_split) and (self.criterion != 0) and (maxDepth is None or depth < maxDepth)


def allUniqClasses(classes):
    uniqueCount = dict()
    for i in range(len(classes)):
        if classes[i] not in uniqueCount:
            uniqueCount[classes[i]] = 1
        else:
            uniqueCount[classes[i]] += 1
    return uniqueCount

def gini(classes):
    allClasses = len(classes)
    uniqueCount = allUniqClasses(classes)
    giniIndex = 0
    for c in np.unique(classes):
        x = uniqueCount[c] / allClasses
        giniIndex += x * (1 - x)
    return allClasses * giniIndex

def entropy(classes):
    allClasses = len(classes)
    uniqueCount = allUniqClasses(classes)
    entropy = 0
    for c in np.unique(classes):
        x = uniqueCount[c] / allClasses
        if x != 0:
            entropy += x * np.log(x)
    return -allClasses * entropy 

class DecisionTree:

    def __init__(self, c, depth, ms, ml, root, d, t):
        self.criterion = c
        self.maxDepth = depth
        self.minToSplit = ms
        self.maxLeaves = ml
        self.treeRoot = root
        self.data = d
        self.targets = t

    def splitWithQueue(self, node):
        pq = PriorityQueue()
        p, _, _, _, _ = self.bestSplit(node.instances)
        pq.put((p, node))
        count = 1
        while not pq.empty() and count < self.maxLeaves:
            _, node = pq.get()
            if node.canSplit(self.minToSplit, 0, self.maxDepth):
                bestCriterion, bestFeature, bestValue, bestLeftInstances, bestRightInstances = self.bestSplit(node.instances)
                leftNode = Node(bestLeftInstances, np.argmax(np.bincount(self.targets[bestLeftInstances])))
                rightNode = Node(bestRightInstances, np.argmax(np.bincount(self.targets[bestRightInstances])))
                node.split(bestFeature, bestValue, leftNode, rightNode, bestCriterion)
                bestLeftCrit, _, _, _, _ = self.bestSplit(leftNode.instances)
                bestRightCrit, _, _, _, _ = self.bestSplit(rightNode.instances)
                pq.put((bestLeftCrit, leftNode))
                pq.put((bestRightCrit, rightNode))
                count += 1 

    def splitWithRecursion(self, node, depth):
        if not node.canSplit(self.minToSplit, depth, self.maxDepth):
            return
        bestCriterion, bestFeature, bestValue, bestLeftInstances, bestRightInstances = self.bestSplit(node.instances)
        leftNode = Node(bestLeftInstances, np.argmax(np.bincount(self.targets[bestLeftInstances])))
        rightNode = Node(bestRightInstances, np.argmax(np.bincount(self.targets[bestRightInstances])))
        # print(len(node.instances), len(bestLeftInstances), len(bestRightInstances), entropy(self.targets[node.instances]))
        node.split(bestFeature, bestValue, leftNode, rightNode, bestCriterion)
        self.splitWithRecursion(leftNode, depth + 1), self.splitWithRecursion(rightNode, depth + 1)
            
    def go(self):
        if self.maxLeaves is None:
            self.splitWithRecursion(self.treeRoot, 0)
        else:
            self.splitWithQueue(self.treeRoot)
    
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
    
    def bestSplit(self, instances):

        taoNode = None
        if self.criterion == 'gini':
            taoNode = gini(self.targets[instances])
        else:
            taoNode = entropy(self.targets[instances])

        bestCriterion = int(1e9)
        for featureIdx in range(len(self.data[0])):
            uniqueSortedValues = np.sort(np.unique(self.data[instances, featureIdx]))
            for i in range(len(uniqueSortedValues) - 1):
                # print(uniqueSortedValues[i], uniqueSortedValues[i+1])
                value = (uniqueSortedValues[i] + uniqueSortedValues[i+1]) / 2
                leftInstances = []
                rightInstances = []
                for instance in instances:
                    if self.data[instance][featureIdx] <= value:
                        leftInstances.append(instance)
                    else:
                        rightInstances.append(instance)
                leftTargets = self.targets[np.array(leftInstances)]
                rightTargets = self.targets[np.array(rightInstances)]
                if self.criterion == 'gini':
                    currentCriterion = gini(leftTargets) + gini(rightTargets) - taoNode
                else: 
                    currentCriterion = entropy(leftTargets) + entropy(rightTargets) - taoNode

                if (currentCriterion < bestCriterion):
                    bestCriterion = currentCriterion
                    bestFeature = featureIdx
                    bestValue = value
                    bestLeftInstances = leftInstances
                    bestRightInstances = rightInstances
        # print(bestValue)
        # print(bestLeftInstances, bestRightInstances)
        return bestCriterion, bestFeature, bestValue, bestLeftInstances, bestRightInstances

def main(args: argparse.Namespace) -> tuple[float, float]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    traindata, testdata, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)
    # TODO: Manually create a decision tree on the training data.
    #
    # - For each node, predict the most frequent class (and the one with
    #   the smallest number if there are several such classes).
    #
    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split decreasing the criterion
    #   the most. Each split point is an average of two nearest unique feature values
    #   of the instances corresponding to the given node (e.g., for four instances
    #   with values 1, 7, 3, 3, the split points are 2 and 5). 
    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not `None`, its depth must be less than `args.max_depth`
    #     (depth of the root node is zero);
    #   - when `args.max_leaves` is not `None`, there are less than `args.max_leaves` leaves
    #     (a leaf is a tree node without children);
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    #
    # - When `args.max_leaves` is `None`, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not `None`), repeatably split a leaf where the
    #   constraints are valid and the overall criterion value ($c_left + c_right - c_node$)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).
    
    # TODO: Finally, measure the training and testing accuracy.
    root = Node(np.arange(len(traindata)), np.argmax(np.bincount(train_target)))
    # print(np.argmax(np.bincount(train_target)))
    tree = DecisionTree(args.criterion, args.max_depth, args.min_to_split, args.max_leaves, root, traindata, train_target)
    tree.go()
    
    predictTrain = tree.predictInput(traindata)
    predictTest = tree.predictInput(testdata)
    train_accuracy = sklearn.metrics.accuracy_score(train_target, predictTrain)
    test_accuracy = sklearn.metrics.accuracy_score(test_target, predictTest)

    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))