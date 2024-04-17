#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--l2", default=1., type=float, help="L2 regularization factor")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.


class Node:
    def __init__(self, instances, depth, node_gs, node_hs, l2):
        self.isLeaf = True # is leaf node
        self.depth = depth
        self.instances = instances # indices of instances
        # self.prediction = prediction # most frequent class
        self.feature = None # feature index to split on (0,1,2,...)
        self.value = None # feature value to split on (average of two nearest unique feature values)
        self.left = None # left child
        self.right = None  # right child
        self.prediction = self._calc_prediction(node_gs, node_hs, l2)
        self.min_to_split = 2

    def __repr__(self) -> str:
        return "Node({}, {}, {}, {}, {})".format(self.isLeaf, self.instances, self.prediction, self.feature, self.value)

    def split(self, feature, value, left_node, right_node):
        self.isLeaf = False 
        self.feature = feature
        self.value = value
        self.left = left_node
        self.right = right_node

    def canSplit(self, min_to_split, maxDepth):
        return (len(self.instances) >= min_to_split) and \
               (maxDepth is None or self.depth < maxDepth)
    
    def _calc_prediction(self, node_gs, node_hs, l2):
        return -np.sum(node_gs) / (np.sum(node_hs) + l2)
    
    def is_leaf(self):
        return self.isLeaf
    
    

class DecisionTree:
    def __init__(self, min_to_split, maxDepth, l2):
        self.min_to_split = min_to_split
        self.maxDepth = maxDepth
        self.l2 = l2
        self.root = None
        self.data = None
        self.gs = None
        self.hs = None

    def train(self, data, gs, hs):
        self.data = data
        self.gs = gs
        self.hs = hs
        data_indices = np.arange(len(data))
        depth = 0
        self.root = Node(data_indices, depth, gs[data_indices], hs[data_indices], self.l2)
        self._split_recursive(self.root)

    def _calc_criterion(self, instance_indices):
        gs = self.gs[instance_indices]
        hs = self.hs[instance_indices]
        return -0.5 * np.sum(gs)**2 / (np.sum(hs) + self.l2)

    def _best_split(self, node):
        # G = np.sum(self.gs[node.instances])
        # H = np.sum(self.hs[node.instances])
        best_criterion = np.inf
        best_feature = None
        best_value = None
        best_left_idxs = None
        best_right_idxs = None

        for feature_idx in range(self.data.shape[1]):
            # sorted_idxs = np.argsort(self.data(node.instances, feature_idx))
            sorted_idxs = np.argsort(self.data[node.instances, feature_idx])
            sorted_idxs = node.instances[sorted_idxs]

            for j in range(1, len(sorted_idxs)):
                prev_idx = sorted_idxs[j - 1]
                curr_idx = sorted_idxs[j]
                if self.data[prev_idx, feature_idx] == self.data[curr_idx, feature_idx]:
                    continue
                split_value = (self.data[prev_idx, feature_idx] + self.data[curr_idx, feature_idx]) / 2
                left_idxs = sorted_idxs[:j]
                right_idxs = sorted_idxs[j:]
                criterion = self._calc_criterion(left_idxs) + self._calc_criterion(right_idxs)
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_feature = feature_idx
                    best_value = split_value
                    best_left_idxs = left_idxs
                    best_right_idxs = right_idxs
        return best_feature, best_value, best_left_idxs, best_right_idxs

    def _split_recursive(self, node):
        if not node.canSplit(self.min_to_split, self.maxDepth):
            return
        split_feature, split_value, left_data_idxs, right_data_idxs = \
            self._best_split(node)
        left_node = Node(left_data_idxs, node.depth + 1, self.gs[left_data_idxs], self.hs[left_data_idxs], self.l2)
        right_node = Node(right_data_idxs, node.depth + 1, self.gs[right_data_idxs], self.hs[right_data_idxs], self.l2)
        node.split(split_feature, split_value, left_node, right_node)
        self._split_recursive(left_node)
        self._split_recursive(right_node)
    
    def predict(self, data):
        predictions = np.zeros(len(data), dtype=np.float32)
        for i in range(len(data)):
            node = self.root
            while not node.is_leaf():
                if data[i, node.feature] <= node.value:
                    node = node.left
                else:
                    node = node.right
            predictions[i] = node.prediction
        return predictions

class MCGradientBoosting:
    def __init__(self, trees, classes, learning_rate, max_depth, l2):
        self.num_trees = trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.l2 = l2
        self.trees = []
        self._classes = classes
        self.min_to_split = 2

    def _softmax(self, z):
        z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return z / np.sum(z, axis=-1, keepdims=True)

    def fit(self, data, target):
        prev_lvl_predicts = np.zeros((len(data), self._classes))
        for i in range(self.num_trees):
            normalized = self._softmax(prev_lvl_predicts)
            self.trees.append([])   # new level of trees
            for c in range(self._classes):
                gs = normalized[:, c] - (target == c)
                hs = normalized[:, c] * (1 - normalized[:, c])  
                new_tree = DecisionTree(self.min_to_split, self.max_depth, self.l2)
                new_tree.train(data, gs, hs)
                self.trees[-1].append(new_tree)
                prev_lvl_predicts[:, c] += self.learning_rate * new_tree.predict(data)

    def predict(self, data, num_tree_levels):
        predictions = np.zeros((len(data), self._classes))
        for i in range(num_tree_levels):
            for j in range(len(self.trees[i])):
                predictions[:, j] += self.trees[i][j].predict(data)
        return np.argmax(predictions, axis=1)
            

def main(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    num_classes = np.max(target) + 1

    # TODO: Create gradient boosted trees on the classification training data.
    #
    # Notably, train for `args.trees` iteration. During the iteration `t`:
    # - the goal is to train `classes` regression trees, each predicting
    #   a part of logit for the corresponding class.
    # - compute the current predictions `y_{t-1}(x_i)` for every training example `i` as
    #     y_{t-1}(x_i)_c = \sum_{j=1}^{t-1} args.learning_rate * tree_{iter=j,class=c}.predict(x_i)
    #   (note that y_0 is zero)
    # - loss in iteration `t` is
    #     E = (\sum_i NLL(onehot_target_i, softmax(y_{t-1}(x_i) + trees_to_train_in_iter_t.predict(x_i)))) +
    #         1/2 * args.l2 * (sum of all node values in trees_to_train_in_iter_t)
    # - for every class `c`:
    #   - start by computing `g_i` and `h_i` for every training example `i`;
    #     the `g_i` is the first and the `h_i` is the second derivative of
    #     NLL(onehot_target_i_c, softmax(y_{t-1}(x_i))_c) with respect to y_{t-1}(x_i)_c.
    #   - then, create a decision tree minimizing the loss E. According to the slides,
    #     the optimum prediction for a given node T with training examples I_T is
    #       w_T = - (\sum_{i \in I_T} g_i) / (args.l2 + sum_{i \in I_T} h_i)
    #     and the value of the loss with this prediction is
    #       c_GB = - 1/2 (\sum_{i \in I_T} g_i)^2 / (args.l2 + sum_{i \in I_T} h_i)
    #     which you should use as a splitting criterion.
    #
    # During tree construction, we split a node if:
    # - its depth is less than `args.max_depth`
    # - there is more than 1 example corresponding to it (this was covered by
    #     a non-zero criterion value in the previous assignments)

    model = MCGradientBoosting(args.trees, num_classes, args.learning_rate, args.max_depth, args.l2)
    model.fit(train_data, train_target)

    # TODO: Finally, measure your training and testing accuracies when
    # using 1, 2, ..., `args.trees` of the created trees.
    #
    # To perform a prediction using t trees, compute the y_t(x_i) and return the
    # class with the highest value (pick the smallest class number if there is a tie).
    train_accuracies = []
    test_accuracies = []
    for num_trees in range(1, args.trees + 1):
        train_accuracies.append(sklearn.metrics.accuracy_score(train_target, model.predict(train_data, num_trees)))
        test_accuracies.append(sklearn.metrics.accuracy_score(test_target, model.predict(test_data, num_trees)))

    return [100 * acc for acc in train_accuracies], [100 * acc for acc in test_accuracies]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracies, test_accuracies = main(args)

    for i, (train_accuracy, test_accuracy) in enumerate(zip(train_accuracies, test_accuracies)):
        print("Using {} trees, train accuracy: {:.1f}%, test accuracy: {:.1f}%".format(
            i + 1, train_accuracy, test_accuracy))