#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=5, type=int, help="Number of classes to use")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.

def classify(train, test):

    classified_train = [[1 if x > 0.5 else 0 for x in sample] for sample in train]
    classified_test  = [[1 if x > 0.5 else 0 for x in sample] for sample in test]

    return (classified_train, classified_test)


def getCounts(classified, target, classes):

    tp, tn, fp, fn = np.zeros(classes), np.zeros(classes), np.zeros(classes), np.zeros(classes)

    for c, t in zip(classified, target):
        for i in range(classes):

            if (c[i] == 1 and t[i] == 1): # True positive
                tp[i] += 1
            
            elif (c[i] == 1 and t[i] == 0): # False Positive
                fp[i] += 1

            elif (c[i] == 0 and t[i] == 1): # False Negative
                fn[i] += 1
            
            elif (c[i] == 0 and c[i] == 0): # true negative
                tn[i] += 1

    return (tp, tn, fp, fn)

def getScores(classified, target, classes):

    tp, _, fp, fn = getCounts(classified, target, classes)
    
    precision = [i / (i + j) if i + j > 0 else 0 for i, j in zip(tp, fp)]
    recall = [i / (i + j) if i + j > 0 else 0 for i, j in zip(tp, fn)]
    F1Score = [2 * p * r / (p + r) if p + r > 0 else 0 for p, r in zip(precision, recall)]
    
    microPrecision = sum(tp) / (sum(tp) + sum(fp)) if sum(tp) + sum(fp) > 0 else 0
    microRecall = sum(tp) / (sum(tp) + sum(fn)) if sum(tp) + sum(fn) > 0 else 0

    microF1Score = 2 * microPrecision * microRecall / (microPrecision + microRecall) if microPrecision + microRecall > 0 else 0    
    macroF1Score = sum(F1Score) / len(F1Score)
    
    return microF1Score, macroF1Score


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial classification dataset.
    data, target_list = sklearn.datasets.make_multilabel_classification(
        n_samples=args.data_size, n_classes=args.classes, allow_unlabeled=False,
        return_indicator=False, random_state=args.seed)

    # TODO: The `target` is a list of classes for every input example. Convert
    # it to a dense representation (n-hot encoding) -- for each input example,
    # the target should be vector of `args.classes` binary indicators.
    #     
    target = np.zeros((len(target_list), args.classes))
    for i, class_ in enumerate(target_list):
        # print(class_)
        target[i, class_] = 1

    #print(target)
    # Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    for epoch in range(args.epochs):

        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.

        data_shuffled = train_data[permutation]
        target_shuffled = train_target[permutation]

        for batch_start in range(0, len(train_data), args.batch_size):

            oneDataBatch = data_shuffled[batch_start : batch_start + args.batch_size]
            oneTargetBatch = target_shuffled[batch_start : batch_start + args.batch_size]

            # Initialize the gradient as zero
            gradient = np.zeros_like(weights)
            prediction = sigmoid(np.dot(oneDataBatch, weights)) 
            errors = prediction - oneTargetBatch
            batchGradient = np.dot(oneDataBatch.T, errors)
            gradient = (1 / args.batch_size) * batchGradient
            
            # # update gradient and our weights for each batch
            weights = weights - args.learning_rate * gradient

        # TODO: After the SGD epoch, compute the micro-averaged and the
        # macro-averaged F1-score for both the train test and the test set.
        # Compute these scores manually, without using `sklearn.metrics`.

        trainDataPrediction = sigmoid(np.dot(train_data, weights))
        testDataPrediction = sigmoid(np.dot(test_data, weights))

        (classifiedTrainPrediction, classifiedTestPrediction) = classify(trainDataPrediction, testDataPrediction)
    
        train_f1_micro, train_f1_macro = getScores(classifiedTrainPrediction, train_target, args.classes)
        test_f1_micro, test_f1_macro = getScores(classifiedTestPrediction, test_target, args.classes)

        print("After epoch {}: train F1 micro {:.2f}% macro {:.2f}%, test F1 micro {:.2f}% macro {:.1f}%".format(
            epoch + 1, 100 * train_f1_micro, 100 * train_f1_macro, 100 * test_f1_micro, 100 * test_f1_macro))

    return weights, [(100 * train_f1_micro, 100 * train_f1_macro), (100 * test_f1_micro, 100 * test_f1_macro)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")