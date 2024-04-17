#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter of our NB classifier")
parser.add_argument("--naive_bayes_type", default="gaussian", choices=["gaussian", "multinomial", "bernoulli"])
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=72, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Train a naive Bayes classifier on the train data.
    #
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    #
    #   During prediction, you can compute the probability density function
    #   of a Gaussian distribution using `scipy.stats.norm`, which offers
    #   `pdf` and `logpdf` methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    #
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Because Bernoulli NB works with binary data, binarize the features as
    #   [feature_value >= 8], i.e., consider a feature as one iff it is >= 8,
    #   during both estimation and prediction.
    #
    # In all cases, the class prior is the distribution of the train data classes.

    if args.naive_bayes_type == "gaussian":
        gauss_params = np.full((train_data.shape[1], args.classes, 2), 0.0)
    elif args.naive_bayes_type == "multinomial":
        mul_param = np.full((train_data.shape[1], args.classes), 0.0)
    elif args.naive_bayes_type == "bernoulli":
        bern_param = np.full((train_data.shape[1], args.classes), 0.0)

    for c in range(args.classes):
        # train_data values that map to class c
        train_data_c = train_data[train_target == c]
        if args.naive_bayes_type == "gaussian":
            # mean and variance of train_data_c
            gauss_params[:, c, 0] = np.mean(train_data_c, axis=0)
            gauss_params[:, c, 1] = np.sum((train_data_c - gauss_params[:,c, 0])**2, axis=0) / train_data_c.shape[0] + args.alpha
        elif args.naive_bayes_type == "multinomial":
            mul_param[:, c] = np.sum(train_data_c, axis=0) + args.alpha    # the numerator
            mul_param[:, c] /= np.sum(mul_param[:, c])             # the denominator (normalization)
        elif args.naive_bayes_type == "bernoulli":
            bern_param[:, c] = np.sum(train_data_c >= 8, axis=0) + args.alpha
            bern_param[:, c] /= (len(train_data_c) + 2 * args.alpha)


    # TODO: Predict the test data classes, and compute
    # - the test set accuracy, and
    # - the joint log-probability of the test set, i.e.,
    #     \sum_{(x_i, t_i) \in test set} \log P(x_i, t_i).
    prior_probs = np.bincount(train_target) / len(train_target)

    # Model prediction
    test_log_probabilities = np.zeros((len(test_data), args.classes))
    test_log_probabilities += np.log(prior_probs)
    if args.naive_bayes_type == "gaussian":
        test_log_probabilities += np.sum(scipy.stats.norm(loc=gauss_params[:, :, 0], scale=np.sqrt(gauss_params[:, :, 1])).logpdf(np.expand_dims(test_data, -1)), axis=1)

    if args.naive_bayes_type == "multinomial":
        test_log_probabilities += np.sum(np.expand_dims(test_data, -1) * np.log(mul_param[:, :]), axis=1)

    if args.naive_bayes_type == "bernoulli":
        test_log_probabilities += np.sum(np.expand_dims(test_data >= 8, -1) * np.log(bern_param[:, :]), axis=1)
        test_log_probabilities += np.sum(np.expand_dims(test_data < 8, -1) * np.log(1 - bern_param[:, :]), axis=1)

    test_accuracy = np.mean(test_target == np.argmax(test_log_probabilities, axis=1))

    # Extracting log probabilities corresponding to the actual classes
    actual_class_log_probs = test_log_probabilities[np.arange(len(test_target)), test_target]

    # Summing these log probabilities for the joint log-probability
    joint_log_probability = np.sum(actual_class_log_probs)

    return 100 * test_accuracy, joint_log_probability


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy, test_log_probability = main(args)

    print("Test accuracy {:.2f}%, log probability {:.2f}".format(test_accuracy, test_log_probability))
