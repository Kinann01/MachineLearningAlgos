#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD training epochs")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization strength")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=92, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[list[float], float, float]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset.
    data, target = sklearn.datasets.make_regression(n_samples=args.data_size, random_state=args.seed)

    # TODO: Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # print(data)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data = sklearn.model_selection.train_test_split(data, test_size=args.test_size, random_state=args.seed)
    train_target, test_target  = sklearn.model_selection.train_test_split(target, test_size=args.test_size, random_state=args.seed )
    
    # Generate initial linear regression weights.
    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)

    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `len(train_data)`.
        #
        # The gradient for the input example $(x_i, t_i)$ is
        # - $(x_i^T weights - t_i) x_i$ for the unregularized loss (1/2 MSE loss),
        # - $args.l2 * weights_with_bias_set_to_zero$ for the L2 regularization loss,
        #   where we set the bias to zero because the bias should not be regularized,
        # and the SGD update is
        #   weights = weights - args.learning_rate * gradient

        # Lets first distribute the training data and the target based on the permutation function
        data_shuffled = train_data[permutation]
        target_shuffled = train_target[permutation]

        # Process the data in order of permutation
        for batch_start in range(0, len(train_data), args.batch_size):

            # Now we get batch by batch and calculate the gradient and based on its descent. 
            oneDataBatch = data_shuffled[batch_start : batch_start + args.batch_size]
            oneTargetBatch = target_shuffled[batch_start : batch_start + args.batch_size]

            # Initialize the gradient as zero
            gradient = np.zeros_like(weights)

            # Gradient formula
            predictions = np.dot(oneDataBatch, weights)
            errors = predictions - oneTargetBatch
            batchGradient = np.dot(oneDataBatch.T, errors)
            gradient = (1 / args.batch_size) * batchGradient

            # Compute the L2 regularized gradient
            gradientWithL2 = args.l2 * weights
            # Since bias should be unregularized
            gradientWithL2[-1] = 0
            # update gradient and our weights for each batch
            gradient += gradientWithL2
            weights = weights - (args.learning_rate * gradient)

        # TODO: Append current RMSE on train/test to `train_rmses`/`test_rmses`.

        predictTrainData = np.dot(train_data, weights)
        predictTestData = np.dot(test_data, weights)

        mseTrainData = np.mean((train_target - predictTrainData) ** 2)
        mseTestData = np.mean((test_target - predictTestData) ** 2)

        currentRMSE_train = np.sqrt(mseTrainData)
        currentRMSE_test = np.sqrt(mseTestData)

        train_rmses.append(currentRMSE_train)
        test_rmses.append(currentRMSE_test)

    # TODO: Compute into `explicit_rmse` test data RMSE when fitting
    # `sklearn.linear_model.LinearRegression` on `train_data` (ignoring `args.l2`).
    model = sklearn.linear_model.LinearRegression()
    model.fit(train_data, train_target)
    predict = model.predict(test_data)
    explicit_rmse = sklearn.metrics.mean_squared_error(test_target, predict , squared=False)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, test_rmses[-1], explicit_rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, sgd_rmse, explicit_rmse = main(args)
    print("Test RMSE: SGD {:.3f}, explicit {:.1f}".format(sgd_rmse, explicit_rmse))
    print("Learned weights:", *("{:.3f}".format(weight) for weight in weights[:12]), "...")