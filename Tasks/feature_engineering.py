#!/usr/bin/env python3

import argparse

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline  

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="diabetes", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    data_train, data_test, target_train, target_test = \
        sklearn.model_selection.train_test_split(dataset.data,
                                                 dataset.target,
                                                 test_size=args.test_size,
                                                 random_state=args.seed)
    

    # TODO: Process the input columns in the following way:
    #
    # - if a column has only integer values, consider it a categorical column
    #   (days in a week, dog breed, ...; in general, integer values can also
    #   represent numerical non-categorical values, but we use this assumption
    #   for the sake of exercise). Encode the values with one-hot encoding
    #   using `sklearn.preprocessing.OneHotEncoder` (note that its output is by
    #   default sparse, you can use `sparse=False` to generate dense output;
    #   also use `handle_unknown="ignore"` to ignore missing values in test set).
    #
    # - for the rest of the columns, normalize their values so that they
    #   have mean 0 and variance 1; use `sklearn.preprocessing.StandardScaler`.
    #
    # In the output, first there should be all the one-hot categorical features,
    # and then the real-valued features. To process different dataset columns
    # differently, you can use `sklearn.compose.ColumnTransformer`.
    categorical_columns = [i for i, column in enumerate(dataset.data.T) if np.all(column.astype(int) == column)]
    column_transformer = ColumnTransformer([
        ("one_hot", OneHotEncoder(sparse=False, handle_unknown="ignore"), categorical_columns),
        ("scaler", StandardScaler(), [i for i in range(dataset.data.shape[1]) if i not in categorical_columns])
    ])


    # TODO: To the current features, append polynomial features of order 2.
    # If the input values are `[a, b, c, d]`, you should append
    # `[a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2]`. You can generate such polynomial
    # features either manually, or you can employ the provided transformer
    #   sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)
    # which appends such polynomial features of order 2 to the given features.

    # train_data_transformed = column_transformer.fit_transform(data_train)
    # poly = PolynomialFeatures(2, include_bias=False)
    # train_data = poly.fit_transform(train_data_transformed)

    # TODO: You can wrap all the feature processing steps into one transformer
    # by using `sklearn.pipeline.Pipeline`. Although not strictly needed, it is
    # usually comfortable.
    pipeline = Pipeline([
        ("preprocessing", column_transformer),
        ("poly_features", PolynomialFeatures(2, include_bias=False))
    ])

    # TODO: Fit the feature preprocessing steps (the composed pipeline with all of
    # them; or the individual steps, if you prefer) on the training data (using `fit`).
    # Then transform the training data into `train_data` (with a `transform` call;
    # however, you can combine the two methods into a single `fit_transform` call).
    # Finally, transform testing data to `test_data`.
    # Identify categorical columns

    train_data = pipeline.fit_transform(data_train)
    test_data = pipeline.transform(data_test)

    return train_data[:5], test_data[:5]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_data, test_data = main(args)
    for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            print(" ".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 140))),
                  *["..."] if dataset.shape[1] > 140 else [])