#!/usr/bin/env python3

import argparse
import lzma
import pickle
import os
import sys
import urllib.request
import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import re

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=False, action="store_true", help="Use IDF weights")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=45, type=int, help="Random seed")
parser.add_argument("--tf", default=False, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
# For these and any other arguments you add, ReCodEx will keep your default value.


class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names


def main(args: argparse.Namespace) -> float:
    # Load the 20newsgroups data.
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

    # Create train-test split.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)
    

    # print(str(train_data[0].length))
    # TODO: Create a feature for every term that is present at least twice
    # in the training data. A term is every maximal sequence of at least 1 word character,
    # where a word character corresponds to a regular expression `\w`.

    # Storing all words in a dictionary with their frequency
    term_freq = dict()
    for doc in train_data:
        for word in re.findall(r"\w+", doc):
            if word in term_freq:
                term_freq[word] += 1
            else:
                term_freq[word] = 1

    # Filtering out words that appear less than twice
    terms = {term for term, freq in term_freq.items() if freq >= 2}
    # Assigning an id to each term
    terms_ids = {term: id for id, term in enumerate(terms)}


    # TODO: For each document, compute its features as
    # - term frequency (TF), if `args.tf` is set (term frequency is
    #   proportional to counts but normalized to sum to 1);
    # - otherwise, use binary indicators (1 if a given term is present, else 0)
    #
    # Then, if `args.idf` is set, multiply the document features by the
    # inverse document frequencies (IDF), where
    # - use the variant which contains `+1` in the denominator;
    # - the IDFs are computed on the train set and then reused without
    #   modification on the test set.

    idf = None
    if args.idf:
        idf = np.zeros(len(terms))

        # Convert each document into a set of unique terms
        unique_terms_in_docs = [set(re.findall(r'\w+', doc)) for doc in train_data]

        # Count the number of documents containing each term
        for term in terms:
            doc_count_containing_term = sum(term in doc_set for doc_set in unique_terms_in_docs)
            idf[terms_ids[term]] = np.log(len(train_data) / (doc_count_containing_term + 1))

    def encode_docs(docs, idf):
        encoded_docs = np.zeros((len(docs), len(terms)))
        for doc_id, doc in enumerate(docs):
            doc_terms = re.findall(r"\w+", doc)
            doc_len = 0
            for term in doc_terms:
                if term not in terms:
                    continue

                doc_len += 1
                term_id = terms_ids[term]
                if args.tf:
                    encoded_docs[doc_id, term_id] += 1
                else:
                    encoded_docs[doc_id, term_id] = 1
            
            if args.tf:
                encoded_docs[doc_id] /= doc_len
            if args.idf:
                encoded_docs[doc_id] *= idf
        return encoded_docs
    
    # TODO: Train a `sklearn.linear_model.LogisticRegression(solver="liblinear", C=10_000)`
    # model on the train set, and classify the test set.

    train_data_encoded = encode_docs(train_data, idf)
    test_data_encoded = encode_docs(test_data, idf)
    model = sklearn.linear_model.LogisticRegression(solver="liblinear", C=10_000)
    model.fit(train_data_encoded, train_target)
    test_predictions = model.predict(test_data_encoded)

    # TODO: Evaluate the test set performance using a macro-averaged F1 score.
    f1_score = sklearn.metrics.f1_score(test_target, test_predictions, average="macro")

    return 100 * f1_score


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(args)
    print("F-1 score for TF={}, IDF={}: {:.1f}%".format(args.tf, args.idf, f1_score))
