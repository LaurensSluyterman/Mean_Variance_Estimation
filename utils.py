import numpy as np


def normalize(x, mean=None, std=None):
    """This function normalizes x using a given mean and standard deviation"""
    if mean is None:
        mean = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)
    return (x - mean) / std


def reverse_normalized(x_normalized, mean, std):
    """This function reverses the normalization done by the function 'normalize' """
    return x_normalized * std + mean


def load_data(directory):
    """Load data from given directory"""
    _DATA_FILE = "./UCI_Datasets/" + directory + "/data/data.txt"
    _INDEX_FEATURES_FILE = "./UCI_Datasets/" + directory + "/data/index_features.txt"
    _INDEX_TARGET_FILE = "./UCI_Datasets/" + directory + "/data/index_target.txt"
    index_features = np.loadtxt(_INDEX_FEATURES_FILE)
    index_target = np.loadtxt(_INDEX_TARGET_FILE)
    data = np.loadtxt(_DATA_FILE)
    X = data[:, [int(i) for i in index_features.tolist()]]
    Y = data[:, int(index_target.tolist())]
    return X, Y


def average_loglikelihood(Y, f, sigma):
    """Calculate the average loglikelihood"""
    N = len(Y)
    LL = - np.sum(np.log(sigma)) - N / 2 * np.log(2 * np.pi) - np.sum(0.5 * ((Y - f) / sigma)**2)
    return LL / N


def rmse(A, B):
    """Calculate the root-mean-squared error"""
    assert np.shape(A) == np.shape(B)
    return np.sqrt(np.mean((A-B)**2))


def maxdiagonal(dictionary):
    """Calculate the maximum value of a dictionary on the diagonal.

    This is a somewhat specific function. The takes as input a
    dictionary that has tuples (a,b) as keys and finds the maximum values
    for the keys for which a==b. This is used in the hyperparameter search
    to check which  equal regularization setting has the highest loglikelihood.

    Arguments:
        dictionary: A dictionary with as keys tuples (a,b) corresponding to the
         regularization of the mean and the variance.

    Returns:
        Tuple (a, a) which corresponds to the key with the highest value.
    """
    dictionary = dict(filter(lambda e: e[0][0] == e[0][1], dictionary.items()))
    return list(max(dictionary, key=dictionary.get))
