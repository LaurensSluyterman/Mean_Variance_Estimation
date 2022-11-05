import keras.models
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # Needed to prevent memory leaks
import gc
from tensorflow import keras
from utils import normalize, reverse_normalized
from keras.layers import Input, Dense, Concatenate, concatenate
from keras.models import Model
from sklearn.model_selection import train_test_split


l2 = keras.regularizers.l2


class MVENetwork:
    """
    This class represents a trained Mean Variance Estimation network.

    The network is trained using the negative loglikelihood as loss function
    and output an estimate for the mean, f, and standard deviation, sigma.
    Normalization happens by default (can be switched off). The
    predictions are given in the original scale. Per default, no
    warm-up is used, this can be turned on. When using warm-up, the mean
    estimate is kept fixed during the second phase. This can be turned off, in
    which case both the mean and variance estimate get updated.

    Attributes:
        model: The trained neural network

    Methods:
        f: An estimate of the mean function, without any normalization .
        sigma: An estimate of the standard deviation, without any normalization.
    """

    def __init__(self, *, X, Y, n_hidden_mean, n_hidden_var, n_epochs,
                 reg_mean=0, reg_var=0, batch_size=None, verbose=False,
                 normalization=True, warmup=False, fixed_mean=True):
        """
        Arguments:
            X: The unnormalized training covariates.
            Y: The unnormalized training targets
            n_hidden_mean (array): An array containing the number of hidden units
                for each hidden layer of the part of the network that gives
                the predictions for the mean.
            n_hidden_var (array): An array containing the number of hidden units
                for each hidden layer of the part of the network that gives
                the predictions for the variance.
            n_epochs (int): The number of training epochs
            reg_mean: The regularization constant for the part of the network
                that gives the predictions for the mean. By default, no
                regularization is used.
            reg_var: The regularization constant for the part of the network
                that gives the predictions for the variance. By default, no
                regularization is used.
            batch_size (int): The used batch size for training, if set to None
                the standard size of 32 is used.
            verbose (bool): Determines if training progress is printed.
            normalization (bool): Determines if the covariates and targets
                are normalized before training. This normalization is
                reversed for the final predictions.
            warmup (bool): Determines if the network first learns the
                mean estimate. Default is False, the mean and variance
                are updated simultaneously.
            fixed_mean (bool): In case of a warmup, this determines if the
                mean estimate is kept fixed during the second training phase.
        """
        self._normalization = normalization

        if normalization is True:
            self._X_mean = np.mean(X, axis=0)
            self._X_std = np.std(X, axis=0)
            self._Y_mean = np.mean(Y, axis=0)
            self._Y_std = np.std(Y, axis=0)
            X = normalize(X)
            Y = normalize(Y)
        model = train_network(X_train=X, Y_train=Y, n_hidden_mean=n_hidden_mean,
                              n_hidden_var=n_hidden_var,
                              loss=get_loss(variance_transformation),
                              reg_mean=reg_mean,
                              reg_var=reg_var,
                              batch_size=batch_size,
                              n_epochs=n_epochs,
                              verbose=verbose,
                              warmup=warmup,
                              fixed_mean=fixed_mean)
        self.model = model

    def f(self, X_test):
        """Return the mean prediction"""
        if self._normalization is True:
            X_test = normalize(X_test, self._X_mean, self._X_std)
            predictions = self.model.predict(X_test, verbose=0)[:, 0]
            _ = gc.collect()  # Alleviates weird memory leak
            return reverse_normalized(predictions, self._Y_mean, self._Y_std)
        else:
            predictions = self.model.predict(X_test, verbose=0)[:, 0]
            _ = gc.collect()  # Alleviates weird memory leak
            return predictions

    def sigma(self, X_test):
        """Return the standard deviation prediction"""
        if self._normalization is True:
            X_test = normalize(X_test, self._X_mean, self._X_std)
            predictions = np.sqrt(variance_transformation(self.model.predict(X_test, verbose=0)[:, 1], ker=False))
            _ = gc.collect()
            return predictions * self._Y_std
        else:
            predictions = np.sqrt(variance_transformation(self.model.predict(X_test, verbose=0)[:, 1], ker=False))
            _ = gc.collect()
            return predictions


def train_network(*, X_train, Y_train, n_hidden_mean, n_hidden_var, n_epochs,
                  loss, reg_mean=0, reg_var=0, batch_size=None,
                  verbose=False, warmup=0, fixed_mean=False):
    """Train a network that outputs the mean and standard deviation.

    This function trains a network that outputs the mean and standard
    deviation. The network is trained using the negative loglikelihood
    of a normal distribution as the loss function.

        Arguments:
            X_train: The training covariates.
            Y_train: The training targets
            n_hidden_mean (array): An array containing the number of hidden units
                for each hidden layer of the part of the network that gives
                the predictions for the mean.
            n_hidden_var (array): An array containing the number of hidden units
                for each hidden layer of the part of the network that gives
                the predictions for the variance.
            n_epochs (int): The number of training epochs
            loss: The loss function
            reg_mean: The regularization constant for the part of the network
                that gives the predictions for the mean. By default, no
                regularization is used.
            reg_var: The regularization constant for the part of the network
                that gives the predictions for the variance. By default, no
                regularization is used.
            batch_size (int): The used batch size for training, if set to None
                the standard size of 32 is used.
            verbose (bool): Determines if training progress is printed.
            warmup (bool): Determines if the network first learns the
                mean estimate. Default is False, the mean and variance
                are updated simultaneously.
            fixed_mean (bool): In case of a warmup, this determines if the
                mean estimate is kept fixed during the second training phase.
    Returns:
        model: A trained network that outputs a mean and log of standard
            deviation.
    """
    try:
        input_shape = np.shape(X_train)[1]
    except IndexError:
        input_shape = 1
    inputs = Input(shape=(input_shape,))
    inter_mean = Dense(n_hidden_mean[0], activation='elu',
                       kernel_regularizer=l2(reg_mean),
                       name='mean')(inputs)
    for i in range(len(n_hidden_mean) - 1):
        inter_mean = Dense(n_hidden_mean[i + 1], activation='elu',
                           kernel_regularizer=keras.regularizers.l2(reg_mean),
                           name=f'mean{i}')(inter_mean)
    inter_var = Dense(n_hidden_var[0], activation='elu',
                      kernel_regularizer=l2(reg_var), name='var')(inputs)
    for i in range(len(n_hidden_var) - 1):
        inter_var = Dense(n_hidden_var[i + 1], activation='elu',
                          kernel_regularizer=keras.regularizers.l2(reg_var),
                          name=f'var{i}')(inter_var)
    output_mean = Dense(1, activation='linear',
                        kernel_regularizer=keras.regularizers.l2(reg_mean),
                        name='meanout')(inter_mean)
    output_var = Dense(1, activation='linear',
                       kernel_regularizer=keras.regularizers.l2(reg_var),
                       bias_initializer=tf.keras.initializers.Constant(value=1),
                       name='varout')(inter_var)

    outputs = concatenate([output_mean, output_var])
    model = Model(inputs, outputs)

    # Without a warmup, we simultaneously learn the mean and variance
    if not warmup:
        model.compile(loss=loss, optimizer=keras.optimizers.Adam(clipvalue=5))
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=n_epochs,
                  verbose=verbose)
        return model

    # Freeze the variance layers
    for layer in model.layers:
        if layer.name[0] == 'v':
            layer.trainable = False

    model.compile(loss=loss, optimizer=keras.optimizers.Adam(clipvalue=5))
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=n_epochs,
              verbose=verbose)
    logmse = np.log(np.mean(np.square(model.predict(X_train, verbose=0)[:, 0] - Y_train)))

    # Set the bias of the output variance to the logmse
    model.layers[-2].set_weights([model.layers[-2].get_weights()[0], np.array([logmse])])

    # Unfreerze the variance layers
    for layer in model.layers:
        layer.trainable = True

    # Freeze the mean layers if desired
    if fixed_mean:
        for layer in model.layers:
            if layer.name[0] == 'm':
                layer.trainable = False

    # Compile and train the final model
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(clipvalue=5))
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=n_epochs,
              verbose=verbose)
    return model


def get_loss(transform):
    def negative_log_likelihood(targets, outputs):
        """Calculate the negative loglikelihood."""
        mu = outputs[..., 0:1]
        var = transform(outputs[..., 1:2])
        y = targets[..., 0:1]
        loglik = - K.log(var) - K.square((y - mu)) / var
        return - loglik
    return negative_log_likelihood


def variance_transformation(b, ker=True):
    """Transform the output such that it is positive.

    An exponential transformation is used with a minimum of 1e-6
    for numerical stability.

    Arguments:
        b: The value that needs to be transformed
        ker (bool): Determines if keras backend or numpy is used. If set to
            True, keras is used.
    """
    if ker:
        return K.exp(b) + 1e-6
    else:
        return np.exp(b) + 1e-6
