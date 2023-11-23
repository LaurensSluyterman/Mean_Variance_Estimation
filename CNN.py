import keras.models
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
import gc
from tensorflow import keras
from utils import normalize, reverse_normalized
from keras.layers import Input, Dense, concatenate, MaxPooling2D, Flatten, Conv2D, Activation, BatchNormalization
from keras.models import Model
l2 = keras.regularizers.l2


class MVECNNNetwork:
    """
    This class represents a trained Mean Variance Estimation network.

    The network is trained using the negative loglikelihood as loss function
    and output an estimate for the mean, f, and standard deviation, sigma.
    Normalization happens by default (can be switched off). The
    predictions are given in the original scale. Per default, no
    warm-up is used, this can be turned on. When using warm-up, the mean
    estimate is kept fixed during the second phase. This can be turned off, in
    which case both the mean and variance estimate get updated.
    Note that a fixed seed is used in the current implementation. The
    weights are restored to ones with the lowest NLL on the validation set.

    Attributes:
        model: The trained CNN

    Methods:
        f: An estimate of the mean function, without any normalization .
        sigma: An estimate of the standard deviation, without any normalization.
    """

    def __init__(self, *, X, Y, n_hidden_mean, n_hidden_var, n_epochs,
                 reg_mean=0, reg_var=0, batch_size=32, verbose=False,
                 normalization=True, warmup_fraction=0, fixed_mean=True, beta=None, X_val=None, Y_val=None):
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
            warmup_fraction (float): Determines if the network first learns the
                mean estimate. Default is 0, the mean and variance
                are updated simultaneously. A value of 1/3 means that the
                variance is kept fixed during the first third of training.
            fixed_mean (bool): In case of a warmup, this determines if the
                mean estimate is kept fixed during the second training phase.
            X_val: The unnormalized validation covariates.
            Y_val: The unnormalized validation targets.
        """
        self._normalization = normalization


        self._Y_mean = np.mean(Y, axis=0)
        self._Y_std = np.std(Y, axis=0)
        Y_val = normalize(Y_val, mean=self._Y_mean, std=self._Y_std)
        Y = normalize(Y)
        model = train_network(X_train=X, Y_train=Y,
                              n_hidden_mean=n_hidden_mean,
                              n_hidden_var=n_hidden_var,
                              loss=get_loss(variance_transformation, beta),
                              reg_mean=reg_mean,
                              reg_var=reg_var,
                              batch_size=batch_size,
                              n_epochs=n_epochs,
                              verbose=verbose,
                              warmup_fraction=warmup_fraction,
                              fixed_mean=fixed_mean,
                              X_val=X_val,
                              Y_val=Y_val)
        self.model = model

    def f(self, X_test):
        """Return the mean prediction"""
        predictions = self.model.predict(X_test, verbose=0)[:, 0]
        _ = gc.collect()  # Alleviates weird memory leak
        return reverse_normalized(predictions, self._Y_mean, self._Y_std)

    def sigma(self, X_test):
        """Return the standard deviation prediction"""
        predictions = np.sqrt(variance_transformation(self.model.predict(X_test, verbose=0)[:, 1], ker=False))
        _ = gc.collect()
        return predictions * self._Y_std


def train_network(*, X_train, Y_train, n_hidden_mean, n_hidden_var, n_epochs,
                  loss, reg_mean=0, reg_var=0, batch_size=32,
                  verbose=False, warmup_fraction=0, fixed_mean=False, X_val=None,
                  Y_val=None, seed=813):
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
            warmup_fraction (float): Determines if the network first learns the
                mean estimate. Default is 0, the mean and variance
                are updated simultaneously. A value of 1/3 means that the
                variance is kept fixed during the first third of training.
            fixed_mean (bool): In case of a warmup, this determines if the
                mean estimate is kept fixed during the second training phase.
            X_val: The unnormalized validation covariates.
            Y_val: The unnormalized validation targets.
            seed: The random seed that is used for training.

    Returns:
        model: A trained network that outputs a mean and log of standard
            deviation.
    """
    # Fixed seed
    keras.utils.set_random_seed(seed)
    lr = 3e-5  # Learning rate for Adam optimizer
    clip_value = 5  # Clip value for Adam optimizer

    # CNN backbone
    inputs = Input(shape=(64, 64, 3))

    # First Convolutional Block
    x = Conv2D(filters=4, kernel_size=(3, 3), strides=1, kernel_regularizer=l2(1))(inputs)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Second Convolutional Block
    x = Conv2D(filters=4, kernel_size=(3, 3), strides=1, kernel_regularizer=l2(1))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Third Convolutional Block
    x = Conv2D(filters=8, kernel_size=(3, 3), strides=1, kernel_regularizer=l2(1))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Fourth Convolutional Block
    x = Conv2D(filters=8, kernel_size=(3,3), strides=1, kernel_regularizer=l2(1))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Flattening followed by Dense Layers
    x = Flatten()(x)
    inter_mean = Dense(n_hidden_mean[0],
                       kernel_regularizer=l2(reg_mean),
                       name='mean')(x)
    inter_mean = BatchNormalization()(inter_mean)
    inter_mean = Activation('elu')(inter_mean)
    for i in range(len(n_hidden_mean) - 1):
        inter_mean = Dense(n_hidden_mean[i + 1],
                           kernel_regularizer=keras.regularizers.l2(reg_mean),
                           name=f'mean{i}')(inter_mean)
        inter_mean = BatchNormalization()(inter_mean)
        inter_mean = Activation('elu')(inter_mean)
    inter_var = Dense(n_hidden_var[0],
                      kernel_regularizer=l2(reg_var), name='var')(x)
    inter_var = BatchNormalization()(inter_var)
    inter_var = Activation('elu')(inter_var)
    for i in range(len(n_hidden_var) - 1):
        inter_var = Dense(n_hidden_var[i + 1],
                          kernel_regularizer=keras.regularizers.l2(reg_var),
                          name=f'var{i}')(inter_var)
        inter_var = BatchNormalization()(inter_var)
        inter_var = Activation('elu')(inter_var)
    output_mean = Dense(1, activation='linear',
                        kernel_regularizer=keras.regularizers.l2(reg_mean),
                        name='meanout')(inter_mean)
    output_var = Dense(1, activation='linear',
                       kernel_regularizer=keras.regularizers.l2(reg_var),
                       kernel_initializer=tf.keras.initializers.Constant(value=0),
                       bias_initializer=tf.keras.initializers.Constant(value=0),
                       name='varout')(inter_var)
    outputs = concatenate([output_mean, output_var])
    model = Model(inputs, outputs)

    # Without a warmup, we simultaneously learn the mean and variance
    if warmup_fraction == 0:
        my_metric = loss
        model.compile(loss=loss, optimizer=keras.optimizers.Adam(clipvalue=clip_value, learning_rate=lr),
                      metrics=[my_metric])
        save_best = tf.keras.callbacks.ModelCheckpoint('./UTKFaceData/tempmodel.h5',
                                                        monitor="val_negative_log_likelihood", mode="min",
                                                        save_best_only=True, verbose=1)
        filepath = "./UTKFaceData/transition/nowu/{epoch:d}.hdf5"
        checkpoint2 = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                        monitor="val_negative_log_likelihood", mode="min",
                                                        save_best_only=False, verbose=1)
        model.save('./UTKFaceData/transition/nowu/0.hdf5')
        model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=batch_size, epochs=n_epochs,
                  verbose=verbose, callbacks=[save_best, checkpoint2])
        model.load_weights('./UTKFaceData/tempmodel.h5') # Restore optimal weights
        return model

    # Freeze the variance layers for the warm-up
    for layer in model.layers:
        if layer.name[0] == 'v':
            layer.trainable = False

    model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=lr, clipvalue=clip_value))
    model.save('./UTKFaceData/transition/wu/0.hdf5')
    filepath = "./UTKFaceData/transition/wu/{epoch:d}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                    monitor="val_negative_log_likelihood", mode="min",
                                                    save_best_only=False, verbose=1)
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=batch_size, epochs=int(n_epochs * warmup_fraction),
              verbose=verbose, callbacks=[checkpoint])

    # Unfreeze the variance layers
    for layer in model.layers:
        layer.trainable = True

    # Freeze the mean layers for the second phase if desired
    if fixed_mean:
        for layer in model.layers:
            if layer.name[0] == 'm':
                layer.trainable = False

    # Compile and train the final model
    filepath = "./UTKFaceData/transition/wu/{epoch:02d}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                    monitor="val_negative_log_likelihood", mode="min",
                                                    save_best_only=False, verbose=1)
    save_best = tf.keras.callbacks.ModelCheckpoint('./UTKFaceData/tempmodel.h5',
                                                   monitor="val_negative_log_likelihood", mode="min",
                                                   save_best_only=True, verbose=1)
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=lr, clipvalue=clip_value))
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=batch_size, epochs=n_epochs,
              verbose=verbose, callbacks=[save_best, checkpoint], initial_epoch=int(n_epochs * warmup_fraction))
    model.load_weights('./UTKFaceData/tempmodel.h5')  # Restore optimal weights
    return model


def get_loss(transform, beta=None):
    if beta:
        def negative_log_likelihood(targets, outputs, beta=beta):
            """Compute beta-NLL loss."""
            mu = outputs[..., 0:1]
            var = transform(outputs[..., 1:2])
            loss = (K.square((targets - mu)) / var + K.log(var))
            loss = loss * K.stop_gradient(var) ** beta
            return loss
        return negative_log_likelihood
    else:
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
