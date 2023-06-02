import numpy as np

__all__ = ['mse', 'rmse', 'mll', 'mae', 'mre', 'smll', 'smse', 'srmse', 'r2']


def mre(mean, data):
    """Mean relative error.

    :param mean: (tensor) mean of prediction.
    :param data: (tensor) reference data.
    """
    return ((mean - data).abs().sum(0) / data.abs().sum(0)).mean()


def mse(mean, data):
    """Mean squared error.

    :param mean: (tensor) mean of prediction.
    :param data: (tensor) reference data.
    """
    return ((mean - data) ** 2).mean()


def smse(mean, data):
    """Standardised mean squared error.

    :param mean: (tensor) mean of prediction.
    :param data: (tensor) reference data.
    """
    return mse(mean, data) / mse(data.mean(), data)


def rmse(mean, data):
    """Root mean squared error.

    :param mean: (tensor) mean of prediction.
    :param data: (tensor) reference data.
    """
    return mse(mean, data) ** .5


def srmse(mean, data):
    """Standardised root mean squared error.

    :param mean: (tensor) mean of prediction.
    :param data: (tensor) reference data.
    """
    return rmse(mean, data) / rmse(data.mean(), data)


def mae(mean, data):
    """Mean absolute error.

    :param mean: (tensor) mean of prediction.
    :param data: (tensor) reference data.
    """
    return np.abs(mean - data).mean()


def mll(mean, variance, data):
    """Mean log loss.

    :param mean: (tensor) mean of prediction.
    :param variance: (tensor) variance of prediction.
    :param data: (tensor) reference data.
    """
    return (0.5 * np.log(2 * np.pi * variance) +
            0.5 * (mean - data) ** 2 / variance).mean()


def smll(mean, variance, data):
    """Standardised mean log loss.

    :param mean: (tensor) mean of prediction.
    :param variance: (tensor) variance of prediction.
    :param data: (tensor) reference data.
    """
    return mll(mean, variance, data) - mll(data.mean(), data.var(ddof=0), data)


def r2(mean, data):
    """R-squared.

    :param mean: (tensor) mean of prediction.
    :param data: (tensor) reference data.
    """
    return 1 - ((data - mean) ** 2).mean() / data.var(ddof=0)
