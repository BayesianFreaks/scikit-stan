from abc import ABCMeta, abstractmethod
from typing import Callable
from typing import Iterable

import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import edward2 as ed


class BaseTFPLinearRegression(metaclass=ABCMeta):
    """
    Abstract base class for Linear regression using TensorFlow Probability.
    """

    def make_log_join_fn(
            self, model_fn: Callable[[Iterable], tfd.Distribution]):
        return ed.make_log_joint_fn(model_fn)

    @abstractmethod
    def posterior_dist(self, features):
        pass


class TFPLinearRegression(BaseTFPLinearRegression):
    """
    Linear regression implementation using TensorFlow Probability.
    """

    def __init__(self):
        pass

    def posterior_dist(self, features):
        """

        Parameters
        ----------
        features

        Returns
        -------

        """
        pass


class TFPLogisticRegression(BaseTFPLinearRegression):
    """
    Logistic regression implementation using TensorFlow Probability.
    """

    def __init__(self):
        pass

    def set_params(self):
        pass

    def posterior_dist(self, features) -> tfd.Distribution:
        """

        Parameters
        ----------
        features

        Returns
        -------

        """
        coeffs = ed.MultivariateNormalDiag(
            loc=tf.zeros(features.shape[1]), name="coeffs")
        labels = ed.Bernoulli(
            logits=tf.tensordot(features, coeffs, [[1], [0]]), name="labels")
        return labels


class PoissonRegression(BaseTFPLinearRegression):
    """
    Poisson regression implementation using TensorFlow Probability.
    """

    def __init__(self):
        pass

    def posterior_dist(self, features):
        """

        Parameters
        ----------
        features

        Returns
        -------

        """
        pass
