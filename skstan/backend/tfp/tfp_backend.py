import tensorflow_probability as tfp

from skstan.backend import BaseBackend


class TFPBackend(BaseBackend):
    """
    TensorFlow backend class.
    """
    def __init__(self):
        # print backend name to check .
        print("tfp backend.")

    def sample(self):
        pass


_tfd = tfp.distributions


class TFDistribution:
    """
    TensorFlow Distribution class.

    This class provides typical distributions.
    """

    @staticmethod
    def normal(mean, std) -> _tfd.Normal:
        """
        Normal distribution object.

        Parameters
        ----------
        mean: float or int
            The mean value of Normal distribution.
        std: float or int
            The standard deviation of Normal distribution.
        Returns
            The Normal distribution instance of TensorFlow Probability.
        -------

        """
        pass

    @staticmethod
    def bernoulli(probs=()) -> _tfd.Bernoulli:
        """

        Parameters
        ----------
        probs

        Returns
        -------

        """
        pass

    def binomial(self) -> _tfd.Binomial:
        pass
