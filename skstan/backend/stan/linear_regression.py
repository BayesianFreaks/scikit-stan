from abc import ABCMeta
from abc import abstractmethod


class BaseStanLinearRegression(metaclass=ABCMeta):
    """
    Abstract base class for Linear regression using Stan.
    """


class StanLinearRegression(BaseStanLinearRegression):
    """
    Linear regression.
    """

    def __init__(self):
        pass


class StanLogisticRegression(BaseStanLinearRegression):
    """
    Logistic regression.
    """

    def __init__(self):
        pass


class StanPoissonRegression(BaseStanLinearRegression):
    """
    Poisson regression.
    """

    def __init__(self):
        pass
