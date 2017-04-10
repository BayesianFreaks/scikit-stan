from abc import ABCMeta

import six
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin


class SampledParametersMixin(ABCMeta):
    pass


class SampledCoefficientsMixin(SampledParametersMixin):
    pass


class LinearRegression(six.with_metaclass(BaseEstimator, LinearClassifierMixin)):
    pass
