from abc import ABCMeta
from abc import abstractmethod

from skstan.stan.omm.ele import StanElement


class StanDataType(metaclass=ABCMeta):

    def __init__(self, variable_name: str):
        self.variable_name_el = StanElement(variable_name)

    @abstractmethod
    def declare(self):
        pass


class StanDataDeclareMixin:

    def _declare(self, variable_name: StanElement, *args):
        return StanElement(self.TYPE_NAME.format(*args)).concat_with_blank(variable_name).semicolon()


class Int(StanDataType, StanDataDeclareMixin):

    TYPE_NAME = 'int'

    def __init__(self, variable_name: str):
        super().__init__(variable_name)

    def declare(self):
        return self._declare(self.variable_name_el)


class Real(StanDataType, StanDataDeclareMixin):

    TYPE_NAME = 'real{}'

    def __init__(self, variable_name: str, lower=None):
        self.lower = lower
        super().__init__(variable_name)

    def declare(self):
        if self.lower is not None:
            return self._declare(self.variable_name_el, '<lower={}>'.format(self.lower))
        return self._declare(self.variable_name_el, '')


class Vector(StanDataType, StanDataDeclareMixin):

    TYPE_NAME = 'vector[{}]'

    def __init__(self, variable_name: str, dim: Int):
        self.dim = dim
        super().__init__(variable_name)

    def declare(self):
        return self._declare(self.variable_name_el, self.dim.variable_name_el.value)


class Matrix(StanDataType, StanDataDeclareMixin):

    TYPE_NAME = 'matrix[{}, {}]'

    def __init__(self, variable_name: str, dim1: Int, dim2: Int):
        self.dim1 = dim1
        self.dim2 = dim2
        super().__init__(variable_name)

    def declare(self):
        return self.dim1.declare() + self.dim2.declare() + \
               self._declare(self.variable_name_el, self.dim1.variable_name_el.value, self.dim2.variable_name_el.value)
