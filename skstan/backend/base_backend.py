from abc import ABCMeta, abstractmethod


class BaseBackend(metaclass=ABCMeta):
    """
    Abstract base class for backends.
    """

    @abstractmethod
    def sample(self):
        pass
