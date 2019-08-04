from skstan.backend.stan.model import StanLinearRegression
from skstan.backend.stan.model import StanLogisticRegression
from skstan.backend.stan.model import StanPoissonRegression


class StanBackend:
    """
    Stan backend class.
    This class provides model classes which are implemented by using Stan.

    """

    _BACKEND = 'stan'

    @classmethod
    def get_name(cls):
        """
        Returns
        -------
        str:
            Current backend name.

        """
        return cls._BACKEND

    # Linear Regression.
    LinearRegression = StanLinearRegression
    LogisticRegression = StanLogisticRegression
    PoissonRegression = StanPoissonRegression
