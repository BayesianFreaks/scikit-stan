from skstan.backend.stan.model import StanLinearRegression
from skstan.backend.stan.model import StanLogisticRegression
from skstan.backend.stan.model import StanPoissonRegression


class StanBackend:
    """
    Stan backend class.
    This class provides model classes which are implemented using Stan.

    """

    LinearRegression = StanLinearRegression
    LogisticRegression = StanLogisticRegression
    PoissonRegression = StanPoissonRegression
