from skstan.backend.tfp.model import TFPLinearRegression
from skstan.backend.tfp.model import TFPLogisticRegression
from skstan.backend.tfp.model import TFPPoissonRegression


class TFPBackend:
    """
    TensorFlow Probability backend class.
    This class provides model classes which are implemented by using TensorFlow
    Probability.
    """

    _BACKEND = 'tfp'

    @classmethod
    def name(cls):
        """

        Returns
        -------
        str:
            Current backend name.

        """
        return cls._BACKEND

    # Linear Regression.
    LinearRegression = TFPLinearRegression
    LogisticRegression = TFPLogisticRegression
    PoissonRegression = TFPPoissonRegression
