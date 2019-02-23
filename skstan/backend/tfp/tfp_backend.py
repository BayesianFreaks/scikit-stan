from skstan.backend.tfp.model import TFPLinearRegression
from skstan.backend.tfp.model import TFPLogisticRegression
from skstan.backend.tfp.model import TFPPoissonRegression


class TFPBackend:
    """
    TensorFlow backend class.
    This class provides model clases which are implemented using TensorFlow
    Probability.
    """

    # Linear Regression.
    LinearRegression = TFPLinearRegression
    LogisitcRegression = TFPLogisticRegression
    PoissionRegression = TFPPoissonRegression
