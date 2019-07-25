from skstan.model import BaseModel


class BaseEstimator(BaseModel):
    """
    Abstract base class for all estimators in scikit-stan.

    """

    def get_params(self, deep=True):
        """

        Parameters
        ----------
        deep

        Returns
        -------

        """
        pass

    @classmethod
    def _get_param_names(cls):
        pass
