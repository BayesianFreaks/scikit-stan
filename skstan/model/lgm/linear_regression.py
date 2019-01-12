from skstan.model.lgm import BaseLinearRegression


class LogisticRegression(BaseLinearRegression):
    """
    Logistic regression model class.

    Parameters
    ----------
    fit_intercept: bool, optional (defalut=True)
        hogehoge.

    intercept_scaling: float, optional (default=True)
        hogehoge

    multi_class: str,
        hogehoge

    verbose: int, optional (default=0)
        hogehoge

    n_jobs: int, optional (default=None)
        The number of CPU cores.

    class_weight: dict or 'balanced', optional (default=None)
        hogehoge.


    """

    def __init__(self, fit_intercept=True, intercept_scaling=1, multi_class=None,
                 verbose=0, n_jobs=None, class_weight=None):
        self._fit_intercept = fit_intercept
        self._intercept_scaling = intercept_scaling
        self._multi_class = multi_class
        self._verbose = verbose
        self._n_jobs = n_jobs
        self._class_weight = class_weight

    def _validate_params(self):
        pass

    def fit(self, X, y):
        """

        Parameters
        ----------
        X: {array-like, sparse matrix}
            Training vector.

        y: array-like
            Target vector relative to X.

        Returns
        -------

        """
        pass

    def predict(self, X):
        """

        Parameters
        ----------
        X: {array-like, sparse matrix}

        Returns
        -------
        T: array-like
            Returns the probability.

        """
        pass
