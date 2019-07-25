class StanModelParams:
    """
    Parameter class for Stan Model.

    Parameters
    ----------
    chains: int
        The number of chains.
    warmup: int
        The number of warmup steps which will be discarded.
    n_itr: int
        The number of iteration.
    n_jobs: int
        The number of jobs to run in parallel.
    shrinkage: int (default=10)
        The standard deviation for non-information prior distribution.
        Usually, the deviation value is very large.
    algorithm: str
        The algorithm name to be used in Stan.
    verbose: bool
        Hoge
    shrinkage: float (default=10.0)
        The standard deviation for non-information prior distribution.
        Usually, the deviation value is very large.
    """

    def __init__(self, chains: int, warmup: int, n_itr: int, n_jobs: int,
                 algorithm: str, verbose: bool, shrinkage: float):
        self._chains = chains
        self._warmup = warmup
        self._n_jobs = n_jobs
        self._n_itr = n_itr
        self._algorithm = algorithm
        self._verbose = verbose
        self._shrinkage = shrinkage

    @property
    def chains(self):
        return self._chains

    @property
    def warmup(self):
        return self._warmup

    @property
    def n_jobs(self):
        return self._n_jobs

    @property
    def n_itr(self):
        return self._n_itr

    @property
    def algorithm(self):
        return self._algorithm

    @property
    def verbose(self):
        return self._verbose

    @property
    def shrinkage(self):
        return self._shrinkage


class StanLinearRegressionParams(StanModelParams):
    pass


class StanLogisticRegressionParams(StanModelParams):
    pass


class StanPoissonRegressionParams(StanModelParams):
    pass


class TFPModelParams:
    """
    Parameter class for TensorFlow Probability Model.

    """

    def __init__(self, verbose: bool):
        # TODO: set parameter fields for tfp model.
        self._verbose = verbose

    @property
    def verbose(self):
        return self._verbose


class TFPLinearRegressionParams(TFPModelParams):
    pass


class TFPLogisticRegressionParams(TFPModelParams):
    pass


class TFPPoissonRegressionParams(TFPModelParams):
    pass
