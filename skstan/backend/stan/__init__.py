from skstan.backend.stan.stan_backend import StanBackend
from skstan.backend.stan.stan_backend import StanModelLoader


__all__ = [
    'StanBackend',
    'StanModelLoader',
]

# This backend class will be imported in skstan.backend.__init__.py, if stan
# backend is selected.
Backend = StanBackend
