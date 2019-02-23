from skstan.backend.stan.stan_backend import StanBackend


__all__ = [
    'StanBackend',
]

# This backend class will be imported in skstan.backend.__init__.py, if stan
# backend is selected.
Backend = StanBackend
