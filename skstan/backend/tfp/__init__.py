from skstan.backend.tfp.tfp_backend import TFPBackend


__all__ = [
    'TFPBackend',
]


# This backend class will be imported in skstan.backend.__init__.py,
# if tensorflow probability backend backend is selected.
Backend = TFPBackend
