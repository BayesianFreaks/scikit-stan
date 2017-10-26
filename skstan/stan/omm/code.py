from typing import List
from skstan.stan.omm.variable import VariableDefinition


class CompileMixin:

    def compile():
        pass


class Data:

    def __init__(self, variables: List[VariableDefinition]):
        pass


class Parameters:

    def __init__(self, variables: List[VariableDefinition]):
        pass


class TransformedParameters:

    def __init__(self, variables: List[VariableDefinition]):
        pass


class Model:

    def __init__(self, variables: List[VariableDefinition]):
        pass
