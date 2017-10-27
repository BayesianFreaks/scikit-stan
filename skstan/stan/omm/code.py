from skstan.stan.omm.datatype import StanDataType
from skstan.stan.omm.ele import StanElement

class CompilerMixin:

    def compile(self):
        pass


class StanCode:

    def __init__(self, *variables):
        self.variables = variables

    def _parenthesis(self, code):
        return self.REP + '{' + code + '}'

    def _default_render(self):
        def_list = [v.declare() for v in self.variables]
        return self._parenthesis(StanElement.join(def_list))


class Data(StanCode, CompilerMixin):

    REP = 'data'

    def __init__(self, *variables: StanDataType):
        super().__init__(*variables)

    def render(self):
        return self._default_render()


class Parameters(StanCode, CompilerMixin):

    REP = 'parameters'

    def __init__(self, *variables: StanDataType):
        super().__init__(*variables)

    def render(self):
        return self._default_render()


class TransformedParameters(StanCode, CompilerMixin):

    REP = 'transformed parameters'

    def __init__(self, *variables: StanDataType):
        super().__init__(*variables)

    def render(self):
        # TODO: implement
        pass


class Model(StanCode, CompilerMixin):

    REP = 'model'

    def __init__(self, *variables: StanDataType):
        super().__init__(*variables)

    def render(self):
        # TODO: implement
        pass
