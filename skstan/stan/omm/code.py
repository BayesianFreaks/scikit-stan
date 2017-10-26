from skstan.stan.omm.variable import VariableDefinition


class CompilerMixin:

    def compile():
        pass


class StanCode:

    def __init__(self, *variables):
        self.variables = variables

    def _parenthesis(self, code):
        return self.REP + '{\n' + code + '}'

    def _default_render(self):
        def_list = [v.render() for v in self.variables]
        return self._parenthesis('\n'.join(def_list))


class Data(StanCode, CompilerMixin):

    REP = 'data'

    def __init__(self, *variables: VariableDefinition):
        super().__init__(variables)

    def render(self):
        return self._default_render()


class Parameters(StanCode, CompilerMixin):

    REP = 'parameters'

    def __init__(self, *variables: VariableDefinition):
        super().__init__(variables)

    def render(self):
        return self._default_render()


class TransformedParameters(StanCode, CompilerMixin):

    REP = 'transformed parameters'

    def __init__(self, *variables: VariableDefinition):
        super().__init__(variables)

    def reander(self):
        # TODO: implement
        pass


class Model(StanCode, CompilerMixin):

    REP = 'model'

    def __init__(self, *variables: VariableDefinition):
        super().__init__(variables)

    def reander(self):
        # TODO: implement
        pass
