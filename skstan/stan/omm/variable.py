from skstan.stan.omm.ele import StanElement


class VariableDefinition:

    def __init__(self, variable_name: str):
        self.variable_name_el = StanElement(variable_name)

    def _render(self):
        return self.variable_name_el.value

    def _render_variable(self):
        return self.variable_name_el.var_end().value


class Int(VariableDefinition):

    REP = StanElement('int')

    def __init__(self, variable_name: str):
        super().__init__(variable_name)

    def render(self):
        return self._render_variable(Int.REP + self.variable_name_el)


class Real(VariableDefinition):

    REP = StanElement('real')

    def __init__(self, variable_name: str, lower=None):
        self.lower = lower
        super().__init__(variable_name)

    def render(self):
        if self.lower is not None:
            Real.REP = Real.REP + '<lower={}>'.format(self.lower)
        return self._render_variable(Real.REP + self.variable_name_el)


class Vector(VariableDefinition):

    REP = StanElement('vector[{}]')

    def __init__(self, variable_name: str, dim: int):
        self.dim = dim
        super().__init__(variable_name)

    def render(self):
        rep = Vector.REP.format(self.dim)
        return self._render_variable(rep + self.variable_name_el)


class Matrix(VariableDefinition):

    REP = StanElement('matrix[{}, {}]')

    def __init__(self, variable_name: str, dim1: int, dim2: int):
        self.dim1 = dim1
        self.dim2 = dim2
        super().__init__(variable_name)

    def render(self):
        rep = Matrix.REP.format(self.dim1, self.dim2)
        return self._render_variable(rep + self.variable_name_el)
