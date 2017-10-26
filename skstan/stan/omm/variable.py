
class VariableDefinition:

    def __init__(self, variable_name: str):
        self.variable_name = variable_name


class Int(VariableDefinition):

    def __init__(self, variable_name):
        super().__init__(variable_name)


class Real(VariableDefinition):

    def __init__(self, variable_name):
        super().__init__(variable_name)


class Vector(VariableDefinition):

    def __init__(self, variable_name):
        super().__init__(variable_name)


class Matrix(VariableDefinition):

    def __init__(self, variable_name):
        super().__init__(variable_name)
