class StanElement:

    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return StanElement(self.value + ' ' + other.value)

    def var_end(self):
        return StanElement(self.value + ';')

    def format(self, *args):
        return StanElement(self.value.format(*args))
