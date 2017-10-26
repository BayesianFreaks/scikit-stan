class Element:

    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        self.value + ' ' + other.value

    def var_end(self):
        self.value = self.value + ';'
        return self

    def format(self, *args):
        self.value = self.value.format(args)
        return self
