from typing import List


class StanElement:

    def __init__(self, value: str):
        self.value = value

    def semicolon(self):
        return StanElement(self.value + ';')

    def concat_with_blank(self, other):
        return StanElement(self.value + ' ' + other.value)

    @classmethod
    def join(cls, el_list: List):
        return StanElement(''.join([el.value for el in el_list]))

    def __add__(self, other):
        return StanElement(self.value + other.value)
