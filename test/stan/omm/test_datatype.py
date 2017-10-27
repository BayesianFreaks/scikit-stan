from unittest import TestCase

from skstan.stan.omm import StanDataType
from skstan.stan.omm import StanDataDeclareMixin
from skstan.stan.omm import StanElement

class TestStanDataDeclareMixin(TestCase):

    def assertEqualStanElement(self, first, second):
        self.assertEqual(first.value, second.value)

    def test_declare(self):

        element = StanElement('a')

        data_type = StanDataDeclareMixin()
        data_type.TYPE_NAME = 'vector[{}]'
        declared = data_type._declare(element, 'n')

        expected = StanElement('vector[n] a;')
        self.assertEqualStanElement(declared, expected)
