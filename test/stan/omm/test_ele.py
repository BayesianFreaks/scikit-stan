from unittest import TestCase
from skstan.stan.omm import StanElement


class TestStanElement(TestCase):

    def assertElementEqual(self, element, expected):
        self.assertEqual(element.value, expected)

    def test_add(self):
        """
        Check sum of StanElement objects.
        """
        ele1 = StanElement('value1')
        ele2 = StanElement('value2')

        ele3 = ele1 + ele2

        expected = 'value1 value2'
        self.assertElementEqual(ele3, expected)

    def test_var_end(self):
        """
        Check that semicolon is added at the end of line.
        """
        ele = StanElement('value')

        expected = 'value;'
        self.assertElementEqual(ele.var_end(), expected)

    def test_format(self):
        """
        Check that formating like string.
        """
        ele = StanElement('matrix[{}, {}]')
        expected = 'matrix[3, 4]'
        self.assertElementEqual(ele.format(3, 4), expected)
