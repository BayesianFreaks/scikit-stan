from unittest import TestCase

from skstan.stan.omm import Int, Vector, Matrix
from skstan.stan.omm import StanDataDeclareMixin
from skstan.stan.omm import StanElement


class TestStanDataDeclareMixin(TestCase):

    def assertEqualStanElement(self, first, second):
        self.assertEqual(first.value, second.value)

    def test_declare(self):
        """
        Check that stan `declare method` declares vector variable.
        """
        element = StanElement('a')

        data_type = StanDataDeclareMixin()
        data_type.TYPE_NAME = 'vector[{}]'
        declared = data_type._declare(element, 'n')

        expected = StanElement('vector[n] a;')
        self.assertEqualStanElement(declared, expected)


class TestInt(TestCase):

    def assertEqualStanElement(self, first, second):
        self.assertEqual(first.value, second.value)

    def test_int_declare(self):
        """
        Check `declare` method of Int data type declare variable.
        """
        int_type = Int('a')
        declared = int_type.declare()

        expected = StanElement('int a;')
        self.assertEqualStanElement(declared, expected)


class TestVector(TestCase):

    def assertEqualStanElement(self, first, second):
        self.assertEqual(first.value, second.value)

    def test_vector_declare(self):
        """
        Check `declare` method of Vector data type declare variable.
        """
        dim = Int('n')
        vector_type = Vector('a', dim)
        declared = vector_type.declare()

        expected = StanElement('int n;vector[n] a;')
        self.assertEqualStanElement(declared, expected)


class TestMatrix(TestCase):

    def assertEqualStanElement(self, first, second):
        self.assertEqual(first.value, second.value)

    def test_matrix_declare(self):
        """
        Check `declare` method of Matrix data type declare variable.
        """
        dim1 = Int('m')
        dim2 = Int('n')
        matrix_type = Matrix('a', dim1, dim2)
        declared = matrix_type.declare()

        expected = StanElement('int m;int n;matrix[m, n] a;')
        self.assertEqualStanElement(declared, expected)
