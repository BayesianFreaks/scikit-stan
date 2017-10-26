from unittest import TestCase

from skstan.stan.omm import VariableDefinition


class TestVariableDefinition(TestCase):

    def test_render(self):
        """
        Check that render variable name.
        """
        variable_name = 'variable_name'

        vd = VariableDefinition(variable_name)
        value = vd._render()

        expected = variable_name
        self.assertEqual(value, expected)

    def test_render_variable(self):
        """
        Check that render variable name and add semicolon
        """
        variable_name = 'variable_name'

        vd = VariableDefinition(variable_name)
        value = vd._render_variable()

        expected = 'variable_name;'
        self.assertEqual(value, expected)
