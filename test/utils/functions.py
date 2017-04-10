from unittest import TestCase

import skstan.utils.functions


class TestFunctions(TestCase):
    def setUp(self):
        pass

    def test_hello(self):
        self.assertEqual('hello', skstan.utils.functions.get_hello())
