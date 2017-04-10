from unittest import TestCase

import pir.utils.functions


class TestFunctions(TestCase):
    def setUp(self):
        pass

    def test_hello(self):
        self.assertEqual('hello', pir.utils.functions.get_hello())
