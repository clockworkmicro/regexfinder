import unittest


class TestSum(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_sum_tuple(self):
        self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")

class Test2(unittest.TestCase):

    def testLength(self):
        self.assertEqual(len('abc'), 3, "Should be 3")

    def testStart(self):
        self.assertEqual('abcdef'[:3], 'abc')


if __name__ == '__main__':
    unittest.main()