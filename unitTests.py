import unittest

from regexfinder import NODE, VECTOR

class NodeTest(unittest.TestCase):

    
    def testVerySimple(self):
        n = NODE('[abc]')
        self.assertEqual(n.regex,'[abc]')
        self.assertEqual(n.cardinality,3)
        self.assertTrue(n.simple)
                
    def testBasicOr(self):
         n = NODE('(a|b)')
         self.assertFalse(n.simple)       
        
class VectorTest(unittest.TestCase):

    
    def testVerySimple(self):
       v = [0]*128
       indices = [97,98,99,100,101]
       for index in indices:
           v[index] = 1
       vector = VECTOR(v) 
       self.assertEqual(vector.regex,'[a-e]')
       self.assertListEqual(vector.support.tolist(), indices)


if __name__ == '__main__':
    unittest.main()