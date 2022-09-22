import unittest

from regexfinder import NODE, VECTOR

class NodeTest(unittest.TestCase):

    
    def testVerySimple(self):
        n = NODE('[abc]')
        self.assertEqual(n.regex,'[abc]')
        self.assertEqual(n.cardinality,3)
        self.assertTrue(n.simple)
                
    def testBasicOr(self):
         print(1)
         n = NODE('(a|b)')
         self.assertFalse(n.simple)       

    def testRegexToVectorToRegex(self):
        print(2)
        n1 = NODE(regex='\w')
        n2 = NODE(vector=VECTOR(n1.vector.v))
        self.assertEqual(n1.regex,n2.regex)


        
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