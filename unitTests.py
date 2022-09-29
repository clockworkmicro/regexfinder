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

    def testRegexToVectorToRegex1(self):
        n1 = NODE(regex='[\w]')
        n2 = NODE(vector=VECTOR(n1.vector.v))
        self.assertEqual(n1.regex,n2.regex)

    def testRegexToVectorToRegex2(self):
        n1 = NODE(regex='[\d]')
        n2 = NODE(vector=VECTOR(n1.vector.v))
        self.assertEqual(n1.regex,n2.regex)

    def testRegexToVectorToRegex3(self):
        n1 = NODE(regex='[A-Za-z]')
        n2 = NODE(vector=VECTOR(n1.vector.v))
        self.assertEqual(n1.regex,n2.regex)
        
    
    # def test2regex(self):
    #     n = NODE(' ')
    #     self.assertEqual(n.cardinality,63, "Should be 63")
    def test3regex(self):
        n = NODE('[a-z]')
        self.assertEqual(n.cardinality,26, "Should be 26")
    def test4regex(self):
        n = NODE('[A-Z]')
        self.assertEqual(n.cardinality,26, "Should be 63")
    def test5regex(self):
        n = NODE('\d')
        self.assertEqual(n.cardinality,10, "Should be 63")
    def test6regex(self):
        n = NODE('\w')
        self.assertEqual(n.cardinality,63, "Should be 63")
        
        
class VectorTest(unittest.TestCase):
    
    def testVerySimple(self):
       v = [0]*128
       indices = [97,98,99,100,101]
       for index in indices:
           v[index] = 1
       vector = VECTOR(v) 
       self.assertEqual(vector.regex,'[a-e]')
       self.assertListEqual(vector.support.tolist(), indices)

    def testAllLowercase(self):
       v = [0]*128
       indices = list(range(97,123))
       for index in indices:
           v[index] = 1
       vector = VECTOR(v) 
       self.assertEqual(vector.regex,'[a-z]')
       self.assertListEqual(vector.support.tolist(), indices)

    def testAllUppercase(self):
       v = [0]*128
       indices = list(range(65,91))
       for index in indices:
           v[index] = 1
       vector = VECTOR(v) 
       self.assertEqual(vector.regex,'[A-Z]')
       self.assertListEqual(vector.support.tolist(), indices)

    def testAllUpperLowercase(self):
       v = [0]*128
       indices = list(range(65,91)) + list(range(97,123)) 
       for index in indices:
           v[index] = 1
       vector = VECTOR(v) 
       self.assertEqual(vector.regex,'[A-Za-z]')
       self.assertListEqual(vector.support.tolist(), indices)

    def testAllUpperLowerNumbers(self):
       v = [0]*128
       indices = list(range(48,58)) +list(range(65,91)) + list(range(97,123))
       for index in indices:
           v[index] = 1
       vector = VECTOR(v) 
       self.assertEqual(vector.regex,'[\\dA-Za-z]')
       self.assertListEqual(vector.support.tolist(), indices)

    def testword(self):
       v = [0]*128
       indices = list(range(48,58)) +list(range(65,91)) + [95] + list(range(97,123))
       for index in indices:
           v[index] = 1
       vector = VECTOR(v) 
       self.assertEqual(vector.regex,'[\\w]')
       self.assertListEqual(vector.support.tolist(), indices)

if __name__ == '__main__':
    unittest.main()
    