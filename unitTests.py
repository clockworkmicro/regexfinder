import unittest

from numpy import isin

from regexfinder import GRAPH, NODE, VECTOR

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
        n = NODE('abcd')
        # self.assertEqual(n.cardinality,1, "Should be 1")
    def test4regex(self):
        n = NODE('[A-Z]')
        self.assertEqual(n.cardinality,26, "Should be 63")
    def test5regex(self):
        n = NODE('\d')
        self.assertEqual(n.cardinality,10, "Should be 63")
    def test6regex(self):
        n = NODE('\w')
        self.assertEqual(n.cardinality,63, "Should be 63")
    def test7regex(self):
        n = NODE('\d{2,4}')
        self.assertEqual(n.cardinality, 11100, "Should be 11100")
        
        
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
       
class GraphTest(unittest.TestCase):
    
    def testGraphRegex(self):
        g1 = GRAPH(regex='[abc]')
        self.assertEqual(g1.regex, '[abc]', "should be abc")
        
    def testGraphType(self):
        g1 = GRAPH(regex='[abcd]')
        self.assertEqual(isinstance(g1.startNode, NODE), True, "Should be true")
        
    def testGraphEmptyList(self):
        g1 = GRAPH(regex='[abcd]')
        # self.assertDictEqual(g1.nodes, {}, "Should be empty")
        g1.simplify()
        self.assertEqual(len(g1.nodes), 1, "Should be 1")
        
    def testSeqPartition(self):
        g1 = GRAPH(regex='[abcd][efg]')
        g1.startNode
        self.assertEqual(g1.startNode.regex, '[abcd][efg]', "Should be same")
        self.assertFalse(g1.startNode.simple, "Should be false")
        g1.simplify()
        self.assertEqual(len(g1.nodes), 2, "Should be 2")
        self.assertEqual(len(g1.sequentialGraphs), 2, "Should be 2")
        self.assertFalse(g1.parallelGraphs, "Should be false")

    def testParaPartition(self):
        g1 = GRAPH(regex='[a]|[d]')
        self.assertEqual(g1.startNode.regex, '[a]|[d]', "Should be same")
        self.assertFalse(g1.startNode.simple, "Should be false")
        g1.simplify()
        self.assertEqual(len(g1.nodes), 2, "Should be 2")
        self.assertFalse(g1.sequentialGraphs, "Should be false")
        self.assertEqual(len(g1.parallelGraphs), 2, "Should be 2")
        
    def testDiamond(self):
        g1 = GRAPH(regex='a(b|c)d')
        self.assertEqual(g1.startNode.regex, 'a(b|c)d', "Should be equal")
        # self.assertEqual(g1.cardinality, 2)
        # self.assertEqual(g1.K, 7)
        g1.simplify()
        self.assertEqual(len(g1.sequentialGraphs), 3)
        self.assertEqual(len(g1.sequentialGraphs[1].parallelGraphs), 2)
        
        
        
        
    # def testPartitionRecursion(self):
    #     g1 = GRAPH(regex='[a]|[b]')
    #     g1.simplify()
    #     self.assertEqual(g1.parallelPartition, False, "Should be false")
        
        
        
        
        
    
    
    
        
    

if __name__ == '__main__':
    unittest.main()
    