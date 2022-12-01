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
        
    
    def test3regex(self):
        n = NODE('a')
        self.assertEqual(n.cardinality,1, "Should be 1")
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
        g1.simplify()
        self.assertEqual(len(g1.nodes), 1, "Should be 1")
        
    def testSeqPartition(self):
        g1 = GRAPH(regex='[abcd][efg]')
        g1.startNode
        self.assertEqual(g1.startNode.regex, '[abcd][efg]', "Should be same")
        self.assertFalse(g1.startNode.simple, "Should be false")
        g1.partition()
        self.assertEqual(len(g1.nodes), 2, "Should be 2")
        self.assertEqual(len(g1.sequentialGraphs), 2, "Should be 2")
        self.assertFalse(g1.parallelGraphs, "Should be false")

    def testParaPartition(self):
        g1 = GRAPH(regex='[a]|[d]')
        self.assertEqual(g1.startNode.regex, '[a]|[d]', "Should be same")
        self.assertFalse(g1.startNode.simple, "Should be false")
        g1.partition()
        self.assertEqual(len(g1.nodes), 2, "Should be 2")
        self.assertEqual(len(g1.parallelGraphs)    , 2, "Should be 2")
        
    def testSequSimple(self):
        g1 = GRAPH(regex="ab")
        g1.partition()
        self.assertEqual(len(g1.sequentialGraphs), 2)
        #self.assertIsInstance(g1.sequentialGraphs[0], GRAPH)
        
    def testPartitionSingleLetter(self):
        g1 = GRAPH(regex='a')
        g1.partition()
        self.assertIsInstance(g1, GRAPH)
        
    def testDiamond(self):
        g1 = GRAPH(regex='a(b|c)d')
        g1.partition()
        self.assertEqual(g1.startNode.regex, 'a(b|c)d', "Should be equal")
        g2 = g1.sequentialGraphs[1]
        self.assertEqual(2, len(g2.parallelGraphs))
        self.assertEqual(2, g2.cardinality)
        
    def testDiamondComplex(self):
        g1 = GRAPH(regex='a(b(c(d|e{4}f{5}|g)h{2}i|j)k)(lm|n)')
        g1.partition()
        self.assertEqual(5, len(g1.sequentialGraphs))
        g2 = g1.sequentialGraphs[2]
        g2.partition()
        self.assertEqual(2, len(g2.parallelGraphs))
        g3 = g2.parallelGraphs[0]
        if not hasattr(g3, 'sequentialGraphs'):
            g3 = g2.parallelGraphs[1]
        g3.partition()
        self.assertEqual(4, len(g3.sequentialGraphs))
        g4 = g3.sequentialGraphs[1]
        g4.partition()
        self.assertEqual(3, len(g4.parallelGraphs))
        x = 0
        g5 = g4.parallelGraphs[x]
        while not hasattr(g5, 'sequentialGraphs'):
            x+=1
            g5 = g4.parallelGraphs[x]
        g5.partition()
        self.assertEqual(2, g5.cardinality)
    
    def testSharedDescendatSets(self):
        g1 = GRAPH(regex='(a(b|c(d|e)f)g)|h')
        g1.simplify()
        a = g1.getSharedDecendantSets()
        self.assertEqual(len(g1.nodes), 8)
        self.assertEqual(len(a), 2)
        
    def testPartitionSimple(self):
        g1 = GRAPH(regex='a|b|c')
        g1.partition()
        self.assertIsInstance(g1, GRAPH)
        self.assertEqual(len(g1.parallelGraphs), 3)
        
    def testPartition(self):
        g1 = GRAPH(regex='(a(b|c(e|d)f)g)|h')
        g1.partition()
        self.assertEqual(len(g1.parallelGraphs), 2)
        
    def testGetNodeEqClasses(self):
        g1 = GRAPH(regex='a(e|f)(g|h)(i|j)')
        g1.simplify()
        eqClasses = g1.getNodeEqClasses()
        eqClassesStrings = set([''.join(sorted([g1.nodes[n].regex for n in eqClass])) for eqClass in eqClasses])
        self.assertSetEqual(eqClassesStrings, set(['a', 'ef', 'gh', 'ij']))
    
    def testTestCutSet(self):
        g1 = GRAPH(regex='a(e|f)(g|h)(i|j)')
        g1.simplify()
        eqClasses = g1.getNodeEqClasses()  
        self.assertTrue(all([g1.testCutSet(eqClass) for eqClass in eqClasses]),"All equivalence classes are cut sets")

    def testGetNextCutSet(self):
        g1 = GRAPH(regex='a(b|(c(d|e)))(f|g)h')
        g1.simplify()
        keyA = [key for key,node in g1.nodes.items() if node.regex == 'a'][0]
        nextCutSet = g1.getNextCutSet(keyA)
        regexVals = sorted([g1.nodes[id_].regex for id_ in nextCutSet])
        self.assertLessEqual(regexVals,['f','g'],"The next cut set corresponds to f|g")
        
    def testGetNodesBetweenCutSets(self):
        g1 = GRAPH(regex='a(b|(c(d|e)))(f|g)h')
        g1.simplify()
        firstCutSet = [key for key,node in g1.nodes.items() if node.regex == 'a']
        nextCutSet = g1.getNextCutSet(firstCutSet[0])
        betweenKeys = g1.getNodesBetweenCutSets(firstCutSet, nextCutSet)
        regexVals = sorted([g1.nodes[id_].regex for id_ in betweenKeys])
        self.assertListEqual(regexVals,['b','c','d','e'],"The next cut set corresponds to f|g")
        
    def testGetNodeDescendantsList(self):
        g1 = GRAPH(regex='a(b|(c(d|e)))(f|g)h')
        g1.simplify()
        firstCutSet = [key for key,node in g1.nodes.items() if node.regex == 'a']
        descendants = g1.getNodeDescendantsList(firstCutSet)
        regexVals = sorted([g1.nodes[id_].regex for id_ in descendants])
        self.assertListEqual(regexVals,['b','c','d','e','f','g','h'],"The next cut set corresponds to f|g")
        
    def testSequentialPartition(self):
        g1 = GRAPH(regex='a(b|(c(d|e)))(f|g)h')
        g1.simplify()
        g1.sequentialPartition()
        self.assertEqual(len(g1.sequentialGraphs),4)
        
    def testSubGraphs1(self):
        g1 = GRAPH(regex='\d(a(c|d{2}|e{3})|(r{2}|\d)v)[yz]')
        g1.simplify()
        firstCutSet = [key for key,node in g1.nodes.items() if node.regex] # Getting the keys
        descendants = g1.getNodeDescendantsList(firstCutSet) # Getting descendants of the very first node
        regexVals = sorted([g1.nodes[id_].regex for id_ in descendants]) # Getting the regex of each of the now simplified nodes; sorted lexicographically
        self.assertListEqual(regexVals, ['[yz]', '\\d', 'a', 'c', 'd{2}', 'e{3}', 'r{2}', 'v'], "This is wrong")

    # write several tests to check recursive partitions, i.e. correct 
    # for example, check that g1 has three sequential graphs, and that the 
    # second of these has two parallel graphs, and that each of these has two 
    # sequential graphs

    # Imitiate testPartition1 for the new regex

    def testPartition1(self):
        g1 = GRAPH(regex='\d(a(c|d{2}|e{3})|(r{2}|\d)v)[yz]')
        g1.partition()
        self.assertEqual(3, len(g1.sequentialGraphs))
        g2 = g1.sequentialGraphs[1]
        g2.partition()
        self.assertEqual(2, len(g2.parallelGraphs))
        g3 = g2.parallelGraphs[0]
        g3.partition()
        self.assertEqual(2, len(g3.sequentialGraphs))
        g4 = g2.parallelGraphs[1]
        self.assertEqual(2, len(g4.sequentialGraphs))

    def testPartition2(self):
        g1 = GRAPH(regex='a(b(c(d|e{4}f{5}|g)h{2}i|j)k)(lm|n)')
        g1.partition()
        self.assertEqual(5, len(g1.sequentialGraphs))
        g2 = g1.sequentialGraphs[2]
        g2.partition()
        self.assertEqual(2, len(g2.parallelGraphs))
        
        # j|c(...)
        g3 = g2.parallelGraphs[0]
        if not (hasattr(g3, 'sequentialGraphs')):
            g3 = g2.parallelGraphs[1]
        g3.partition()
        self.assertEqual(4, len(g3.sequentialGraphs))
        g4 = g3.sequentialGraphs[1]
        g4.partition()
        self.assertEqual(3, len(g4.parallelGraphs))
        x = 0
        g5 = g4.parallelGraphs[x]
        while not hasattr(g5, 'sequentialGraphs'):
            x+=1
            g5 = g4.parallelGraphs[x]
        g5.partition()
        self.assertEqual(2, len(g5.sequentialGraphs))
        
        # (n|lm)
        g6 = g1.sequentialGraphs[4]
        g6.partition
        self.assertEqual(2, len(g6.parallelGraphs))
        g7 = g6.parallelGraphs[0]
        if not hasattr(g7, 'sequentialGraphs'):
            g7 = g6.parallelGraphs[1]
        g7.partition()
        self.assertEqual(2, len(g7.sequentialGraphs))
 
                   
        
if __name__ == '__main__':
    unittest.main()
    