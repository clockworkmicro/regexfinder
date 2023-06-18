import unittest
from regexfinder import EDGE, GRAPH, NODE, VECTOR


class NodeTest(unittest.TestCase):
    ################################
    ###     BASIC REGEX TESTS    ###
    ################################
    def testVerySimple(self):
        n = NODE('[abc]')
        self.assertEqual(n.regex, '[abc]')
        self.assertEqual(n.cardinality, 3)
        self.assertTrue(n.simple)

    def testBasicOr(self):
        n = NODE('(a|b)')
        self.assertFalse(n.simple)

        ####################################

    ###     REGEX TO VECTOR TESTS    ###
    ####################################
    def testRegexToVectorToRegex1(self):
        n1 = NODE(regex='[\w]')
        n2 = NODE(vector=VECTOR(n1.vector.v))
        self.assertEqual(n1.regex, n2.regex)

    def testRegexToVectorToRegex2(self):
        n1 = NODE(regex='[\d]')
        n2 = NODE(vector=VECTOR(n1.vector.v))
        self.assertEqual(n1.regex, n2.regex)

    def testRegexToVectorToRegex3(self):
        n1 = NODE(regex='[A-Za-z]')
        n2 = NODE(vector=VECTOR(n1.vector.v))
        self.assertEqual(n1.regex, n2.regex)

    #####################################
    ###     NODE CARDINALITY TESTS    ###
    #####################################
    def test3regex(self):
        n = NODE('a')
        self.assertEqual(n.cardinality, 1, "Should be 1")

    def test4regex(self):
        n = NODE('[A-Z]')
        self.assertEqual(n.cardinality, 26, "Should be 63")

    def test5regex(self):
        n = NODE('\d')
        self.assertEqual(n.cardinality, 10, "Should be 63")

    def test6regex(self):
        n = NODE('\w')
        self.assertEqual(n.cardinality, 63, "Should be 63")

    def test7regex(self):
        n = NODE('\d{2,4}')
        self.assertEqual(n.cardinality, 11100, "Should be 11100")

    #####################################
    ###     NODE QUANTIFIER TESTS    ###
    #####################################
    def testQuantifier(self):
        n1 = NODE('\d{3}')
        self.assertEqual(n1.getQuantifier, '{3}')
        n2 = NODE('a')
        self.assertEqual(n2.getQuantifier, None)
        n3 = NODE('ab')
        n4 = NODE('\d{3}(ab)')
        with self.assertRaises(Exception):
            n3.getQuantifier
            n4.getQuantifier

    def testMergeQuantifier(self):
        n1 = NODE('a{4}')
        n2 = NODE('b{3}')
        self.assertEqual((3,4), n1.mergeQuantifiers([n2]))
        n1 = NODE('a{3,4}')
        self.assertEqual((3,4), n1.mergeQuantifiers([n2]))
        n1 = NODE('a{2,5}')
        self.assertEqual((2,5), n1.mergeQuantifiers([n2]))
        n2 = NODE('b{1,6}')
        self.assertEqual((1,6), n1.mergeQuantifiers([n2]))



class VectorTest(unittest.TestCase):

    def testVerySimple(self):
        v = [0] * 128
        indices = [97, 98, 99, 100, 101]
        for index in indices:
            v[index] = 1
        vector = VECTOR(v)
        self.assertEqual(vector.regex, '[a-e]')
        self.assertListEqual(vector.support.tolist(), indices)

    def testAllLowercase(self):
        v = [0] * 128
        indices = list(range(97, 123))
        for index in indices:
            v[index] = 1
        vector = VECTOR(v)
        self.assertEqual(vector.regex, '[a-z]')
        self.assertListEqual(vector.support.tolist(), indices)

    def testAllUppercase(self):
        v = [0] * 128
        indices = list(range(65, 91))
        for index in indices:
            v[index] = 1
        vector = VECTOR(v)
        self.assertEqual(vector.regex, '[A-Z]')
        self.assertListEqual(vector.support.tolist(), indices)

    def testAllUpperLowercase(self):
        v = [0] * 128
        indices = list(range(65, 91)) + list(range(97, 123))
        for index in indices:
            v[index] = 1
        vector = VECTOR(v)
        self.assertEqual(vector.regex, '[A-Za-z]')
        self.assertListEqual(vector.support.tolist(), indices)

    def testAllUpperLowerNumbers(self):
        v = [0] * 128
        indices = list(range(48, 58)) + list(range(65, 91)) + list(range(97, 123))
        for index in indices:
            v[index] = 1
        vector = VECTOR(v)
        self.assertEqual(vector.regex, '[\\dA-Za-z]')
        self.assertListEqual(vector.support.tolist(), indices)

    def testword(self):
        v = [0] * 128
        indices = list(range(48, 58)) + list(range(65, 91)) + [95] + list(range(97, 123))
        for index in indices:
            v[index] = 1
        vector = VECTOR(v)
        self.assertEqual(vector.regex, '[\\w]')
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

    ##################################
    ###     DIAMOND GRAPH TESTS    ###
    ##################################   
    def testDiamond(self):
        g1 = GRAPH(regex='a(b|c)d')
        g1.partition()
        self.assertEqual(g1.startNode.regex, 'a(b|c)d', "Should be equal")
        g2 = g1.sequentialGraphs[1]
        self.assertEqual(2, len(g2.parallelGraphs))
        self.assertEqual(2, g2.cardinality)
        self.assertEqual(1, g1.entropy)
        self.assertEqual(7, g1.K)
        self.assertEqual(8, g1.phi)

    def testDiamondComplex(self):
        g1 = GRAPH(regex='a(b(c(d|e{4}f{5}|g)h{2}i|j)k)(lm|n)')
        g1.partition()
        self.assertEqual(5, len(g1.sequentialGraphs))
        g2 = g1.sequentialGraphs[2]
        self.assertEqual(2, len(g2.parallelGraphs))
        g3 = g2.parallelGraphs[0]
        if not hasattr(g3, 'sequentialGraphs'):
            g3 = g2.parallelGraphs[1]
        self.assertEqual(4, len(g3.sequentialGraphs))
        g4 = g3.sequentialGraphs[1]
        self.assertEqual(3, len(g4.parallelGraphs))
        x = 0
        g5 = g4.parallelGraphs[x]
        while not hasattr(g5, 'sequentialGraphs'):
            x += 1
            g5 = g4.parallelGraphs[x]
        g5.partition()
        self.assertEqual(1, g5.cardinality)

    ##################################################
    ###     CARDINALITY, K, ENTROPY GRAPH TESTS    ###
    ##################################################
    def testSimpleGraphCardinality(self):
        g1 = GRAPH(regex='a')
        g1.partition()
        self.assertEqual(1, g1.cardinality)
        g2 = GRAPH(regex='abc')
        g2.partition()
        self.assertEqual(1, g2.cardinality)
        g3 = GRAPH(regex='[ab]')
        g3.partition()
        self.assertEqual(2, g3.cardinality)
        g4 = GRAPH(regex='a|(b[cd])')
        g4.partition()
        self.assertEqual(3, g4.cardinality)

    def testGraphCardinalty(self):
        g1 = GRAPH(regex='\d(a(c|d{2}|e{3})|(r{2}|\d)v)[yz]')
        g1.partition()
        self.assertEqual(140, g1.cardinality)
        g2 = GRAPH(regex='ab(c(d|e{4}f{5}|g)h{2}i|j)k(lm|n)', alpha=0.5)
        g2.partition()
        self.assertEqual(8, g2.cardinality)
        self.assertEqual(3, g2.entropy)
        self.assertEqual(19.5, g2.phi)

    def testSimpleGraphK(self):
        g1 = GRAPH(regex='a')
        g1.partition()
        self.assertEqual(1, g1.K)
        g2 = GRAPH(regex='abc')
        g2.partition()
        self.assertEqual(3, g2.K)
        g3 = GRAPH(regex='[ab]')
        g3.partition()
        self.assertEqual(4, g3.K)
        g4 = GRAPH(regex='a|b[cd]')
        g4.partition()
        self.assertEqual(7, g4.K)

    def testGraphK(self):
        g1 = GRAPH(regex='\d(a(c|d{2}|e{3})|(r{2}|\d)v)[yz]')
        g1.partition()
        self.assertEqual(31, g1.K)
        g2 = GRAPH(regex='ab(c(d|e{4}f{5}|g)h{2}i|j)k(lm|n)')
        g2.partition()
        self.assertEqual(33, g2.K)

    #################################################
    ###     NODES, DESCENDANTS, CUT SETS TESTS    ###
    #################################################
    def testSharedDescendatSets(self):
        g1 = GRAPH(regex='(a(b|c(d|e)f)g)|h')
        g1.simplify()
        a = g1.getSharedDecendantSets()
        self.assertEqual(len(g1.nodes), 8)
        self.assertEqual(len(a), 2)

    def testGetNodeEqClasses(self):
        g1 = GRAPH(regex='a(e|f)(g|h)(i|j)')
        g1.simplify()
        eqClasses = g1.getNodeEqClasses()
        eqClassesStrings = set([''.join(sorted([g1.nodes[n].regex for n in eqClass])) for eqClass in eqClasses])
        self.assertSetEqual(eqClassesStrings, {'a', 'ef', 'gh', 'ij'})

    def testTestCutSet(self):
        g1 = GRAPH(regex='a(e|f)(g|h)(i|j)')
        g1.simplify()
        eqClasses = g1.getNodeEqClasses()
        self.assertTrue(all([g1.testCutSet(eqClass) for eqClass in eqClasses]), "All equivalence classes are cut sets")

    def testGetNextCutSet(self):
        g1 = GRAPH(regex='a(b|(c(d|e)))(f|g)h')
        g1.simplify()
        keyA = [key for key, node in g1.nodes.items() if node.regex == 'a'][0]
        nextCutSet = g1.getNextCutSet(keyA)
        regexVals = sorted([g1.nodes[id_].regex for id_ in nextCutSet])
        self.assertLessEqual(regexVals, ['f', 'g'], "The next cut set corresponds to f|g")

    def testGetNodesBetweenCutSets(self):
        g1 = GRAPH(regex='a(b|(c(d|e)))(f|g)h')
        g1.simplify()
        firstCutSet = [key for key, node in g1.nodes.items() if node.regex == 'a']
        nextCutSet = g1.getNextCutSet(firstCutSet[0])
        betweenKeys = g1.getNodesBetweenCutSets(firstCutSet, nextCutSet)
        regexVals = sorted([g1.nodes[id_].regex for id_ in betweenKeys])
        self.assertListEqual(regexVals, ['b', 'c', 'd', 'e'], "The next cut set corresponds to f|g")

    def testGetNodeDescendantsList(self):
        g1 = GRAPH(regex='a(bc|de)')
        g1.simplify()
        firstCutSet = [key for key, node in g1.nodes.items() if node.regex == 'a']
        descendants = g1.getNodeDescendantsList(firstCutSet)
        self.assertEqual(len(descendants),4)

    def testGetNodeDescendantsList2(self):
        g1 = GRAPH(regex='a(b|(c(d|e)))(f|g)h')
        g1.simplify()
        firstCutSet = [key for key, node in g1.nodes.items() if node.regex == 'a']
        descendants = g1.getNodeDescendantsList(firstCutSet)
        regexVals = sorted([g1.nodes[id_].regex for id_ in descendants])
        self.assertListEqual(regexVals, ['b', 'c', 'd', 'e', 'f', 'g', 'h'], "The next cut set corresponds to f|g")

    def testSubGraphs1(self):
        g1 = GRAPH(regex='\d(a(c|d{2}|e{3})|(r{2}|\d)v)[yz]')
        g1.simplify()
        firstCutSet = [key for key, node in g1.nodes.items() if node.regex]  # Getting the keys
        descendants = g1.getNodeDescendantsList(firstCutSet)  # Getting descendants of the very first node
        regexVals = sorted([g1.nodes[id_].regex for id_ in
                            descendants])  # Getting the regex of each of the now simplified nodes; sorted lexicographically
        self.assertListEqual(regexVals, ['\\d', 'a', 'c', 'd{2}', 'e{3}', 'r{2}', 'v', 'y', 'z'], "This is wrong")

    ####################################
    ###     GRAPH PARTITION TESTS    ###
    ####################################
    def testPartitionSimple(self):
        g1 = GRAPH(regex='a|b|c')
        g1.partition()
        self.assertIsInstance(g1, GRAPH)
        self.assertEqual(len(g1.parallelGraphs), 3)

    def testPartition(self):
        g1 = GRAPH(regex='(a(b|c(e|d)f)g)|h')
        g1.partition()
        self.assertEqual(len(g1.parallelGraphs), 2)

    def testSequentialPartition1(self):
        g1 = GRAPH(regex='a(bc|de)')
        g1.simplify()
        g1.sequentialPartition()
        self.assertIsNotNone(g1.sequentialGraphs)

    def testSequentialPartition2(self):
        g1 = GRAPH(regex='a(b|(c(d|e)))(f|g)h')
        g1.simplify()
        g1.sequentialPartition()
        self.assertEqual(len(g1.sequentialGraphs), 4)

    def testSeqPartition(self):
        g1 = GRAPH(regex='[abcd][efg]')
        # g1.startNode
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
        self.assertEqual(len(g1.parallelGraphs), 2, "Should be 2")

    def testSequSimple(self):
        g1 = GRAPH(regex="ab")
        g1.partition()
        self.assertEqual(len(g1.sequentialGraphs), 2)
        # self.assertIsInstance(g1.sequentialGraphs[0], GRAPH)

    def testPartitionSingleLetter(self):
        g1 = GRAPH(regex='a')
        g1.partition()
        self.assertIsInstance(g1, GRAPH)

    # write several tests to check recursive partitions, i.e. correct 
    # for example, check that g1 has three sequential graphs, and that the 
    # second of these has two parallel graphs, and that each of these has two 
    # sequential graphs

    # Imitiate testPartition1 for the new regex
    def testPartition1(self):
        g1 = GRAPH(regex='\d(a(c|d{2}|e{3})|(r{2}|\d)v)[yz]')
        g1.partition()
        self.assertEqual(4, len(g1.sequentialGraphs))
        g2 = g1.sequentialGraphs[1]
        self.assertEqual(2, len(g2.parallelGraphs))
        g3 = g2.parallelGraphs[0]
        self.assertEqual(2, len(g3.sequentialGraphs))
        g4 = g2.parallelGraphs[1]
        self.assertEqual(2, len(g4.sequentialGraphs))

    def testPartition2(self):
        g1 = GRAPH(regex='a(b(c(d|e{4}f{5}|g)h{2}i|j)k)(lm|n)')
        g1.partition()
        self.assertEqual(5, len(g1.sequentialGraphs))
        g2 = g1.sequentialGraphs[2]
        self.assertEqual(2, len(g2.parallelGraphs))

        # j|c(...)
        g3 = g2.parallelGraphs[0]
        if not (hasattr(g3, 'sequentialGraphs')):
            g3 = g2.parallelGraphs[1]
        self.assertEqual(4, len(g3.sequentialGraphs))
        g4 = g3.sequentialGraphs[1]
        self.assertEqual(3, len(g4.parallelGraphs))
        x = 0
        g5 = g4.parallelGraphs[x]
        while not hasattr(g5, 'sequentialGraphs'):
            x += 1
            g5 = g4.parallelGraphs[x]
        self.assertEqual(2, len(g5.sequentialGraphs))

        # (n|lm)
        g6 = g1.sequentialGraphs[4]
        self.assertEqual(2, len(g6.parallelGraphs))
        g7 = g6.parallelGraphs[0]
        if not hasattr(g7, 'sequentialGraphs'):
            g7 = g6.parallelGraphs[1]
        self.assertEqual(2, len(g7.sequentialGraphs))

    ####################################
    ###     GRAPH PARTITION TESTS    ###
    ####################################
    def testGraphCopy(self):
        g1 = GRAPH(regex='abc')
        g2 = g1.deepCopy()
        self.assertNotEqual(id(g1), id(g2))
        self.assertNotEqual(g1.startNode, g2.startNode)

    ########################################
    ###     GRAPH MERGE (NODES) TESTS    ###
    ########################################
    def testMergeNodes1(self):
        n1 = NODE('a')
        n2 = NODE('b')
        nodeDict = {n1.id_: n1, n2.id_: n2}
        # Currently we do not allow lonely nodes,
        # nodes with no edges, to be merged with anything
        edge = EDGE(n1.id_, n2.id_)
        g1 = GRAPH(nodes=nodeDict, edges=[edge])
        g1.mergeRelatedNodes([n1.id_, n2.id_], 'sequential')
        self.assertEqual(g1.outRegex, '[ab]{2}')
        # What about regex?

    def testMergeNodes2(self):
        n1 = NODE('[a-c]')
        n2 = NODE('[bfg]')
        nodeDict = {n1.id_: n1, n2.id_: n2}
        edge = EDGE(n1.id_, n2.id_)
        g1 = GRAPH(nodes=nodeDict, edges=[edge])
        g1.mergeRelatedNodes([n1.id_, n2.id_], 'sequential')
        self.assertEqual(g1.outRegex, '[a-cfg]{2}')

    def testMergeNodes3(self):
        n1 = NODE('[aceg{]')
        n2 = NODE('[bdfh}]')
        nodeDict = {n1.id_: n1, n2.id_: n2}
        edge = EDGE(n1.id_, n2.id_)
        g1 = GRAPH(nodes=nodeDict, edges=[edge])
        g1.mergeRelatedNodes([n1.id_, n2.id_], 'sequential')
        self.assertEqual(g1.outRegex, '[a-h{}]{2}')

    ################################
    ###     GRAPH MERGE TESTS    ###
    ################################

    def test1GraphOfMergedNodes(self):
        n1 = NODE('a')
        n2 = NODE('d')
        nodeDict = {n1.id_: n1, n2.id_: n2}
        e = EDGE(n1.id_, n2.id_)

        g1 = GRAPH(nodes=nodeDict, edges=[e])
        # Why does mergeNodes not work when these run?
        # g1.simplify()
        # g1.partition()
        g1.mergeRelatedNodes([n1.id_, n2.id_], "sequential")

        self.assertEqual(g1.outRegex, '[ad]{2}')

    def test2GraphOfMergedNodes(self):
        n1 = NODE('[acegi]')
        n2 = NODE('[bdfhj]')
        nodeDict = {n1.id_: n1, n2.id_: n2}
        e = [EDGE(n1.id_, n2.id_)]

        g1 = GRAPH(nodes=nodeDict, edges=e)
        # g1.simplify()
        # g1.partition()
        g1.mergeRelatedNodes([n1.id_, n2.id_], 'sequential')

        self.assertEqual(g1.outRegex, '[a-j]{2}')


    ###################################
    ###     GRAPH CREATION TESTS    ###
    ###################################
    def testBuildGraph(self):
        n1 = NODE('[abc][def]')
        n2 = NODE('\d')
        n3 = NODE('[A-H]')
        e13 = EDGE(n1.id_, n3.id_)
        e23 = EDGE(n2.id_, n3.id_)
        edgeList = [e13, e23]
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3]])
        print(nodeDict)
        G = GRAPH(nodes=nodeDict, edges=edgeList)
        G.partition()

        G1 = G.sequentialGraphs[0]
        G1.partition()
        G2 = G1.parallelGraphs[0]
        if not (hasattr(G2, 'sequentialGraphs')):
            G2 = G1.parallelGraphs[1]
        G2.partition()

        self.assertEqual(len(G2.sequentialGraphs), 2)

    ###########################################
    ###     GRAPH MERGE QUANTIFIER TESTS    ###
    ###########################################

    def testSimpleSequentialQuantMerge(self):
        n1 = NODE('a{2}')
        n2 = NODE('d{3}')
        nodeDict = {n1.id_: n1, n2.id_: n2}
        e = [EDGE(n1.id_, n2.id_)]

        g1 = GRAPH(nodes=nodeDict, edges=e)
        g1.partition()
        g1.mergeRelatedNodes([n1.id_, n2.id_], "sequential")
        self.assertEqual('[ad]{5}', g1.outRegex)

    def testSimpleParallelQuantMerge(self):
        n1 = NODE('a')
        n2 = NODE('b{2}')
        n3 = NODE('c{3}')
        n4 = NODE('d')
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3, n4]])
        e = [EDGE(n1.id_, n2.id_), EDGE(n1.id_, n3.id_),
             EDGE(n2.id_, n4.id_), EDGE(n3.id_, n4.id_)]

        g1 = GRAPH(nodes=nodeDict, edges=e)
        g1.partition()
        g1.mergeRelatedNodes([n2.id_, n3.id_], "parallel")
        self.assertEqual('a[bc]{2,3}d', g1.outRegex)

    def testMultiQuantifierMerge(self):
        regex = '\d{2}([a-m]{2}[n-z]{2}|[A-Z]{4})\d'
        G = GRAPH(regex=regex)
        G.partition()

        nodeIds1 = []
        for key in G.nodes:
            if G.nodes[key].regex == '[a-m]{2}' or G.nodes[key].regex == '[n-z]{2}':
                nodeIds1.append(key)
        G.mergeRelatedNodes(nodeIds1, 'sequential')
        
        # Since this is parallel it will pass and fail randomly, depending on what is chosen
        self.assertEqual("\d{2}([A-Z]{4}|[a-z]{4})\d", G.outRegex)               

        nodeIds2 = []
        for key in G.nodes:
            # print(key + ", " + G.nodes[key].regex)
            if G.nodes[key].regex == '[A-Z]{4}' or G.nodes[key].regex == '[a-z]{4}':
                nodeIds2.append(key)
        G.mergeRelatedNodes(nodeIds2, 'parallel')
        self.assertEqual("\d{2}[A-Za-z]{4}\d", G.outRegex)


    def testEdgeAddFail(self):
        n1 = NODE('a')
        edge = EDGE(n1.id_, "Nonexisting Key")
        nodeDict = dict([(n1.id_, n1)])

        g1 = GRAPH(nodes=nodeDict)
        with self.assertRaises(Exception):
            g1.addEdge(edge)

    def testFailedEdgeCreation(self):
        with self.assertRaises(Exception):
            EDGE("Same Word", "Same Word")




    ###########################################
    ###     GRAPH MERGE LIMIT TESTS    ###
    ###########################################

    def test1MergeNodes(self):
        n1 = NODE('a')
        n2 = NODE('b')
        n3 = NODE('c')
        e12 = EDGE(n1.id_, n2.id_)
        e13 = EDGE(n1.id_, n3.id_)
        edgeList = [e12, e13]
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3]])
        G = GRAPH(nodes=nodeDict, edges=edgeList)
        # G.partition()

        mergedNode = G.createMergedNodes([n1.id_, n2.id_, n3.id_], "sequential")

        self.assertEqual('[a-c]{3}', mergedNode.regex)


    def test2MergeNodes(self):
        n1 = NODE('a')
        n2 = NODE('b')
        n3 = NODE('c')
        n4 = NODE('d')
        n5 = NODE('e')
        n6 = NODE('f')
        n7 = NODE('g')
        n8 = NODE('h')

        e12 = EDGE(n1.id_, n2.id_)
        e23 = EDGE(n2.id_, n3.id_)
        e24 = EDGE(n2.id_, n4.id_)
        e45 = EDGE(n4.id_, n5.id_)
        e46 = EDGE(n4.id_, n6.id_)
        e37 = EDGE(n3.id_, n7.id_)
        e78 = EDGE(n7.id_, n8.id_)

        edgeList = [e12, e23, e24, e45, e46, e37, e78]
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3, n4, n5, n6, n7, n8]])
        G = GRAPH(nodes=nodeDict, edges=edgeList)
        G.partition()

        self.assertTrue(G.isMergeNodesValid([n1.id_, n2.id_, n3.id_]))

        self.assertFalse(G.isMergeNodesValid([n1.id_, n4.id_]))

        self.assertTrue(G.isMergeNodesValid([n1.id_, n2.id_, n3.id_, n4.id_]))

        self.assertFalse(G.isMergeNodesValid([n2.id_, n5.id_, n6.id_]))

        self.assertTrue(G.isMergeNodesValid([n5.id_, n6.id_, n8.id_]))

        # G.mergeNodes([n1.id_, n2.id_, n3.id_, n4.id_], )

    def testGraphMergedNodes(self):
        n1 = NODE('a')
        n2 = NODE('b')
        n3 = NODE('c')
        e12 = EDGE(n1.id_, n2.id_)
        e23 = EDGE(n2.id_, n3.id_)
        edgeList = [e12, e23]
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3]])
        G = GRAPH(nodes=nodeDict, edges=edgeList)
        G.partition()

        self.assertTrue(G.isMergeNodesValid([n1.id_, n2.id_]))
        self.assertFalse(G.isMergeNodesValid([n1.id_, n3.id_]))

    def testFullGraphMerge(self):
        n1 = NODE('[a1]')
        n2 = NODE('[b2]')
        n3 = NODE('[c3]')
        n4 = NODE('[d4]')
        n5 = NODE('[e5]')
        n6 = NODE('[f6]')
        n7 = NODE('[g7]')
        n8 = NODE('[h8]')

        e12 = EDGE(n1.id_, n2.id_)
        e23 = EDGE(n2.id_, n3.id_)
        e24 = EDGE(n2.id_, n4.id_)
        e45 = EDGE(n4.id_, n5.id_)
        e46 = EDGE(n4.id_, n6.id_)
        e57 = EDGE(n5.id_, n7.id_)
        e67 = EDGE(n6.id_, n7.id_)
        e37 = EDGE(n3.id_, n8.id_)
        e78 = EDGE(n7.id_, n8.id_)

        edgeList = [e12, e23, e24, e45, e46, e37, e57, e67, e78]
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3, n4, n5, n6, n7, n8]])
        G = GRAPH(nodes=nodeDict, edges=edgeList)
        G.simplify()
        # G.createVisual()

        G.mergeNodeIds([n1.id_, n2.id_, n3.id_, n4.id_, n5.id_, n6.id_, n7.id_, n8.id_])
        self.assertEqual("[1-8a-h]{4,6}", G.outRegex)

        n1 = NODE('[a1]')
        n2 = NODE('[b2]')
        n3 = NODE('[c3]')
        n4 = NODE('[d4]')
        n5 = NODE('[e5]')
        n6 = NODE('[f6]')
        n7 = NODE('[g7]')
        n8 = NODE('[h8]')

        e12 = EDGE(n1.id_, n2.id_)
        e23 = EDGE(n2.id_, n3.id_)
        e24 = EDGE(n2.id_, n4.id_)
        e45 = EDGE(n4.id_, n5.id_)
        e46 = EDGE(n4.id_, n6.id_)
        e57 = EDGE(n5.id_, n7.id_)
        e67 = EDGE(n6.id_, n7.id_)
        e37 = EDGE(n3.id_, n8.id_)
        e78 = EDGE(n7.id_, n8.id_)

        edgeList = [e12, e23, e24, e45, e46, e37, e57, e67, e78]
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3, n4, n5, n6, n7, n8]])
        G = GRAPH(nodes=nodeDict, edges=edgeList)
        G.simplify()

        G.mergeNodes([n1, n2, n3, n4, n5, n6, n7, n8])
        self.assertEqual("[1-8a-h]{4,6}", G.outRegex)

    def testMultiMerges(self):
        n1 = NODE('a')
        n2 = NODE('b')
        n3 = NODE('c')
        n4 = NODE('d')
        n5 = NODE('e')
        n6 = NODE('f')
        n7 = NODE('g')
        n8 = NODE('h')

        e12 = EDGE(n1.id_, n2.id_)
        e23 = EDGE(n2.id_, n3.id_)
        e24 = EDGE(n2.id_, n4.id_)
        e45 = EDGE(n4.id_, n5.id_)
        e46 = EDGE(n4.id_, n6.id_)
        e37 = EDGE(n3.id_, n7.id_)
        e78 = EDGE(n7.id_, n8.id_)

        edgeList = [e12, e23, e24, e45, e46, e37, e78]
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3, n4, n5, n6, n7, n8]])
        G = GRAPH(nodes=nodeDict, edges=edgeList)
        G.partition()
        G1 = G.deepCopy()
        
        G.multiMergeNodeIds([[n4.id_, n5.id_, n6.id_], [n3.id_, n7.id_, n8.id_]])
        G.multiMergeNodeIds([[n1.id_, n2.id_]])
        
        with self.assertRaises(Exception):
            G1.multiMergeNodeIds([[n4.id_, n5.id_, n6.id_], [n3.id_, n7.id_, n8.id_, n4.id_]])
            G1.multiMergeNodeIds([[n4.id_, n5.id_, n8.id_], [n3.id_, n7.id_, n8.id_]])
            G1.multiMergeNodeIds([[n4.id_, n5.id_, n6.id_], []])
            G1.multiMergeNodeIds([[]])

    def testAddNodes(self):
        n1 = NODE('a')
        n2 = NODE('b')
        n3 = NODE('c')
        e12 = EDGE(n1.id_, n2.id_)
        e23 = EDGE(n2.id_, n3.id_)
        edgeList = [e12, e23]
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3]])
        G = GRAPH(nodes=nodeDict, edges=edgeList)
        G.partition()
        n4 = NODE('d')
        e34 = EDGE(n3.id_, n4.id_)
        e34List = [e34]

        self.assertIsInstance(e34List, list)

    def testAddLonelyNodeFail1(self):
        n1 = NODE('a')
        n2 = NODE('b')
        n3 = NODE('c')
        e12 = EDGE(n1.id_, n2.id_)
        e23 = EDGE(n2.id_, n3.id_)
        edgeList = [e12, e23]
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3]])
        G = GRAPH(nodes=nodeDict, edges=edgeList)
        G.partition()

        n4 = NODE('d')
        G.addNode(n4)

        self.assertFalse(G.isMergeNodesValid([n1.id_, n4.id_]))

    def testAddLonelyNodeFail2(self):

        n1 = NODE('a')
        n2 = NODE('b')
        n3 = NODE('c')
        n4 = NODE('d')
        n5 = NODE('e')
        n6 = NODE('f')
        n7 = NODE('g')
        n8 = NODE('h')

        e12 = EDGE(n1.id_, n2.id_)
        e23 = EDGE(n2.id_, n3.id_)
        e24 = EDGE(n2.id_, n4.id_)
        e45 = EDGE(n4.id_, n5.id_)
        e46 = EDGE(n4.id_, n6.id_)
        e37 = EDGE(n3.id_, n7.id_)
        e78 = EDGE(n7.id_, n8.id_)

        edgeList = [e12, e23, e24, e45, e46, e37, e78]
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3, n4, n5, n6, n7, n8]])
        G = GRAPH(nodes=nodeDict, edges=edgeList)
        G.simplify()

        n11 = NODE('z')
        G.addNode(n11)

        self.assertFalse(G.isMergeNodesValid([n11.id_, n4.id_]))
        self.assertFalse(G.isMergeNodesValid([n11.id_, n1.id_, n2.id_, n3.id_, n4.id_]))

    def testAddLonelyNodesFail3(self):
        n1 = NODE('a')
        n2 = NODE('b')
        n3 = NODE('c')
        n4 = NODE('d')
        n5 = NODE('e')

        # diamond with node in middle
        e12 = EDGE(n1.id_, n2.id_)
        e14 = EDGE(n1.id_, n4.id_)
        e25 = EDGE(n2.id_, n5.id_)
        e45 = EDGE(n4.id_, n5.id_)

        edgeList = [e12, e14, e25, e45]
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3, n4, n5]])
        G = GRAPH(nodes=nodeDict, edges=edgeList)
        G.simplify()

        self.assertFalse(G.isMergeNodesValid([n1.id_, n3.id_]))
        self.assertFalse(G.isMergeNodesValid([n1.id_, n2.id_, n3.id_]))
        self.assertFalse(G.isMergeNodesValid([n1.id_, n2.id_, n3.id_, n4.id_]))
        self.assertFalse(G.isMergeNodesValid([n1.id_, n2.id_, n3.id_, n4.id_, n5.id_]))
        self.assertTrue(G.isMergeNodesValid([n1.id_, n2.id_, n4.id_, n5.id_]))

    def testNewQuantifierInput(self):
        vec = VECTOR([1])
        n1 = NODE(vector=vec, quantifier=3)
        n2 = NODE(vector=vec, quantifier='3')
        n3 = NODE(vector=vec, quantifier=',3')
        n4 = NODE(vector=vec, quantifier='3,4')

        self.assertEqual('{3}', n1.getQuantifier)
        self.assertEqual('{3}', n2.getQuantifier)
        self.assertEqual('{,3}', n3.getQuantifier)
        self.assertEqual(0, n3.getQuantifierMin)
        self.assertEqual(3, n3.getQuantifierMax)
        self.assertEqual('{3,4}', n4.getQuantifier)
        self.assertEqual(3, n4.getQuantifierMin)
        self.assertEqual(4, n4.getQuantifierMax)

        with self.assertRaises(Exception):
            NODE(vector=[0], quantifier='3,')
            NODE(vector=[0], quantifier='-3')
            NODE(vector=[0], quantifier=-3)
            NODE(vector=[0], quantifier='three')
            NODE(vector=[0], quantifier=',3.0')
            NODE(vector=[0], quantifier='-2,3')
            NODE(vector=[0], quantifier='2,-3')
            NODE(vector=[0], quantifier='-2,-3')
            NODE(vector=[0], quantifier='3,2')
            NODE(vector=[0], quantifier=',')
            NODE(vector=[0], quantifier='a,4')
            NODE(vector=[0], quantifier='4,a')
            NODE(vector=[0], quantifier='4a6')
            NODE(vector=[0], quantifier='4aa6')
            NODE(vector=[0], quantifier='a4,6')
            NODE(vector=[0], quantifier='4a,6')
            NODE(vector=[0], quantifier='4,a6')
            NODE(vector=[0], quantifier='4,6a')
            NODE(vector=[0], quantifier='')
            NODE(vector=[0], quantifier=None)
            
    def testMultiGraphs(self):
        g1 = GRAPH(regex='a')
        g1.partition()
        self.assertFalse(g1.multiGraphsExist)
        g2 = GRAPH(regex='a|b')
        g2.partition()
        self.assertTrue(g2.multiGraphsExist)
        
        g3 = GRAPH(regex='a(b|c)d')
        g3.partition()
        self.assertFalse(g3.multiGraphsExist)
        
        g4 = GRAPH(regex='([ab][ab])|(c[abd])|([abc]d)')
        g4.partition()
        self.assertTrue(g4.multiGraphsExist)
        
        
    def testStartNode(self):
        n1 = NODE(regex='a')
        nDict = {n1.id_:n1}
        g1 = GRAPH(nodes=nDict)
        g1.partition()
        self.assertEqual(n1.id_, g1.getStartNodeId())
        
        n1 = NODE(regex='a')
        n2 = NODE(regex='b')
        nDict = {n1.id_:n1, n2.id_:n2}
        g1 = GRAPH(nodes=nDict)
        g1.partition()
        self.assertEqual([n1.id_, n2.id_], g1.getStartNodeId())
        
        n1 = NODE(regex='a')
        n2 = NODE(regex='b')
        nDict = {n1.id_:n1, n2.id_:n2}
        e = [EDGE(n1.id_, n2.id_)]
        g1 = GRAPH(nodes=nDict, edges=e)
        g1.partition()
        self.assertEqual(n1.id_, g1.getStartNodeId())
        
        n1 = NODE('a')
        n2 = NODE('b')
        n3 = NODE('c')
        n4 = NODE('d')
        n5 = NODE('e')
        n6 = NODE('f')
        n7 = NODE('g')
        n8 = NODE('h')

        e12 = EDGE(n1.id_, n2.id_)
        e23 = EDGE(n2.id_, n3.id_)
        e24 = EDGE(n2.id_, n4.id_)
        e45 = EDGE(n4.id_, n5.id_)
        e46 = EDGE(n4.id_, n6.id_)
        e37 = EDGE(n3.id_, n7.id_)
        e78 = EDGE(n7.id_, n8.id_)

        edgeList = [e12, e23, e24, e45, e46, e37, e78]
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3, n4, n5, n6, n7, n8]])
        G = GRAPH(nodes=nodeDict, edges=edgeList)
        G.partition()
        self.assertEqual(n1.id_, G.getStartNodeId())
        
    def testLog2Cardinality(self):
        g1 = GRAPH(regex='a(b(c(d|e{4}f{5}|g)h{2}i|j)k)(lm|n)')
        g2 = GRAPH(regex='q([abcdefg]{2}[as]|\d{2})[ab]{2}([cd]{2}|[rs]{2}[vw])#{2}[xyz]x[@#$]\dx([abc]{3}|\w{2}\d{2}\w{3})z')
        g1.partition()
        g2.partition()
        
        self.assertEqual(8, g1.cardinality)
        self.assertEqual(3, g1.log2Cardinality)
        
        self.assertEqual(36.0, g1.phi)

        self.assertEqual(84889052165142720, g2.cardinality)
        self.assertEqual(56.2364, g2.log2Cardinality)
        
        self.assertEqual(154.2364, g2.phi)
        
        
        
        
    def testNgraphCheck(self):
        n1 = NODE('a')
        n2 = NODE('b')
        n3 = NODE('c')
        n4 = NODE('d')
        n5 = NODE('e')
        
        gE1 = EDGE(n1.id_, n2.id_)
        gE2 = EDGE(n3.id_, n2.id_)
        gE3 = EDGE(n3.id_, n4.id_)
        gE4 = EDGE(n5.id_, n4.id_)
        
        edgeList = [gE1, gE2, gE3, gE4]
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3, n4, n5]])
        G = GRAPH(nodes=nodeDict, edges=edgeList)
        self.assertTrue(G.doesNgraphExist)
        with self.assertRaises(Exception):
            G.partition()

        
        n1 = NODE('a')
        n2 = NODE('b')
        n3 = NODE('c')
        n4 = NODE('d')
        n5 = NODE('e')
        n6 = NODE('f')
        n7 = NODE('g')
        n8 = NODE('h')

        e12 = EDGE(n1.id_, n2.id_)
        e23 = EDGE(n2.id_, n3.id_)
        e24 = EDGE(n2.id_, n4.id_)
        e45 = EDGE(n4.id_, n5.id_)
        e46 = EDGE(n4.id_, n6.id_)
        e37 = EDGE(n3.id_, n7.id_)
        e78 = EDGE(n7.id_, n8.id_)

        edgeList = [e12, e23, e24, e45, e46, e37, e78]
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3, n4, n5, n6, n7, n8]])
        G = GRAPH(nodes=nodeDict, edges=edgeList)

        self.assertFalse(G.doesNgraphExist)
        
        
    def testStraighShotNgraph(self):
        
        n1 = NODE('a')
        n2 = NODE('b')
        n3 = NODE('c')
        n4 = NODE('d')
        n5 = NODE('e')
        
        gE1 = EDGE(n1.id_, n2.id_)
        gE2 = EDGE(n3.id_, n2.id_)
        gE3 = EDGE(n3.id_, n4.id_)
        gE4 = EDGE(n5.id_, n4.id_)
        gE5 = EDGE(n1.id_, n3.id_)
        
        edgeList = [gE1, gE2, gE3, gE4, gE5]
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3, n4, n5]])
        G = GRAPH(nodes=nodeDict, edges=edgeList)
        self.assertTrue(G.doesNgraphExist)
        with self.assertRaises(Exception):
            G.partition()

    def testNgraphMerge(self):
        
        n1 = NODE('a')
        n2 = NODE('b')
        n3 = NODE('c')
        n4 = NODE('d')
        n5 = NODE('e')
        
        e1 = EDGE(n1.id_, n2.id_)
        e2 = EDGE(n1.id_, n3.id_)
        e3 = EDGE(n4.id_, n5.id_)
        
        edgeList = [e1, e2, e3]
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3, n4, n5]])
        G = GRAPH(nodes=nodeDict, edges=edgeList)
    
        self.assertTrue(G.willNgraphAppear([n3.id_, n5.id_]))
        
        n1 = NODE('a')
        n2 = NODE('b')
        n3 = NODE('c')
        n4 = NODE('d')
        n5 = NODE('e')
        n6 = NODE('f')
        n7 = NODE('g')
        n8 = NODE('h')

        e12 = EDGE(n1.id_, n2.id_)
        e23 = EDGE(n2.id_, n3.id_)
        e24 = EDGE(n2.id_, n4.id_)
        e45 = EDGE(n4.id_, n5.id_)
        e46 = EDGE(n4.id_, n6.id_)
        e37 = EDGE(n3.id_, n7.id_)
        e78 = EDGE(n7.id_, n8.id_)

        edgeList = [e12, e23, e24, e45, e46, e37, e78]
        nodeDict = dict([(n.id_, n) for n in [n1, n2, n3, n4, n5, n6, n7, n8]])
        G = GRAPH(nodes=nodeDict, edges=edgeList)
        
        self.assertFalse(G.willNgraphAppear([n1.id_, n2.id_]))
        





if __name__ == '__main__':
    unittest.main()
