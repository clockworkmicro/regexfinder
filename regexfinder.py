import copy
import numpy as np
import pandas as pd
from graphviz import Digraph
from itertools import combinations
import more_itertools as mit
import random

from utils import *


class NODE:
    def __init__(self, regex=None, vector=None, replaced=False, alpha=1, quantifier=None):
        """
        Node creation is defined firstly by a regex. If a regex is provided with a vector and/or quantifier,
        the latter two will be ignored and the node will be created from the given regex. If a regex
        is not given, a vector muse be given. Quantifier is optional, but must have a finite upper bound.
        An empty quantifier/none implies a quantifier of {1}
        """

        if regex is None and vector is None:
            raise Exception('Either regex or vector required.')

        self.alpha = alpha
        self.regex = regex
        self.vector = vector
        self.id_ = next(strCounter)
        self.reduced = False
        self.replaced = replaced

        if self.regex is None:
            # if merging nodes this will make the characters go in alphabetical order
            self.regex = self.vector.regex
            if quantifier is not None:
                regexUpdates = setClassQuantList(self.regex, quantifier)
                self.classQuantList = regexUpdates[0]
                self.regex = regexUpdates[1]
            else:
                self.classQuantList = getClassQuantList(self.regex)
        else:
            self.classQuantList = getClassQuantList(self.regex)

        self.removeOuterParentheses()

        if self.simple:
            self.createVector()
        else:
            pass

    @property
    def simple(self):
        """
        Returns true if a node is simple, false is not. This comes into play when building and
        simplifing/partitioning graph objects.
        """
        return not ('(' in self.regex or '|' in self.regex or len(self.classQuantList) > 1)

    @property
    def getQuantifier(self):
        """
        Returns the quantifier of the given node. Throws Exception if
        the node is not simple. If the quantifier is {1}, None is returned.
        """
        if self.simple:
            return self.classQuantList[0]['quantifier']
        else:
            raise Exception("Node is not simple")

    @property
    def getQuantifierMin(self):
        """
        Returns the lower bound of the given's node quantifier. Throws Exception
        if the node is not simple. If the quantifier is {1}, None is returned.
        """
        if self.simple:
            return self.classQuantList[0]['min']
        else:
            raise Exception("Node is not simple")

    @property
    def getQuantifierMax(self):
        """
        Returns the upper bound of the given's node quantifier. Throws Exception
        if the node is not simple. If the quantifier is {1}, None is returned.
        """
        if self.simple:
            return self.classQuantList[0]['max']
        else:
            raise Exception("Node is not simple")

    def mergeQuantifiers(self, nodeList):
        """
        Returns the new lower and upper bounds following the merging of the nodes' quantifier.
        """
        nodeList.append(self)
        lowestLow = min(lowestMin.getQuantifierMin for lowestMin in nodeList)
        highestHigh = max(highestMax.getQuantifierMax for highestMax in nodeList)

        if lowestLow == highestHigh:
            return lowestLow
        else:
            return (lowestLow, highestHigh)

    @property
    def topOrExists(self):
        """
        CHECK:
        Returns boolean if the node has an or statement seperating the whole regex.
        """
        return topOrExists(self.regex)

    def removeOuterParentheses(self):
        """
        Removes unnecesary outer parenthesis if they exist in a node
        i.e. (a).
        """
        while self.regex[0] == '(' and self.regex[-1] == ')':
            self.regex = self.regex[1:-1]

    def createVector(self):
        """
        Creates a vector for the given node.
        """

        assert self.simple
        reClass = self.classQuantList[0]['class']
        vector = np.zeros(128, dtype=int)

        if len(reClass) == 1:
            vector[ord(reClass)] = 1

        elif reClass == '\d':
            vector[48:58] = 1

        elif reClass == '\D':
            vector[31:47] = 1
            vector[59:127] = 1

        elif reClass == '\w':
            vector[48:58] = 1
            vector[65:91] = 1
            vector[95] = 1
            vector[97:123] = 1

        elif reClass == '\W':
            vector[31:48] = 1
            vector[58:65] = 1
            vector[91:95] = 1
            vector[96] = 1
            vector[123:127] = 1

        elif '[' in reClass:
            pieces = partitionClass(reClass)
            for piece in pieces:
                if '-' in piece:
                    start = ord(piece[0])
                    end = ord(piece[2]) + 1
                    vector[start:end] = 1
                elif piece == '\d':
                    vector[48:58] = 1

                elif piece == '\D':
                    vector[31:47] = 1
                    vector[59:127] = 1

                elif piece == '\w':
                    vector[48:58] = 1
                    vector[65:91] = 1
                    vector[95] = 1
                    vector[97:123] = 1

                elif piece == '\W':
                    vector[31:48] = 1
                    vector[58:65] = 1
                    vector[91:95] = 1
                    vector[96] = 1
                    vector[123:127] = 1
                else:
                    vector[ord(piece)] = 1
        self.vector = VECTOR(vector, self.alpha)

    def reduce(self):
        """
        Reduces the regex of a given simple node if possible, i.e. 'abcde' -> 'a-e'.
        Same as VECTOR method.
        """
        if not self.simple:
            raise Exception('Node is not simple. A node cannot be reduced if it is not simple.')
        elif not self.isPureSimple:
            if not self.vector:
                self.createVector()
            else:
                pass
            self.vector.reduce()
            self.regex = self.vector.regex + (self.getQuantifier if self.getQuantifier else "")
            self.reduced = True

    @property
    def cardinality(self):
        """
        Returns the cardinality, the possible number of strings that satisfy its given regex, of the
        given node.
        """
        if self.vector is None:
            self.createVector()
        valCount = sum(self.vector.v)
        quantifier = self.classQuantList[0]['quantifier']

        if quantifier is None:
            return valCount
        elif quantifier == '?':
            return valCount + 1
        elif self.classQuantList[0]['min'] and self.classQuantList[0]['max']:
            return sum([valCount ** exponent for exponent in
                        range(self.classQuantList[0]['min'], self.classQuantList[0]['max'] + 1)])

        else:
            raise Exception('Check class quantifier')

    @property
    def singleQuantifier(self):
        """
        Returns boolean if the quantifier is the same for its lower bound and upper bound i.e. {5}.
        """
        return self.getQuantifierMin == self.getQuantifierMax

    @property
    def entropy(self):
        """
        Returns the entropy of a node, as defined by Information-Theoretic entropy.
        """
        return round(np.log2(self.cardinality), 4)

    @property
    def K(self):
        """
        Returns the length of given node's regex.
        """
        return len(self.regex)

    @property
    def phi(self):
        """
        Returns the phi of the node, being the entropy of the node added to the product
        of the alpha (weight parameter) and its K value.
        """
        return self.entropy + self.alpha * self.K

    def match(self, inString, boolean=False):
        """
        CHECK
        """
        matches = re.finditer(self.regex, inString)
        if boolean:
            return bool(list(matches))
        else:
            return matches

    @property
    def isPureSimple(self):
        """
        Checks if a node's regex is one 'item' i.e. 'a', \d, 'y' not [ab]
        
        Returns:
            Boolean
        """
        if not self.vector:
            self.createVector()

        if self.vector.minIndex == self.vector.maxIndex:
            return True

        rawRegex = getClassQuantList(self.regex)[0]['class']
        if rawRegex == '\\d' or rawRegex == '\\w':
            return True
        else:
            return False

    @property
    def random(self):
        """
        CHECK
        Returns a random string that satifies a given node's regex.
        """
        if self.vector is None:
            self.createVector()
        quantifier = self.classQuantList[0]['quantifier']
        if quantifier is None:
            return self.vector.random
        else:
            mm = self.classQuantList[0]['min']
            mM = self.classQuantList[0]['max']
        if mm == mM:
            return ''.join([self.vector.random for x in range(mm)])
        else:
            raise Exception('still need to fix min != max')


class EDGE:
    def __init__(self, parent, child, words=None):
        """
        CHECK WORDS
        An edge is what connects 2 nodes, and exactly two nodes.
        Its parent and child must not be the same.
        """
        if words is None:
            words = []
        if parent != child:
            self.parent = parent
            self.child = child
        else:
            raise Exception("Parent and child cannot be the same")
        self.words = words


class VECTOR:
    def __init__(self, vector, alpha=None):
        """
        CHECK ALPHA
        A vector is a matrix of 1's and 0's 128 big, each place
        representing an ASCII character (i.e. a space 33 in represents the '!' character.
        """
        self.v = vector
        self.alpha = alpha

    @property
    def regex(self):
        """
        Returns the regex built from the given vector.
        """
        if not self.consecutiveSublists:
            raise Exception('Vector has no nonzero values.')
        else:
            pass
        toReturn = '['
        for subList in self.consecutiveSublists:
            subList = [chr(x) for x in subList]

            if len(subList) == 1:
                toReturn += subList[0]
            elif len(subList) == 2:
                toReturn += subList[0]
                toReturn += subList[1]

            elif subList[0] == '0' and subList[-1] == '9':
                toReturn += '\d'
            else:
                toReturn += subList[0]
                toReturn += '-'
                toReturn += subList[-1]
        toReturn += ']'
        if toReturn == '[\\dA-Z_a-z]':
            toReturn = '[\\w]'
        return toReturn

    def reduce(self):
        """
        Reduces the regex of a given vector if possible, i.e. 'abcde' -> 'a-e'.
        Same as NODE method.
        """
        flatten = lambda x: [y for z in x for y in z]

        subLists = self.consecutiveSublists

        combs = [list(combinations(subLists, i)) for i in range(1, len(subLists) + 1)]
        combs = flatten(combs)

        for comb in combs:
            support = flatten(comb)
            first = min(support)
            last = max(support)

            temp = VECTOR(self.v.copy(), self.alpha)
            temp.v[first:last] = 1

            if temp.phi < self.phi:
                self.v = temp.v

    @property
    def consecutiveSublists(self):
        """
        CHECK
        """
        return [list(group) for group in mit.consecutive_groups(self.support)]

    @property
    def minIndex(self):
        """
        CHECK
        Returns the first (min) index in the given vector field.
        """
        return np.min(np.where(self.v))

    @property
    def maxIndex(self):
        """
        CHECK
        Returns the last (max) index in the given vector field.
        """
        return np.max(np.where(self.v))

    @property
    def ent(self):
        """
        Returns the Informatic-Theory entropy of the given vector field, much like
        a regex.
        """
        return round(np.log2(np.sum(self.v)), 4)

    @property
    def support(self):
        """
        CHECK
        """
        return np.where(self.v)[0]

    @property
    def K(self):
        """
        Returns the length of the regex built from the given vector.
        """
        return len(self.regex)

    @property
    def phi(self):
        """
        Returns the phi of the vector, being the entropy of the vector added to the product
        of the alpha (weight parameter) and its K value.
        """
        return self.ent + self.alpha * self.K

    @property
    def random(self):
        """
        CHECK
        Returns a random string that satifies a given vector's regex.
        """
        return chr(random.sample(list(np.where(self.v)[0]), 1)[0])


class ALPHABET:
    def __init__(self, alphabetList):
        """
        CHECK
        """
        self.alphabetList = alphabetList

    def sample(self, count=1):
        """
        CHECK
        """
        return ''.join(random.choices(self.alphabetList, k=count))


class WORDCOLLECTION:
    def __init__(self, words):
        """
        CHECK
        """
        self.words = words
        self.maxLength = max([len(word) for word in words])
        self.M = [list(x) for x in self.words]
        self.df = pd.DataFrame([list(word) for word in words])
        self.prefixDicts = []
        self.suffixDicts = []
        self.spClasses = []

        self.setToStr = lambda x: ''.join(sorted(list(x))) if isinstance(x, set) else x
        self.dictValSetToStr = lambda d: dict([(k, self.setToStr(v)) for k, v in d.items()])

        for i in range(self.maxLength):
            prefixes = {}
            suffixes = {}

            if i != 0:
                for v in self.M:
                    if v[i] in prefixes.keys():
                        prefixes[v[i]].add(v[i - 1])
                    else:
                        prefixes[v[i]] = set(v[i - 1])
                prefixes = self.dictValSetToStr(prefixes)
                self.prefixDicts.append(prefixes)

            else:
                pass

            if i != self.maxLength - 1:

                for v in self.M:
                    if v[i] in suffixes.keys():
                        suffixes[v[i]].add(v[i + 1])
                    else:
                        suffixes[v[i]] = set(v[i + 1])

                suffixes = dict([(k, self.setToStr(v)) for k, v in suffixes.items()])
                self.suffixDicts.append(suffixes)

            else:
                pass

    def createClasses(self):
        """
        CHECK
        """
        eq = {}
        # for k in set(list(preClasses.keys()) + list(suffClasses.keys())):
        #    eq[k] = (preClasses.get(k,None),suffClasses.get(k,None))
        for i in range(1, self.maxLength - 1):

            for k in set(list(self.prefixDicts[i].keys()) + list(self.suffixDicts[i].keys())):
                eq[k] = (
                    self.setToStr(self.prefixDicts[i].get(k, None)), self.setToStr(self.suffixDicts[i].get(k, None)))

            # The keys of eq are the alphabet. The values are each a tuple, where the first value is a 
            # string of the prefixes and the second is a string of the suffixes.
            classes = self.partitionValueEquality(eq)

            self.spClasses.append(classes)
        self.spClasses = [self.partitionValueEquality(self.prefixDicts[0])] + self.spClasses
        self.spClasses = self.spClasses + [self.partitionValueEquality(self.suffixDicts[-1])]

    def partitionValueEquality(self, inDict):
        """
        Returns a dictionary grouping the input dictionary keys by same value
        """
        outDict = {}
        for k, v in inDict.items():
            if v in outDict.keys():
                outDict[v].add(k)
            else:
                outDict[v] = set(k)
        outDict = self.dictValSetToStr(outDict)
        return outDict


class GRAPH:
    def __init__(self, regex=None, wordList=None, nodes=None, edges=None, alpha=1):
        """
        CHECK
        A Graph is a structure made of nodes and edges. Graphs are built by priority from regex,
        followed by wordlist and nodes, a dictionary. If a graph is built from a regex, a non-simple node will be 
        made and added to the nodes dictionary. None however are required to instantiate a graph object. 
        Edges are not required to be made with nodes.
        """
        self.regex = regex
        self.wordList = wordList
        self.alpha = alpha

        self.nodes = {}
        self.edges = []
        # self.sequentialGraphs = []
        # self.parallelGraphs = []

        if self.regex:
            try:
                re.compile(self.regex)
            except:
                raise Exception('Invalid argument. re.compile failed')
        elif self.wordList:
            self.wordCollection = WORDCOLLECTION(self.wordList)
            self.wordCollection.createClasses()
            self.columnState = 0
        elif nodes:
            self.nodes = nodes
            if edges:
                for edge in edges:
                    self.addEdge(edge)
        else:
            pass

        if self.regex:
            self.startNode = NODE(self.regex)
            self.nodes.update({self.startNode.id_: self.startNode})
        else:
            pass

        test = self.deepCopy()
        test.parallelPartition()
        if test.parallelGraphs:
            self.topLevelOrExists = True
        else:
            self.topLevelOrExists = False

    def deepCopy(self):
        """
        Returns a content-equal copy of the given graph, with a different
        memory address
        """
        return copy.deepcopy(self)

    def addNode(self, node, edges=None):
        """
        Adds a node into the given graph. Edges not required.
        """
        if edges is None:
            edges = []
        if node.id_ in self.nodes.keys():
            raise Exception('Node key already exists')

        self.nodes[node.id_] = node

        if edges:
            for currEdge in edges:
                self.edges.append(currEdge)
        else:
            pass

    def removeNodeAndEdges(self, nodeList):
        """
        Removes the given nodes, and its edges, from the given graph.
        Returns the affected nodes and edges adjecent to the group of nodes removed.
        """
        upperAffectedNodes = []
        lowerAffectedNodes = []
        edgesToRemove = []

        # Future, could be more efficient by not going through entire edgeList
        for edge in self.edges:
            if edge.parent in nodeList or edge.child in nodeList:
                edgesToRemove.append(edge)

            if edge.child in nodeList and edge.parent not in nodeList:
                if edge.parent not in upperAffectedNodes:
                    upperAffectedNodes.append(edge.parent)
            elif edge.parent in nodeList and edge.child not in nodeList:
                if edge.child not in lowerAffectedNodes:
                    lowerAffectedNodes.append(edge.child)

        [self.edges.remove(x) for x in edgesToRemove]

        for node in nodeList:
            del self.nodes[node]

        return [upperAffectedNodes, lowerAffectedNodes]

    def addEdge(self, edge):
        """
        Adds the given edge to the graph ONLY if the parent and child nodes do not already
        exist and if an edge for said parent and child nodes does not already exist.
        """
        for gEdge in self.edges:
            if edge == gEdge:
                raise Exception("Edge already exists")
        if edge.parent not in self.nodes.keys() or edge.child not in self.nodes.keys():
            raise Exception("Parent or Child node does not exist")
        self.edges.append(edge)

    def getEdge(self, parent, child):
        """
        Returns EDGE object for a given parent and child node, if it exists. Returns
        false otherwise.
        """
        matches = [edge for edge in self.edges if (edge.parent == parent and edge.child == child)]
        if len(matches) > 1:
            raise Exception('More than one edge with same parent/child.')
        elif len(matches) == 1:
            return matches[0]
        else:
            return False

    def removeEdge(self, parent, child):
        """
        Removes edge binding two nodes (parent and child) IF they exist in the given graph.
        """
        if parent not in self.nodes.keys() or child not in self.nodes.keys():
            raise Exception("Parent or Child node does not exist")
        toRemove = [edge for edge in self.edges if (edge.parent == parent and edge.child == child)]

        if toRemove:
            [self.edges.remove(edge) for edge in toRemove]

    def removeEdgesList(self, edgeList):
        """
        Removes all edges in the given graph.
        """
        [self.edges.remove(edge) for edge in edgeList]

    def getParents(self, id_):
        """
        Returns node ID(s) of the parent(s) of a given node via ID. Returns false if nonexistent.
        """
        return [x.parent for x in self.edges if x.child == id_]

    def getChildren(self, id_):
        """
        Returns node ID(s) of the child(ren) of a given node via ID. Returns false if nonexistent.
        """
        return [x.child for x in self.edges if x.parent == id_]

    def getSiblings(self, id_):
        """
        CHECK
        Returns neighboring nodes, siblings of a given node via ID.
        """
        if not self.getParents(id_):
            return self.getNodesNoParents()
        else:
            return sorted(list(set(flatten([self.getChildren(parent) for parent in self.getParents(id_)]))))

    def getNodesNoChildren(self):
        """
        Returns node IDs for all nodes with no children (nodes that extend to any given node).
        """
        return [x.id_ for x in self.nodes.values() if not self.getChildren(x.id_)]

    def getNodesNoParents(self):
        """
        Returns node IDs for all nodes with no parents (nodes that do not extend from any node).
        """
        return [x.id_ for x in self.nodes.values() if not self.getParents(x.id_)]

    def getLonelyNodes(self):
        """
        Returns all nodes with no edges.
        """
        setOfNoParents = set(self.getNodesNoParents())
        setOfNoChildren = set(self.getNodesNoChildren())

        return list(setOfNoChildren.intersection(setOfNoParents))

    def getNotSimple(self):
        """
        CHECK
        Returns all nodes that are not simple.
        """
        return [id_ for id_, node in self.nodes.items() if (not node.replaced and not node.simple)]

    def simplify(self):
        """
        CHECK
        Simplifies all existing nodes.
        """
        while self.getNotSimple():
            self.process(self.getNotSimple()[0])
        self.nodes = dict([(name, node) for name, node in self.nodes.items() if node.simple])

    def getNodeEqClasses(self):
        """
        CHECK
        Returns a list of node IDs that have the same parents and children.
        """
        # two nodes are equivalent if they have the same parents and children
        tempDict = {}
        for id_ in self.nodes.keys():
            parents = tuple(sorted(self.getParents(id_)))
            children = tuple(sorted(self.getChildren(id_)))
            if (parents, children) in tempDict.keys():
                tempDict[(parents, children)].append(id_)
            else:
                tempDict[(parents, children)] = [id_]
        return list(tempDict.values())

    def getNodeAncestorsList(self, inList):
        """
        Returns a list of node IDs of ancestors, the parents of all of each node's parents
        """
        toReturn = []
        for id_ in inList:
            toReturn += self.getNodeAncestors(id_)
        return toReturn

    def getNodeDescendantsList(self, inList):
        """
        Returns a list of node IDs of descendants, the children of all of each node's children
        """
        if not isinstance(inList, list):
            raise Exception(('value inList must be a list'))
        toReturn = []
        for id_ in inList:
            toReturn += self.getNodeDescendants(id_)
        toReturn = list(set(toReturn))
        return toReturn

    def getNodeAncestors(self, id_):
        """
        Returns a list of node IDs of ancestors, the parents of a given node's parents
        """
        ancestors = []
        parents = self.getParents(id_)
        while parents:
            parent = parents.pop()
            parents += self.getParents(parent)
            ancestors.append(parent)
        return sorted(list(set(ancestors)))

    def getNodeDescendants(self, id_):
        """
        Returns a list of node IDs of descendants, the children of a given node's children
        """
        descendants = self.getChildren(id_)
        nextGen = descendants

        while nextGen:
            parents = nextGen
            nextGen = []
            for parent in parents:
                nextGen += self.getChildren(parent)
            descendants += nextGen
        return descendants

    # A set of nodes is a CutSet if the set of nodes union with all its ancestors
    # and al of its descendants yields the entire graph
    def testCutSet(self, inList):
        """
        CHECK
        """
        if not all([id_ in self.nodes.keys() for id_ in inList]):
            raise Exception('Node included in inSet that is not in self.nodes.keys().')
        else:
            pass

        allAncestors = []
        allDescendants = []
        for id_ in inList:
            allAncestors += self.getNodeAncestors(id_)
            allDescendants += self.getNodeDescendants(id_)
        return set(inList + allAncestors + allDescendants) == set(self.nodes.keys())

    def getCutSets(self):
        """
        CHECK
        """
        eqClasses = self.getNodeEqClasses()
        return [sorted(eqClass) for eqClass in eqClasses if self.testCutSet(eqClass)]

    def getNextCutSet(self, id_):
        """
        CHECK
        """
        if not id_ in self.nodes.keys():
            raise Exception('Node id not not valid.')
        children = self.getChildren(id_)
        while not (children in self.getCutSets()) and children:
            newchildren = self.getChildren(children[0])
            children = newchildren
        return children

    def getNodesBetweenCutSets(self, cutSetUpper, cutSetLower):
        """
        CHECK
        """
        if cutSetUpper not in self.getCutSets() and cutSetUpper != self.getNodesNoParents():
            raise Exception('cutSetUpper is not a cutSet and is not noParents')

        if cutSetLower not in self.getCutSets():
            raise Exception('cutSetLower is not a cutSet')

        if not all([node in self.getNodeDescendants(cutSetUpper[0]) for node in cutSetLower]):
            raise Exception('cutSetUpper is not ancestor of cutSetLower')

        allDescendants = []
        toExclude = cutSetLower.copy()

        for node in cutSetLower:
            toExclude += self.getNodeDescendants(node)

        for node in cutSetUpper:
            allDescendants += self.getNodeDescendants(node)

        return sorted(list(set([x for x in allDescendants if x not in toExclude])))

    def addLayer(self):
        """
        CHECK
        """
        if not self.wordCollection:
            raise Exception('addLayer only used when wordCollection is provided')
        else:
            pass

        newNodes = []
        for n in self.wordCollection.spClasses[self.columnState].values():
            newNodes.append(NODE(self.strToRegex(n)))

        if self.columnState == 0:
            [self.addNode(node) for node in newNodes]

        elif self.columnState == self.wordCollection.maxLength - 1:
            print('All layers added')
            return

        else:
            nodesNoChildren = self.getNodesNoChildren()
            for word in self.wordCollection.words:
                for node in nodesNoChildren:
                    match = node.match(word[0], boolean=True)
                    if match:
                        parentNode = node
                        break
                for node in newNodes:
                    match = node.match(word[1], boolean=True)
                    if match:
                        childNode = node
                        existingEdge = self.getEdge(parentNode.id_, childNode.id_)
                        if not existingEdge:
                            edge = EDGE(parentNode.id_, childNode.id_, words=[word])
                            self.addEdge(edge)
                        else:
                            existingEdge.words.append(word)
                        break

            [self.addNode(node) for node in newNodes]

        self.columnState += 1

    def createVisual(self, labels=False):
        """
        Creates and displays a PDF file of the structure of the graph, regardless of simplification
        level.
        """

        dot = Digraph()
        # dot.node('',shape='point')
        for key, node in self.nodes.items():
            if not node.replaced:
                display = ''.join([r'\\' if x == '\\' else x for x in node.regex])
                if labels:
                    display += '  ({key})'.format(key=key)
                else:
                    pass

                dot.node(str(node.id_), display)

        for edge in self.edges:
            dot.edge(edge.parent, edge.child)

        dot.render(view=True)

    def process(self, id_):
        """
        CHECK
        """
        if self.nodes[id_].topOrExists:
            segments = getTopOrSegments(self.nodes[id_].regex)
            self.nodes[id_].replaced = True
            parents = self.getParents(self.nodes[id_].id_)
            children = self.getChildren(self.nodes[id_].id_)

            for segment in segments:
                n = NODE(segment)
                self.addNode(n)
                for parent in parents:
                    self.removeEdge(parent, self.nodes[id_].id_)
                    self.addEdge(EDGE(parent, n.id_))
                for child in children:
                    self.removeEdge(self.nodes[id_].id_, child)
                    self.addEdge(EDGE(n.id_, child))

        elif '(' in self.nodes[id_].regex:
            segments = getParenthesesSegments(self.nodes[id_].regex)
            parents = self.getParents(self.nodes[id_].id_)
            children = self.getChildren(self.nodes[id_].id_)
            self.nodes[id_].replaced = True
            n = NODE(segments[0])
            self.addNode(n)
            previous = n.id_
            for parent in parents:
                self.removeEdge(parent, self.nodes[id_].id_)
                self.addEdge(EDGE(parent, n.id_))
            for segment in segments[1:]:
                # print('segment: ',segment)
                n = NODE(segment)
                self.addNode(n)
                self.addEdge(EDGE(previous, n.id_))
                previous = n.id_

            for child in children:
                self.removeEdge(self.nodes[id_].id_, child)
                self.addEdge(EDGE(n.id_, child))

        else:
            cQList = getClassQuantList(self.nodes[id_].regex)
            parents = self.getParents(self.nodes[id_].id_)
            self.nodes[id_].replaced = True
            toString = lambda d: d['class'] + str(d['quantifier']) if d['quantifier'] else d['class']

            # Currently the simple parameter does nothing
            # n = NODE(regex=toString(cQList[0]), simple=True)
            n = NODE(regex=toString(cQList[0]))
            self.addNode(n)
            previous = n.id_
            for parent in parents:
                self.addEdge(EDGE(parent, n.id_))
                self.removeEdge(parent, self.nodes[id_].id_)
            for cQ in cQList[1:]:
                # n = NODE(regex=toString(cQ), simple=True)
                n = NODE(regex=toString(cQ))
                self.addNode(n)
                self.addEdge(EDGE(previous, n.id_))
                previous = n.id_

            children = self.getChildren(self.nodes[id_].id_)
            for child in children:
                self.removeEdge(self.nodes[id_].id_, child)
                self.addEdge(EDGE(previous, child))

    def getSharedDecendantSets(self):
        """
        CHECK
        """
        noParents = self.getNodesNoParents()

        combs = list(combinations(noParents, 2))
        d = {}

        for id_ in noParents:
            d[id_] = self.getNodeDescendants(id_)

        pairs = []
        for first, second in combs:
            disjoint = set(d[first]).isdisjoint(d[second])
            if not disjoint:
                pairs.append([first, second])

        sharedDescendantSets = set([])
        for id_ in noParents:
            matches = [pair for pair in pairs if id_ in pair]
            if not matches:
                sharedDescendantSets.add(tuple([id_]))
            else:
                # Test if tuple is necesarry
                sharedDescendantSets.add(tuple(set([x for y in matches for x in y])))
        return sharedDescendantSets
    
    def areNodesConnected(self, nodeList:list):
        for node in nodeList:
            loopBreak = False
            for currNode in nodeList:
                if currNode is node:
                    continue
                if (self.getEdge(node, currNode) or self.getEdge(currNode, node)):
                    loopBreak = True
                    break
            if not loopBreak:
                return False
        return True
                

    def isMergeNodesValid(self, nodeList):
        """
        Returns boolean if a given list of node IDs is able to be merged
        i.e. a->b->c->d, 'b' and 'd' cannot be merged but 'b' 'c' d' can.
        """
        if self.getLonelyNodes():
            # If we are trying to merge any lonely node
            if any(node in self.getLonelyNodes() for node in nodeList):
                if self.edges:
                    return False

        nodeAncestorsList = []
        nodeDescendantsList = []
        # get the ancestors of all nodes in nodeList
        # adds only unique ancestors; no repeats
        for n in nodeList:
            currNodeAncestors = self.getNodeAncestors(n)
            currNodeDescendants = self.getNodeDescendants(n)

            for currNodeAncestItem in currNodeAncestors:
                if currNodeAncestItem not in nodeAncestorsList:
                    nodeAncestorsList.append(currNodeAncestItem)

            for currNodeDescItem in currNodeDescendants:
                if currNodeDescItem not in nodeDescendantsList:
                    nodeDescendantsList.append(currNodeDescItem)

        # List of ids
        topNodes = []
        bottomNodes = []
        # check if the current node is in the list of ancestors
        # if it is then it's not a top node
        for n in nodeList:
            if n not in nodeAncestorsList:
                bottomNodes.append(n)

        for n in nodeList:
            if n not in nodeDescendantsList:
                topNodes.append(n)

        # If the intersection is not distinct AND the nodes that intersect are not a part of nodeList

        # List of ids
        topNodeAncestors = []
        topNodeDescendants = []
        bottomNodeAncestors = []
        bottomNodeDescendants = []

        for topNode in topNodes:
            [topNodeAncestors.append(topNodeAncest) for topNodeAncest in self.getNodeAncestors(topNode) if
             topNodeAncest not in topNodeAncestors]

            [topNodeDescendants.append(topNodeDesc) for topNodeDesc in self.getNodeDescendants(topNode) if
             topNodeDesc not in topNodeDescendants]

        for bottomNode in bottomNodes:
            [bottomNodeAncestors.append(bottomNodeAncest) for bottomNodeAncest in self.getNodeAncestors(bottomNode) if
             bottomNodeAncest not in bottomNodeAncestors]

            [bottomNodeDescendants.append(bottomNodeDesc) for bottomNodeDesc in self.getNodeDescendants(bottomNode) if
             bottomNodeDesc not in bottomNodeDescendants]

        setOfTopNodeAncest = set(topNodeAncestors)
        setOfTopNodeDesc = set(topNodeDescendants)
        setOfBottomNodeAncest = set(bottomNodeAncestors)
        setOfbottomNodeDesc = set(bottomNodeDescendants)

        intersectTABD = list(setOfTopNodeAncest.intersection(setOfbottomNodeDesc))
        intersectTDBA = list(setOfTopNodeDesc.intersection(setOfBottomNodeAncest))

        if not (intersectTABD or intersectTDBA):
            return True
        else:
            for node in intersectTABD:
                try:
                    nodeList.remove(node)
                    nodeList.append(node)
                except ValueError:
                    return False

            for node in intersectTDBA:
                try:
                    nodeList.remove(node)
                    nodeList.append(node)
                except ValueError:
                    return False

            return True

    def createMergedNodes(self, nodeList, nodeRelationship):
        """
        Creates and retuns a NODE object made from a list of node IDs being merged.
        Returns false if the merged node was not able to be made
        """
        if not set(nodeList).issubset(set(self.nodes.keys())):
            raise Exception('Node list includes invalid node.')

        if self.isMergeNodesValid(nodeList):
            M = np.array([self.nodes[n].vector.v for n in nodeList])
            #  print(M)
            newv = M.any(axis=0).astype(int)
            # This is passing in a list of node *ids*
            if nodeRelationship.lower() == 'sequential':
                newQuantifier = self.createMergedSequentialNodesQuantifier(nodeList)
            else:
                newQuantifier = self.createMergedParallelNodesQuantifier(nodeList)

            return NODE(vector=VECTOR(newv), quantifier=newQuantifier)
        else:
            return False

    def createMergedParallelNodesQuantifier(self, nodeList):
        """
        Returns a new quantifier (string) made from the merging of ALL nodes that are parallel
        to eachother. None is returned if no quantifier ({1}) is found. 
        """
        if any(self.nodes[nodeId].getQuantifier for nodeId in nodeList):
            lowestLow = min(
                self.nodes[nodeId].getQuantifierMin if self.nodes[nodeId].getQuantifierMin else 1 for nodeId in
                nodeList)
            highestHigh = max(
                self.nodes[nodeId].getQuantifierMax for nodeId in nodeList if self.nodes[nodeId].getQuantifierMax)

            if lowestLow == highestHigh:
                return lowestLow
            else:
                return str((str(lowestLow) + "," + str(highestHigh)))
        else:
            return None

    def createMergedSequentialNodesQuantifier(self, nodeList):
        """
        Returns a new quantifier (string) made from the merging of ALL nodes that are sequential
        to eachother. If no quantifier is found ({1}), the length of the list of node IDs is returned. 
        """
        if any(self.nodes[nodeId].getQuantifier for nodeId in nodeList):
            lowestLows = [self.nodes[nodeId].getQuantifierMin if self.nodes[nodeId].getQuantifierMin
                          else 1 for nodeId in nodeList]
            highestHighs = [self.nodes[nodeId].getQuantifierMax if self.nodes[nodeId].getQuantifierMax
                            else 1 for nodeId in nodeList]

            sumLowestLows = sum(lowestLows)
            sumHighestHighs = sum(highestHighs)

            if sumLowestLows == sumHighestHighs:
                return sumHighestHighs
            else:
                return str((str(sumLowestLows) + "," + str(sumHighestHighs)))
        else:
            return len(nodeList)

    def mergeRelatedNodes(self, nodeList, nodeRelationship):
        """
        Given a list of node IDs and a shared relationship between them (parallel or sequential), a new node is created
        from merging the given nodes as well as placing itself where the group of nodes was before,
        with new edges created to bind it with affected adjacent edges.
        """
        if not set(nodeList).issubset(set(self.nodes.keys())):
            raise Exception('Node list includes invalid node.')

        if isinstance(nodeRelationship, str):
            if not ((nodeRelationship.lower() == "sequential") or (nodeRelationship.lower() == "parallel")):
                raise Exception("Node relationship is either 'sequential' or 'parallel'")
        else:
            raise Exception("Node relationship should be a string 'sequential' or 'parallel'")

        mergedNode = self.createMergedNodes(nodeList, nodeRelationship)

        self.graphStichIn(nodeList, nodeRelationship, mergedNode)

    def graphStichIn(self, nodeList, nodeRelationship, mergedNode):
        """
        Given a list of node IDs and a shared relationship between them (parallel or sequential) and a new node,
        the new node will replace the position of where all the nodes used to occupy. If the mergedNode is not
        an instance of NODE an exception is thrown.
        """

        # nodeList is a list of ids,
        # while mergedNodes is an actual node

        if isinstance(nodeRelationship, str):
            if not ((nodeRelationship.lower() == "sequential") or (nodeRelationship.lower() == "parallel")):
                raise Exception("Node relationship is either 'sequential' or 'parallel'")
        else:
            raise Exception("Node relationship should be a string 'sequential' or 'parallel'")

        if isinstance(mergedNode, NODE):
            newEdgeList = []

            affectedNodes = self.removeNodeAndEdges(nodeList)
            for upperNode in affectedNodes[0]:
                newEdge = EDGE(upperNode, mergedNode.id_)
                newEdgeList.append(newEdge)
            for lowerNode in affectedNodes[1]:
                newEdge = EDGE(mergedNode.id_, lowerNode)
                newEdgeList.append(newEdge)

            self.nodes[mergedNode.id_] = mergedNode
            self.edges.extend(newEdgeList)

            if nodeRelationship.lower() == "sequential":
                self.sequentialPartition()
            else:
                self.parallelPartition()

            self.partition()
        else:
            raise Exception("The mergedNode is not a node, check merging")

    def createSubGraphNodes(self, nodeList):
        """
        Returns a GRAPH object with nodes and edges made up of all NODE objects given.
        """

        necessaryEdges = [edge for edge in self.edges if
                          (self.nodes[edge.parent] in nodeList and self.nodes[edge.child] in nodeList)]

        nodeDict = dict([(n.id_, n) for n in nodeList])

        nodeListGraph = GRAPH(nodes=nodeDict, edges=necessaryEdges)

        return nodeListGraph

    def mergeToNodeRecursive(self):
        """
        Merges a group of nodes regardless of relationship status. Shoudl be used in conjunction with
        'getNode[Id]ListGraph'.
        """
        self.simplify()
        self.parallelPartition()
        changedNodes = []
        nodeList = []
        if self.parallelGraphs:
            mergeType = "parallel"
            graphWasChanged = False
            for x in self.parallelGraphs:
                if len(x.nodes) != 1:
                    graphWasChanged = True
                    nodeInfoList = x.mergeToNodeRecursive()
                    for infoList in nodeInfoList:
                        changedNodes.append(infoList)

            if graphWasChanged:
                for nodeTuple in changedNodes:
                    self.graphStichIn(nodeTuple[1], mergeType, nodeTuple[0])

            for x in self.nodes:
                nodeList.append(x)
                # merge all nodes in parallel graphs
            self.mergeRelatedNodes(nodeList, mergeType)
        else:
            mergeType = "sequential"
            self.sequentialPartition()
            graphWasChanged = False
            for x in self.sequentialGraphs:
                if len(x.nodes) != 1:
                    graphWasChanged = True
                    nodeInfoList = x.mergeToNodeRecursive()
                    for infoList in nodeInfoList:
                        changedNodes.append(infoList)

            if graphWasChanged:
                for nodeTuple in changedNodes:
                    self.graphStichIn(nodeTuple[1], mergeType, nodeTuple[0])

            for x in self.nodes:
                nodeList.append(x)
                # merge all nodes in sequential graphs
            self.mergeRelatedNodes(nodeList, mergeType)

        for x in self.nodes:
            changedNodes.append((self.nodes[x], nodeList, mergeType))
            return changedNodes

    def mergeNodeIds(self, nodeList):
        """
        Given a list of node IDs, Creates a new node merged from each node, and is
        'stiched' into the graph to preserve continuity. NodeList should contain node IDs
        that are able to be merged
        """
        if self.isMergeNodesValid(nodeList):
            if len(nodeList) > 1:
                # A graph made up of only the nodes
                # nodes = [self.nodes[n] for n in nodeList]
                nodeIdGraph = self.createSubGraph(nodeList)
                nodeIdGraph.partition()

                # Makes a one node graph of those nodes
                fullyMergedNodeInstructions = nodeIdGraph.mergeToNodeRecursive()

                for nodeTuple in fullyMergedNodeInstructions:
                    self.graphStichIn(nodeTuple[1], nodeTuple[2], nodeTuple[0])

    def mergeNodes(self, nodeList):
        """
        Given a list of nodes , Creates a new node merged from each node, and is
        'stiched' into the graph to preserve continuity. NodeList should contain nodes
        that are able to be merged
        """
        toCheck = [x.id_ for x in nodeList]
        if self.isMergeNodesValid(toCheck):
            if len(nodeList) > 1:
                # A graph made up of only the nodes
                nodeIdGraph = self.createSubGraphNodes(nodeList)

                # Makes a one node graph of those nodes
                fullyMergedNodeInstructions = nodeIdGraph.mergeToNodeRecursive()

                for nodeTuple in fullyMergedNodeInstructions:
                    self.graphStichIn(nodeTuple[1], nodeTuple[2], nodeTuple[0])

    def mergeEverything(self):
        """
        Merges every node into one node
        """
        self.mergeNodeIds(list(self.nodes.keys()))

    def createSubGraph(self, nodeList):
        """
        Returns a GRAPH object with nodes and edges made up of all node IDs given.
        """
        subNodes = {}

        for node in nodeList:
            subNodes[node] = self.nodes[node]
        subEdges = [edge for edge in self.edges if (edge.parent in nodeList and edge.child in nodeList)]

        subG = GRAPH(nodes=subNodes, edges=subEdges, alpha=self.alpha)
        return subG

    def phiReduction(self, k): 
        if len(self.nodes.keys()) == 1:
            return True

        subSets = self.getNodeIdSequences(k)
                
        subSets = [list(subSet) for subSet in subSets if self.areNodesConnected(list(subSet))]
        subSets = [list(subSet) for subSet in subSets if self.isMergeNodesValid(list(subSet))]
        
        if subSets is None:
            Gcopy = self.deepCopy()
            phi1 = Gcopy.phi
            Gcopy.mergeEverything()
            phi2 = Gcopy.phi
            if phi2 < phi1:
                self.mergeEverything()
                return False
            else:
                return True
        else:
            for nodeSet in subSets:  # Could also make a method that simply projects the new phi value if [nodes] were to be merged
                Gcopy = self.deepCopy()
                phi1 = Gcopy.phi
                Gcopy.mergeNodeIds(nodeSet)
                phi2 = Gcopy.phi
                if phi2 < phi1:
                    self.mergeNodeIds(nodeSet)
                    return False
            return True

    def reducePhi(self, k):
        hasBeenUpdated = False
        while not hasBeenUpdated:
            hasBeenUpdated = self.phiReduction(k)
    
    def getNodeIdSequences(self, sequenceLength):
        return list(combinations(self.nodes.keys(), sequenceLength))

    def partition(self):
        """
        CHECK
        Partitions the graph into simple nodes. Should be run after instantiating a graph via regex.
        Potent method.
        """
        self.simplify()

        if len(self.nodes) == 1:
            return

        else:
            self.parallelPartition()
            if self.parallelGraphs:  # is not none
                [g.partition() for g in self.parallelGraphs]
            else:
                self.sequentialPartition()
                [g.partition() for g in self.sequentialGraphs]

    def sequentialPartition(self):

        sequentialGraphsNodes = []
        cutSets = self.getCutSets()
        noParents = self.getNodesNoParents()

        currentSet = noParents
        nextCutSet = self.getNextCutSet(currentSet[0])
        descendantList = self.getNodeDescendantsList(currentSet)
        if not nextCutSet and descendantList:
            if currentSet in cutSets:
                sequentialGraphsNodes.append(currentSet)
                sequentialGraphsNodes.append(descendantList)
            else:
                sequentialGraphsNodes.append(currentSet + descendantList)
        else:
            currentSetInCutSets = False
            for cutSet in cutSets:
                if len(cutSet) == currentSet:
                    if sorted(currentSet) == sorted(cutSet):
                        currentSetInCutSets = True
                        break
            if currentSetInCutSets:
                sequentialGraphsNodes.append(currentSet)
                firstSetAdded = True
            else:
                firstSet = currentSet
                firstSetAdded = False
            while nextCutSet or descendantList:
                if nextCutSet:
                    middle = self.getNodesBetweenCutSets(currentSet, nextCutSet)
                    if not firstSetAdded:
                        middle = firstSet + middle
                        firstSetAdded = True
                    if middle:
                        sequentialGraphsNodes.append(middle)
                    sequentialGraphsNodes.append(nextCutSet)
                    currentSet = nextCutSet.copy()
                    descendantList = self.getNodeDescendantsList(nextCutSet)
                    nextCutSet = self.getNextCutSet(nextCutSet[0])
                else:
                    if descendantList:
                        sequentialGraphsNodes.append(descendantList)
                    break
        self.sequentialGraphs = []
        for nodeSet in sequentialGraphsNodes:
            self.sequentialGraphs.append(self.createSubGraph(nodeSet))
        if len(self.sequentialGraphs) == 1:
            self.sequentialGraphs = None
        else:
            pass
        return

    def parallelPartition(self):
        """
        CHECK
        """
        sharedDescendantSets = self.getSharedDecendantSets()

        parallel = []
        for s in sharedDescendantSets:

            nodes = set(s)
            for node in s:
                for descendant in self.getNodeDescendants(node):
                    nodes.add(descendant)
            parallel.append(nodes)
        if len(parallel) == 1:
            self.parallelGraphs = None
        else:
            self.parallelGraphs = [self.createSubGraph(nodeSet) for nodeSet in parallel]

    def reduce(self):
        for node in self.nodes.values():
            node.reduce()

    @property
    def cardinality(self):
        """
        Returns the cardinality, how many strings are able to satisfy the regex of the graph.
        """
        if hasattr(self, 'parallelGraphs') and self.parallelGraphs is not None:
            return sum([g.cardinality for g in self.parallelGraphs])
        elif hasattr(self, 'sequentialGraphs') and self.sequentialGraphs is not None:
            return np.prod([g.cardinality for g in self.sequentialGraphs])
        else:
            k = list(self.nodes.keys())[0]
            return self.nodes[k].cardinality

    @property
    def entropy(self):
        """
        Returns the Information-Theoretic entropy of the given graph's regex.
        """
        return round(np.log2(self.cardinality), 4)

    @property
    def K(self):
        """
        Returns the length of the given graph's regex.
        """
        return len(self.outRegex)

    @property
    def outRegexRecursive(self):
        """
        Returns the joined regexes of each individual subNode, 
        based on parallel or sequential relationship.
        """
        if hasattr(self, 'parallelGraphs') and self.parallelGraphs:
            # if hasattr(self, 'parallelGraphs') and self.parallelGraphs is not None:
            toReturn = "(" + "|".join([g.outRegexRecursive for g in self.parallelGraphs]) + ")"
        elif hasattr(self, 'sequentialGraphs') and self.sequentialGraphs:
            # elif hasattr(self, 'sequentialGraphs') and self.sequentialGraphs is not None:
            toReturn = "".join([g.outRegexRecursive for g in self.sequentialGraphs])
        else:
            k = list(self.nodes.keys())[0]
            toReturn = self.nodes[k].regex
        return toReturn

    @property
    def outRegex(self):
        """
        Returns the regex of the given graph.
        """
        toReturn = self.outRegexRecursive
        if toReturn[0] == '(' and toReturn[-1] == ')':
            return toReturn[1:-1]
        else:
            return toReturn

    @property
    def phi(self):
        """
        Returns the phi of the graph, being the entropy of the graph added to the product
        of the alpha (weight parameter) and its K value.
        """
        return round(np.log2(self.cardinality), 4) + self.alpha * self.K

    @property
    def random(self):
        """
        CHECK
        """
        if self.parallelGraphs:
            weights = [subG.cardinality for subG in self.parallelGraphs]
            subG = random.choices(self.parallelGraphs, weights)[0]
            return subG.random
        elif self.sequentialGraphs:
            return ''.join([subG.random for subG in self.sequentialGraphs])
        else:
            k = list(self.nodes.keys())[0]
            node = self.nodes[k]
            if not node.vector:
                return node.random
