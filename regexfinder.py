import copy
import random
from itertools import combinations, chain
from math import inf

import more_itertools as mit
import numpy as np
import pandas as pd
from graphviz import Digraph

from utils import *


class NODE:
    def __init__(self, regex:str=None, vector=None, replaced=False, alpha=1, quantifier=None):
        """
        Node creation is defined firstly by a regex. If a regex is provided with a vector and/or quantifier,
        the latter two will be ignored and the node will be created from the given regex. If a regex
        is not given, a vector muse be given. Quantifier is optional, but must have a finite upper bound.
        An empty quantifier/none implies a quantifier of {1}
        """

        if regex is None and vector is None:
            raise Exception('Either regex or vector required.')

        self.alpha = alpha
        self.regex:str = regex
        self.vector:VECTOR = vector
        self.id_:str = next(strCounter)
        self.reduced:bool = False
        self.replaced = replaced

        if self.regex is None:
            # if merging nodes this will make the characters go in alphabetical order
            self.regex:str = self.vector.regex
            if quantifier is not None:
                regexUpdates:list[str] = setClassQuantList(self.regex, quantifier)
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
        """Returns the quantifier of the given node. Throws Exception if
        the node is not simple. If the quantifier is {1}, None is returned.

        Raises:
            Exception: Node is not simple

        Returns:
            str: Node quantifier
        """        
        if self.simple:
            return self.classQuantList[0]['quantifier']
        else:
            raise Exception("Node is not simple")

    @property
    def getQuantifierMin(self):
        """Returns the lower bound of the given's node quantifier. Throws Exception
        if the node is not simple. If the quantifier is {1}, None is returned.

        Raises:
            Exception: Node is not simple

        Returns:
            int: Quantifier min
        """        
        if self.simple:
            quantifier = self.getQuantifier
            if quantifier == '?':
                return 0
            else:
                return self.classQuantList[0]['min']
        else:
            raise Exception("Node is not simple")

    @property
    def getQuantifierMax(self):
        """Returns the upper bound of the given's node quantifier. Throws Exception
        if the node is not simple. If the quantifier is {1}, None is returned.

        Raises:
            Exception: Node is not simple

        Returns:
            int: Quantifier max
        """        
        if self.simple:
            quantifier = self.getQuantifier
            if quantifier == '?':
                return 1
            else:
                return self.classQuantList[0]['max']
        else:
            raise Exception("Node is not simple")

    def mergeQuantifiers(self, nodeList:list):
        """Returns the new lower and upper bounds following the merging of the nodes' quantifier.

        Args:
            nodeList (list[str]): list of node ids

        Returns:
            tuple[int,int]: A tuple containing the new minimum and maximum quantifier
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
        """Returns boolean if the node has an or statement seperating the whole regex.

        Returns:
            bool: True or false on whether the top or exists
        """        
        return topOrExists(self.regex)

    def removeOuterParentheses(self):
        """Removes unnecesary outer parenthesis if they exist in a node
        i.e. (a).
        """        
        if hasOuterParentheses(self.regex):
            while self.regex[0] == '(' and self.regex[-1] == ')':
                self.regex = self.regex[1:-1]

    def createVector(self):
        """Creates a vector for the given node.
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
        """Reduces the regex of a given simple node if possible, i.e. 'abcde' -> 'a-e'.
        Same as VECTOR method.

        Raises:
            Exception: Node is not simple. A node cannot be reduced if it is not simple.
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
        """Returns the cardinality, the possible number of strings that satisfy its given regex, of the
        given node.

        Raises:
            Exception: Check class quantifier

        Returns:
            int: Cardinality, number of possible allowable string matches
        """        
        if self.vector is None:
            self.createVector()
        valCount = sum(self.vector.v)
        quantifier = self.classQuantList[0]['quantifier']

        if quantifier is None:
            return valCount
        elif quantifier == '?':
            return valCount + 1
        elif self.classQuantList[0]['min'] is not None and self.classQuantList[0]['max']:
            return sum([valCount ** exponent for exponent in
                        range(self.classQuantList[0]['min'], self.classQuantList[0]['max'] + 1)])
        else:
            raise Exception('Check class quantifier')

    @property
    def singleQuantifier(self):
        """Returns boolean if the quantifier is the same for its lower bound and upper bound i.e. {5}.

        Returns:
            bool: True or false on whether there is a single quantifier
        """        
        return self.getQuantifierMin == self.getQuantifierMax

    @property
    def entropy(self):
        """Returns the entropy of a node, as defined by Information-Theoretic entropy.

        Returns:
            float: Node entropy
        """        
        return round(np.log2(self.cardinality), 4)

    @property
    def K(self):
        """Returns the length of given node's regex.

        Returns:
            int: Regex length
        """        
        return len(self.regex)

    @property
    def phi(self):
        """Returns the phi of the node, being the entropy of the node added to the product
        of the alpha (weight parameter) and its K value.

        Returns:
            float: Node phi
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
        """Returns a random string that satifies a given node's regex.

        Raises:
            Exception: if min != max

        Returns:
            str: Random string that matches the regex
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
    def __init__(self, parent:str, child:str, words=None):
            """CHECK WORDS
            An edge is what connects 2 nodes, and exactly two nodes.
            Its parent and child must not be the same.

            Args:
                parent (str): Parent node
                child (str): Child node
                words (TYPECHECK, optional): CHECK. Defaults to None.

            Raises:
                Exception: Parent and Child must not be the same
            """        
            if words is None:
                words:list = []
            if parent != child:
                self.parent = parent
                self.child = child
            else:
                raise Exception("Parent and child cannot be the same")
            self.words = words


class VECTOR:
    def __init__(self, vector:np.ndarray, alpha=None):
        """        A vector is a matrix of 1's and 0's 128 big, each place
        representing an ASCII character (i.e. a space 33 in represents the '!' character.

        Args:
            vector (np.ndarray): An array of size 128, each position denotes an ASCII character
            alpha ((float, int), optional): Weighted value. Defaults to None.
        """        
        self.v = vector
        self.alpha = alpha

    @property
    def regex(self):
        """Returns the regex built from the given vector.

        Raises:
            Exception: Vector has no nonzero values

        Returns:
            str: The regex of the given vector
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

    @property
    def getndArray(self):
        """Returns Ndarray

        Returns:
            ndarray
        """        
        return self.v

    def reduce(self):
        """Reduces the regex of a given vector if possible, i.e. 'abcde' -> 'a-e'.
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
        """CHECKReturns the first (min) index in the given vector field.

        Returns:
            int: Index of minimum value in ndArray
        """
        return np.min(np.where(self.v))

    @property
    def maxIndex(self):
        """CHECKReturns the last (max) index in the given vector field.

        Returns:
            int: Index of maximum value 
        """        
        return np.max(np.where(self.v))

    @property
    def ent(self):
        """Returns the Informatic-Theory entropy of the given vector field, much like
        a regex.

        Returns:
            float: Entropy of the vector
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
        """Returns the length of the regex built from the given vector.

        Returns:
            int: Length of regex
        """        
        return len(self.regex)

    @property
    def phi(self):
        """Returns the phi of the vector, being the entropy of the vector added to the product
        of the alpha (weight parameter) and its K value.

        Returns:
            float: Vector Phi value
        """        
        return self.ent + self.alpha * self.K

    @property
    def random(self):
        """Returns a random string that satifies a given vector's regex.

        Returns:
            str: String that matches the regex
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
    def __init__(self, words:list):
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
    def __init__(self, regex:str=None, wordList=None, nodes:dict[str, NODE]=None, edges:list[EDGE]=None, alpha=1):
        """        CHECK
        A Graph is a data structure made of nodes and edges. Graphs are built by priority from either a regex,
        a wordlist, or nodes[dictionary]. If a graph is built from a regex, a non-simple node will be
        made and added to the nodes dictionary. Nodes are not required to instantiate a graph object.
        Edges are not required to be made with nodes.

        Args:
            regex (str, optional): Build the graph from a regex. Will take priority if wordlist or nodes are provided. Defaults to None.
            wordList (WORDCOLLECTION, optional): Build the graph from a wordcollectionCHECK. Takes priorty over nodes. Defaults to None.
            nodes (dict[str, NODE], optional): Build the graph from given nodes (Edges are not included here). Defaults to None.
            edges (list[EDGE], optional): Provide a list of edges to pair with nodes. Defaults to None.
            alpha (int, optional): Weighted parameter. Defaults to 1.

        Raises:
            Exception: re.compile failed
        """        
        self.regex = regex
        self.wordList = wordList
        self.alpha = alpha

        self.nodes:dict[str, NODE] = {}
        self.edges:list[EDGE] = []

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

    def deepCopy(self):
        """Returns a content-equal copy of the given graph, with a different
        memory address

        Returns:
            GRAPH: Content-equal copy of this graph
        """        
        return copy.deepcopy(self)

    def addNode(self, node:NODE, edges=None):
        """Adds a node into the given graph. Edges not required.

        Args:
            node (NODE): Node to add
            edges (_type_, optional): Edge to connect to the node. Defaults to None.

        Raises:
            Exception: Node key already exists
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

    def removeNodeAndEdges(self, nodeList:list[str]):
        """Removes the given nodes, and its edges, from the given graph.
        Returns the affected nodes and edges adjecent to the group of nodes removed.

        Args:
            nodeList (list[str]): THe list of nodes to remove

        Returns:
            list[list[str], list[str]]: A list of nodes above and below the affected area
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

    def addEdge(self, edge:EDGE):
        """Adds the given edge to the graph ONLY if the parent and child nodes do not already
        exist and if an edge for said parent and child nodes does not already exist.

        Args:
            edge (EDGE): Edge to add

        Raises:
            Exception: Edge alreay exists
            Exception: Parent or child does not exits
        """        
        for gEdge in self.edges:
            if edge == gEdge:
                raise Exception("Edge already exists")
        if edge.parent not in self.nodes.keys() or edge.child not in self.nodes.keys():
            raise Exception("Parent or Child node does not exist")
        self.edges.append(edge)

    def getEdge(self, parent:str, child:str):
        """Returns EDGE object for a given parent and child node, if it exists. Returns
        false otherwise.

        Args:
            parent (str): Id of parent node
            child (str): Id of child node

        Raises:
            Exception: More than one edge with same parent/child

        Returns:
            EDGE: The requested edge, or False if nonexistent
        """        
        matches = [edge for edge in self.edges if (edge.parent == parent and edge.child == child)]
        if len(matches) > 1:
            raise Exception('More than one edge with same parent/child.')
        elif len(matches) == 1:
            return matches[0]
        else:
            return False

    def removeEdge(self, parent:str, child:str):
        """Removes edge binding two nodes (parent and child) IF they exist in the given graph.

        Args:
            parent (str): Id of the parent node
            child (str): Id of the child node

        Raises:
            Exception: Parent or child node does not exists
        """        
        if parent not in self.nodes.keys() or child not in self.nodes.keys():
            raise Exception("Parent or Child node does not exist")
        toRemove = [edge for edge in self.edges if (edge.parent == parent and edge.child == child)]

        if toRemove:
            [self.edges.remove(edge) for edge in toRemove]

    def removeEdgesList(self, edgeList:list[EDGE]):
        """Removes all edges in the given graph.

        Args:
            edgeList (list[EDGE]): Edges to remove
        """        
        [self.edges.remove(edge) for edge in edgeList]

    def removeStraightShots(self):
        straightShot = self.doesStraightShotExist
        while straightShot:
            self.edges.remove(straightShot)
            straightShot = self.doesStraightShotExist
            

    def getParents(self, id_:str):
        """Returns node ID(s) of the parent(s) of a given node via ID. Returns false if nonexistent.

        Args:
            id_ (str): Id of the node in question

        Returns:
            list[str]: List of nodes that are parents to the nodes in question
        """        
        return [x.parent for x in self.edges if x.child == id_]

    def getChildren(self, id_:str):
        """Returns node ID(s) of the child(ren) of a given node via ID. Returns false if nonexistent.

        Args:
            id_ (str): Id of the node in question

        Returns:
            list[str]: Children of the node in question
        """        
        return [x.child for x in self.edges if x.parent == id_]

    def getSiblings(self, id_:str):
        """Returns neighboring nodes, siblings of a given node via ID. Nodes are siblings if they share the same parent
        (nodes with no parents are also siblings to eachother)

        Args:
            id_ (str): Id of the node in question

        Returns:
            list[str]: A list of the siblings of the node in question
        """         
        if not self.getParents(id_):
            return self.getNodesNoParents()
        else:
            return sorted(list(set(flatten([self.getChildren(parent) for parent in self.getParents(id_)]))))

    def getNodesNoChildren(self):
        """Returns node IDs for all nodes with no children.

        Returns:
            list[str]: A list of node ids that have no children
        """        
        return [x.id_ for x in self.nodes.values() if not self.getChildren(x.id_)]

    def getNodesNoParents(self):
        """Returns node IDs for all nodes with no parents (nodes that do not extend from any node).

        Returns:
            list[str]: A list of node ids that have no parents
        """        
        return [x.id_ for x in self.nodes.values() if not self.getParents(x.id_)]

    def getLonelyNodes(self):
        """Returns all nodes with no edges.

        Returns:
            list[str]: A list of all nodes(ids) with no edges
        """        
        setOfNoParents = set(self.getNodesNoParents())
        setOfNoChildren = set(self.getNodesNoChildren())

        return list(setOfNoChildren.intersection(setOfNoParents))

    def getNotSimple(self):
        """Returns all nodes that are not simple.

        Returns:
            list[str]: A list of all nodes(ids) that are not simple
        """        
        return [id_ for id_, node in self.nodes.items() if (not node.replaced and not node.simple)]

    def simplify(self):
        """CHECK
        Simplifies all existing nodes.
        """        
        while self.getNotSimple():
            self.process(self.getNotSimple()[0])
        self.nodes = dict([(name, node) for name, node in self.nodes.items() if node.simple])

    def getNodeEqClasses(self):
        """Returns a list of node IDs that have the same parents and children.

        Returns:
            list[str]: A list of all nodes that have the same parents and children
        """        
        # two nodes are equivalent if they have the same parents and children
        tempDict:dict[tuple[str, str], str] = {}
        for id_ in self.nodes.keys():
            parents = tuple(sorted(self.getParents(id_)))
            children = tuple(sorted(self.getChildren(id_)))
            if (parents, children) in tempDict.keys():
                tempDict[(parents, children)].append(id_)
            else:
                tempDict[(parents, children)] = [id_]
        return list(tempDict.values())

    def getGenerationalSets(self):
        """A method to get a list of how far away each node is from the topNodes (via edges)

        Returns:
            list[[list[tuple]]]: Returns a list of lists containing tuples (nodeId, distanceFromTopNodes). List groups are ordered.
        """        

        topNodes = self.getNodesNoParents()
        generationMarkedNodes = [(topNode, 1) for topNode in topNodes]
        
        for node in topNodes:
            children = self.getChildren(node)
            for child in children:
                returnList = self.getGenerationalSetsRecursive(child, 2)
                for element in returnList:
                    generationMarkedNodes.append(element)

        maxDepth = 0
        for element in generationMarkedNodes:
            depth = element[1]
            if depth > maxDepth:
                maxDepth = depth
        
        nodeIdList = []
        recDepthList = []
        for element in generationMarkedNodes:
            if element[0] in nodeIdList:
                nodeIdPlace = nodeIdList.index(element[0])
                if element[1] > recDepthList[nodeIdPlace]:
                    nodeIdList.append(element[0])
                    recDepthList.append(element[1])
                    nodeIdList.pop(nodeIdPlace)
                    recDepthList.pop(nodeIdPlace)
            else:
                nodeIdList.append(element[0])
                recDepthList.append(element[1])
                
        
        generationMarkedNodes = [(nodeIdList[n], recDepthList[n]) for n in range(len(nodeIdList))]
        
        toReturn = [[] for _ in range(maxDepth)]
        for element in generationMarkedNodes:
            place = element[1]-1
            toReturn[place].append(element[0])
        
        return toReturn

    def getGenerationalSetsRecursive(self, currNodeId, recursionDepth):   
        """Recursive method for getGenerationalSets

        Args:
            currNodeId (str): The id of the current node being recursed into
            recursionDepth (int): The current recursion depth

        Returns:
            list[tuple]: [Recursive return] Returns information of all children of current Node
        """        
        
        '''
        Base Case:
            At a node. Return (nodeId, depth) + all previous nodes and depths
        Else:
            Recurse into children, +1 depth
        '''
        
        children = self.getChildren(currNodeId)
        returnsList = [(currNodeId, recursionDepth)]
        
        if children:
            for child in children:
                downstream = self.getGenerationalSetsRecursive(child, recursionDepth+1)
                for element in downstream:
                    returnsList.append(element)
                
        return returnsList

    def getNodeAncestorsList(self, inList:list[str]):
        """Returns a list of node IDs of ancestors, the parents of all of each node's parents

        Args:
            inList (list[str]): A list of node ids in question

        Returns:
            list[str]: A list of all the nodes' ancestors
        """        
        toReturn = []
        for id_ in inList:
            if self.getNodeAncestors(id_) not in toReturn:
                toReturn += self.getNodeAncestors(id_)
        return toReturn

    def getNodeDescendantsList(self, inList:list[str]):
        """Returns a list of node IDs of descendants, the children of all of each node's children

        Args:
            inList (list[str]): A list of the node ids in question

        Raises:
            Exception: Value inList must be a list

        Returns:
            list[str]: A list of all the nodes' descendants
        """        
        if not isinstance(inList, list):
            raise Exception(('value inList must be a list'))
        toReturn = []
        for id_ in inList:
            if self.getNodeDescendants(id_) not in toReturn:
                toReturn += self.getNodeDescendants(id_)
        toReturn = list(set(toReturn))
        return toReturn

    def getNodeAncestors(self, id_:str):
        """Returns a list of node IDs of ancestors, the parents of a given node's parents

        Args:
            id_ (str): Id of the node in question

        Returns:
            list[str]: A list of the node's ancestors
        """        
        ancestors:list[str] = []
        parents = self.getParents(id_)
        while parents:
            parent = parents.pop()
            parents += self.getParents(parent)
            ancestors.append(parent)
        return sorted(list(set(ancestors)))

    def getNodeDescendants(self, id_:str):
        """Returns a list of node IDs of descendants, the children of a given node's children

        Args:
            id_ (str): Id of the node in question

        Returns:
            list[str]: A list of the noode's descendants
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

    def getGroupNodeAncestOrDesc(self, nodeList:list[str], relationship:str):
        """Returns the unique ancestors or descendants of a given group of nodes. Nodes within the group are not considered
        for ancestry.

        Args:
            nodeList (list[str]): The list of node Ids to be checked
            relationship (str): Relationship is either "ancestors" or "descendants", case insensitive

        Raises:
            Exception: Exception is raised if relationship is not entered as "ancestors" or "descendants"

        Returns:
            list[str]: The list of ancestors or descendants, as node ids
        """
        
        if relationship.lower() != 'ancestors' and relationship.lower() != 'descendants':
            raise Exception("Relationship is either 'ancestors' or 'descendants'")
        
        totalRelatives = []
        for node in nodeList:
            if relationship.lower() == 'ancestors':
                nodeRelatives = self.getNodeAncestors(node)
            else:
                nodeRelatives = self.getNodeDescendants(node)
            toRemove = []
            for ancestor in nodeRelatives:
                if ancestor in nodeList:
                    toRemove.append(ancestor)
                    
            for remove in toRemove:
                nodeRelatives.remove(remove)
                
            if nodeRelatives:
                totalRelatives.extend(nodeRelatives)
        
        return totalRelatives
    
    # A set of nodes is a CutSet if the set of nodes union with all its ancestors
    # and al of its descendants yields the entire graph
    def testCutSet(self, inList:list[str]):
        """Tests a cutSet

        Args:
            inList (list[str]): The list of node ids to test

        Raises:
            Exception: Exception raise if a node included is not in self.nodes.keys()

        Returns:
            boolean: Returns true if the cutSet can yeild the entire graph
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
        # The union of all the ancestors, descendants, and initial nodes should yeild the entire graph
        # if it is a valid cutSet
        return set(inList + allAncestors + allDescendants) == set(self.nodes.keys())

    def getCutSets(self):
        """A cutSet is a set of nodes that, when unioned with all its ancestors and descendants, yeild the entire graph.

        Returns:
            list[list[str]]: Returns all eqClasses and noneqClass cutSets in order from top to bottom
        """
        eqClasses = self.getNodeEqClasses()

        sortedEqClasses = [sorted(eqClass) for eqClass in eqClasses if self.testCutSet(eqClass)]
        fullOrderedCutSets = []
        
        nodesNoParents = self.getNodesNoParents()
        
        if (len(sortedEqClasses)) == 1:
            ancestors = self.getNodeAncestors(sortedEqClasses[0][0])
            if ancestors:
                sortedEqClasses.append(ancestors)
            descendants = self.getNodeDescendants(sortedEqClasses[0][0])
            if descendants:
                sortedEqClasses.append(descendants)

        flat_list = [item for sublist in sortedEqClasses for item in sublist]
        if sorted(flat_list) != sorted(list(self.nodes.keys())) and len(sortedEqClasses) != 1:
            
            '''
            Get the highest eqClass, then move down from there.
            Get the next eqClass below it, check for any in between nodes
            '''
            
            topEqClass = []
            
            for eqClass in sortedEqClasses:
                if set(nodesNoParents).issubset(set(eqClass)):
                    topEqClass = eqClass.copy()
            
            if not topEqClass:
                topEqClass = self.getNextEqClass(nodesNoParents, sortedEqClasses)
            
            totalTopEqGraph = self.getGroupNodeAncestOrDesc(topEqClass, 'ancestors')
            
            if totalTopEqGraph:
                currentCutSet = totalTopEqGraph.copy()
            else:
                currentCutSet = topEqClass.copy()
            
            fullOrderedCutSets.append(sorted(currentCutSet))
            nextCutSet:list[str] = self.getNextEqClass(currentCutSet, sortedEqClasses)
            inBetweenGraphs:list[str] = []
            while nextCutSet:
                
                totalDescendants = self.getGroupNodeAncestOrDesc(currentCutSet, 'descendants')
                
                totalAncestors = self.getGroupNodeAncestOrDesc(nextCutSet, 'ancestors')
                
                inBetween = set(totalAncestors).intersection(set(totalDescendants))

                if inBetween:
                    fullOrderedCutSets.append(sorted(inBetween))
                fullOrderedCutSets.append(sorted(nextCutSet))
                
                currentCutSet = nextCutSet.copy()
                
                nextCutSet:list[str] = self.getNextEqClass(currentCutSet, sortedEqClasses)

            lastEqGraph = currentCutSet.copy()
            lastEqGraphDescendants = self.getGroupNodeAncestOrDesc(lastEqGraph, 'descendants')
            
            if lastEqGraphDescendants:
                if lastEqGraphDescendants not in inBetweenGraphs:
                    fullOrderedCutSets.append(sorted(lastEqGraphDescendants))
        else:
            for eqClass in sortedEqClasses:
                if set(nodesNoParents).issubset(set(eqClass)):
                    fullOrderedCutSets.append(sortedEqClasses.pop(sortedEqClasses.index(eqClass)))
                    break
            
            while sortedEqClasses:
                toAppend = self.getNextEqClass(fullOrderedCutSets[len(fullOrderedCutSets)-1], sortedEqClasses)
                fullOrderedCutSets.append(toAppend)
                sortedEqClasses.remove(toAppend)

        return fullOrderedCutSets

    def getNextEqClass(self, eqClass:list[str], eqClassList:list[list[str]]):
        """Gets the next sequential eqClas

        Args:
            eqClass (list[str]): The current eqClass
            eqClassList (list[list[str]]): The entire list of eqClasses

        Returns:
            list[str]: The next eqClass
        """        
        
        for node in eqClass:
            children = self.getChildren(node)
            while children:
                for kid in children:
                    for currEqClass in eqClassList:
                        if kid in currEqClass:
                            if eqClass != currEqClass:
                                return currEqClass
                children1 = []
                for kid in children:
                    children1.extend(self.getChildren(kid))
                children.clear()
                children = children1
        
        return None

    def getCutSetGroup(self, cutSetUpper:list[str], cutSetLower:list[str]):
        """NEEDSWORK gets the next sequential cutSet

        Args:
            cutSetUpper (list[str]): The cutSet above
            cutSetLower (list[str]): The cutSet below
        """        
        
        fullNodeGroup = []
        fullEdgeGroup = []
        fullNodeGroup.extend(cutSetUpper)
        fullNodeGroup.extend(cutSetLower)
        
        intersection:set[str] = set(self.getGroupNodeAncestOrDesc(cutSetUpper, 'descendants')).intersection(set(self.getGroupNodeAncestOrDesc(cutSetLower, 'ancestors')))
        if intersection:
            fullNodeGroup.extend(intersection)
            for edge in self.edges:
                if edge.parent in fullNodeGroup or edge.child in fullNodeGroup:
                    fullEdgeGroup.append(edge)

    def getNextCutSet(self, id_:str):
        """NEEDSWORK Gets the next subsequent cutSet

        Args:
            id_ (str): Id of a node in the current cutSet

        Raises:
            Exception: Node id is not valid

        Returns:
            list[str]: A list of the node ids in the next cutSet
        """        
        if not id_ in self.nodes.keys():
            raise Exception('Node id not valid.')
        children = self.getChildren(id_)
        while not (children in self.getCutSets()) and children:
            newchildren = self.getChildren(children[0])
            children = newchildren
        return children

    def getNodesBetweenCutSets(self, cutSetUpper:list, cutSetLower:list):
        """Gets all nodes between cutSets

        Args:
            cutSetUpper (list): The upper cutSet
            cutSetLower (list): The lower cutSet

        Raises:
            Exception: cutSetUpper is not a cutSet and is not noParents
            Exception: cutSetLower is not a cutset
            Exception: cutSetUpper is not ancestor of cutSetLower

        Returns:
            list[str]: A list of node ids between the cutSets
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
        """Creates and displays a PDF file of the structure of the graph, regardless of simplification
        level.

        Args:
            labels (bool, optional): Displays labels for each node. Used for merging. Defaults to False.
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

    def createMultiGraphVisual(self, graphList:list=None, graphRegexList:list[str]=None, includeThisGraph:bool=False, labels=False):
        """Creates a visual of multiple graphs at once

        Args:
            graphList (list[GRAPH], optional): A list of GRAPH objects to visualize. Prioritized. Defaults to None.
            graphRegexList (list[str], optional): A list of GRAPH regexes to visualize. Defaults to None.
            includeThisGraph (bool, optional): Whether this graph should be included in the visualization. Defaults to True.
            labels (bool, optional): Whether labels for nodes are on. Defaults to False.

        Raises:
            Exception: Either a graphList or a graphRegexList must be provided
        """        
        if not graphList and not graphRegexList:
            raise Exception("A list of GRAPH object or GRAPH regexes must be passed in")
        

        if includeThisGraph:
            combinedRegex = self.outRegex
            firstOr = 1
        else:
            combinedRegex = ""
            firstOr = 0
        
        
        if graphList:
            for graph in graphList:
                if firstOr:
                    combinedRegex += "|"
                else:
                    firstOr += 1
                    
                combinedRegex += graph.outRegex
        else:
            for graph in graphRegexList:
                if firstOr:
                    combinedRegex += "|"
                else:
                    firstOr += 1
                    
                combinedRegex += graph
            
            
        toVisualize = GRAPH(regex=combinedRegex)
        toVisualize.partition()
        toVisualize.createVisual(labels=labels)

    def process(self, id_:str):
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

        combs = list(combinations(noParents, 2)) # every possible pair made of every node with no parent
        d = {} # dictionary of every noParent node's descendants

        for id_ in noParents:
            d[id_] = self.getNodeDescendants(id_)

        pairs = []
        for first, second in combs:
            disjoint = set(d[first]).isdisjoint(set(d[second]))
            if not disjoint: # do the pair share any descendants
                pairs.append([first, second])

        sharedDescendantSets = set()
        for id_ in noParents:
            matches = [pair for pair in pairs if id_ in pair] # for this id_, find every pair it's in
            if not matches: # if matches is empty
                sharedDescendantSets.add(tuple([id_])) # add this id_
            else:
                # Test if tuple is necesarry
                sharedDescendantSets.add(tuple(set([x for y in matches for x in y]))) # otherwise
        return sharedDescendantSets

    def areNodesConnected(self, nodeList:list[str]):
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

    def checkGenerationalRelationship(self, nodeList:list[str]):
        """Returns True if a given list of node IDs is able to be merged
        i.e. a->b->c->d, 'b' and 'd' cannot be merged but 'b' 'c' d' can.

        Args:
            nodeList (list[str]): The list of node ids in question

        Returns:
            bool: True or False on whether the nodes can be merged
        """        
        if self.getLonelyNodes():
            # If we are trying to merge any lonely node
            if any(node in self.getLonelyNodes() for node in nodeList):
                if self.edges:
                    return False

        # get the ancestors of all nodes in nodeList
        # adds only unique ancestors; no repeats
        nodeAncestorsList = []
        nodeDescendantsList = []
        for n in nodeList:
            currNodeAncestors = self.getNodeAncestors(n)
            currNodeDescendants = self.getNodeDescendants(n)

            nodeAncestorsList.extend(list(set(currNodeAncestItem for currNodeAncestItem in currNodeAncestors)))
            nodeDescendantsList.extend(list(set(currNodeDescItem for currNodeDescItem in currNodeDescendants)))

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

        for topNode in topNodes:
            topNodeAncestors = set(topNodeAncest for topNodeAncest in self.getNodeAncestors(topNode))
            topNodeDescendants = set(topNodeDesc for topNodeDesc in self.getNodeDescendants(topNode))

        for bottomNode in bottomNodes:
            bottomNodeAncestors = set(bottomNodeAncest for bottomNodeAncest in self.getNodeAncestors(bottomNode))
            bottomNodeDescendants = set(bottomNodeDesc for bottomNodeDesc in self.getNodeDescendants(bottomNode))

        intersectTABD = list(topNodeAncestors.intersection(bottomNodeDescendants))
        intersectTDBA = list(topNodeDescendants.intersection(bottomNodeAncestors))

        if not (intersectTABD or intersectTDBA):
            return True
        else:
            for node in intersectTABD:
                if node not in nodeList:
                    return False

            for node in intersectTDBA:
                if node not in nodeList:
                    return False
            return True

    def willNgraphAppear(self, nodeList:list[str]):
        """Check to see if merging the group of nodes will result in an n-graph

        Args:
            nodeList (list[str]): The list of node ids in question

        Returns:
            bool: True or False depending on whether the n-graph will appear
        """ 
        '''
        If a node has multiple parents, each parent must have it as its unique child,
        straight shots are an exception
        '''      

        tempList = nodeList.copy()
        toRemove = set()
        for i in range(len(tempList)): # Removing all nodes that are directly related to each other
            currNodeDescendants = self.getNodeDescendants(tempList[i])
            for kid in currNodeDescendants:
                if kid in tempList:
                    toRemove.add(kid)
        
        for item in toRemove:
            tempList.remove(item)

        totalParents = []
        for node in tempList:
            parentList = self.getParents(node)
            if parentList:
                totalParents.extend(parentList)

        totalParents = list(set(totalParents))
        if len(totalParents)>1:
            for currNode in tempList:
                currNodeParents = self.getParents(currNode)
                if not set(currNodeParents).issubset(set(tempList)):
                    for parent in currNodeParents:
                        parentsChildren = self.getChildren(parent)
                        if len(parentsChildren)>1:
                            if not set(parentsChildren).issubset(set(tempList)):
                                # for kid in parentsChildren:
                                result = self.checkAlternatePath(parent, currNode, 0)
                                if not result:
                                    # If there are no other alternate pathes, it is not a straight shot
                                    return True
        return False
        
        # return self.checkAlternatePathMultiNodes(self.getNodesNoParents(), nodeList, 0)

    def willStraightShotAppear(self, nodeList:list[str]):
        """Checks to see if the nodes will create a straight shot if merged

        Args:
            nodeList (list[str]): The list of node ids to be checked (if merged)

        Returns:
            bool: Whether or not a straight shot will be created
        """
        
        upperParents = []
        lowerChildren = []
        for node in nodeList:
            currNodeParents = self.getParents(node)
            currNodeChildren = self.getChildren(node)
            for kid in currNodeParents:
                if kid not in nodeList:
                    upperParents.append(kid)
            for kid in currNodeChildren:
                if kid not in nodeList:
                    lowerChildren.append(kid)
        
        results = []    
        for parent in upperParents:
            results.append(self.checkAlternatePathMultiNodesQuestion(parent, nodeList, 0))
        for kid in lowerChildren:
            results.append(self.checkAlternatePathMultiNodesUpStream(nodeList, kid, 0))
            
        return any(results)

    def isMergeNodesValid(self, nodeList:list[str]):
        """Checks whether merging a group of nodes is allowed

        Args:
            nodeList (list[str]): The list of node ids to be merged

        Returns:
            bool: True or False on whether the merge is allowed
        """        
        return (self.checkGenerationalRelationship(nodeList) and not self.willNgraphAppear(nodeList))

    def createMergedNodes(self, nodeList:list[str], nodeRelationship:str):
        """Creates and retuns a NODE object made from a list of node IDs being merged.
        Returns false if the merged node was not able to be made

        Args:
            nodeList (list[str]): The list of node ids to merge
            nodeRelationship (str): Whether the nodes are sequential or parallel *precondition needed

        Raises:
            Exception: Node list include invalid node

        Returns:
            NODe, False: Returns the new node, False if can't be merged
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

    def createMergedParallelNodesQuantifier(self, nodeList:list[str]):
        """Returns a new quantifier (string) made from the merging of ALL nodes that are parallel
        to eachother. None is returned if no quantifier ({1}) is found.

        Args:
            nodeList (list[str]): The list of node ids to be merged

        Returns:
            str | None: Returns the new quantifier (none if there are no quantifiers present)
        """        
        if any(self.nodes[nodeId].getQuantifier for nodeId in nodeList):
            lowestLow = min(
                self.nodes[nodeId].getQuantifierMin if self.nodes[nodeId].getQuantifierMin is not None else 1 for nodeId in
                nodeList)
            highestHigh = max(
                self.nodes[nodeId].getQuantifierMax for nodeId in nodeList if self.nodes[nodeId].getQuantifierMax)

            if lowestLow == highestHigh:
                return lowestLow
            else:
                return str((str(lowestLow) + "," + str(highestHigh)))
        else:
            return None

    def createMergedSequentialNodesQuantifier(self, nodeList:list[str]):
        """Returns a new quantifier (string) made from the merging of ALL nodes that are sequential
        to eachother. If no quantifier is found ({1}), the length of the list of node IDs is returned.

        Args:
            nodeList (list[str]): The list of nodes to be merged

        Returns:
            str | int: The new quantifier (int if there are no quantifiers present, they get added up)
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

    def mergeRelatedNodes(self, nodeList:list[str], nodeRelationship:str):
        """Given a list of node IDs and a shared relationship between them (parallel or sequential), a new node is created
        from merging the given nodes as well as placing itself where the group of nodes was before,
        with new edges created to bind it with affected adjacent edges.

        Args:
            nodeList (list[str]): The list of node ids to be merged
            nodeRelationship (str): Whether the nodes are sequential or parallely related

        Raises:
            Exception: Node list includes invalid node
            Exception: Node relationship is not either 'sequential' or 'parallel'
            Exception: Node relationship is not a string
        """        
        if not set(nodeList).issubset(set(self.nodes.keys())):
            raise Exception('Node list includes invalid node.')

        if isinstance(nodeRelationship, str):
            if not ((nodeRelationship.lower() == "sequential") or (nodeRelationship.lower() == "parallel")):
                raise Exception("Node relationship is not either 'sequential' or 'parallel'")
        else:
            raise Exception("Node relationship should be a string 'sequential' or 'parallel'")

        mergedNode = self.createMergedNodes(nodeList, nodeRelationship)

        self.graphStichIn(nodeList, mergedNode)

    def graphStichIn(self, nodeList:list[str], mergedNode:NODE):
        """Given a list of node IDs and a shared relationship between them (parallel or sequential) and a new node,
        the new node will replace the position of where all the nodes used to occupy. If the mergedNode is not
        an instance of NODE an exception is thrown.

        Args:
            nodeList (list[str]): The list of node ids to be merged
            nodeRelationship (str): The relationship of the nodes
            mergedNode (NODE): The merged node from the given node ids

        Raises:
            Exception: Node relationship is not either 'sequential' or 'parallel'
            Exception: Node relationship should be a string 'sequential' or 'parallel'
            Exception: The mergedNode is not a node, check merging
        """        

        # nodeList is a list of ids,
        # while mergedNodes is an actual node

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

            self.partition()
        else:
            raise Exception("The mergedNode is not a node, check merging")

    def createSubGraphNodes(self, nodeList:list[NODE]):
        """Returns a GRAPH object with nodes and edges made up of all NODE objects given.

        Args:
            nodeList (list[NODE]): The list of nodes to be used to create the subGraph

        Returns:
            GRAPH: A new graph built from the given nodes, and edges provided from this graph
        """        

        necessaryEdges = [edge for edge in self.edges if
                          (self.nodes[edge.parent] in nodeList and self.nodes[edge.child] in nodeList)]

        nodeDict = dict([(n.id_, n) for n in nodeList])

        nodeListGraph = GRAPH(nodes=nodeDict, edges=necessaryEdges)

        return nodeListGraph

    def mergeToNodeRecursive(self):
        """Merges a group of nodes regardless of relationship status. Should be used in conjunction with
        'getNode[Id]ListGraph'.

        Returns:
            list[str]: (Recursive return) All changed nodes from this change and downstream
        """    
        
        '''
        Base case:
            The subGraph you're at is made up of only simple nodes. Merge them. Return the new node as well as what the relationship was
        Else:
            Go through each subGraph. If the subGraph is not simple (has more than one node) recurse into the graph until the subGraphs are simple,
            then treat the first iteration as a base case
        '''
        self.partition()
        
        if hasattr(self, 'parallelGraphs') and self.parallelGraphs is not None:
            eachSimpleGraph = [graph for graph in self.parallelGraphs if len(graph.nodes) == 1]
            if len(self.parallelGraphs) == len(eachSimpleGraph):
                nodeIdList = list(self.nodes.keys())
                self.mergeRelatedNodes((nodeIdList), 'parallel')
                return (nodeIdList, list(self.nodes.values())[0])
            else:
                originalKeys = list(self.nodes.keys())
                for graph in self.parallelGraphs:
                    if len(graph.nodes) > 1:
                        returnTuple = graph.mergeToNodeRecursive()
                        self.graphStichIn(returnTuple[0], returnTuple[1])
                nodeIdList = list(self.nodes.keys())
                self.mergeRelatedNodes((nodeIdList), 'parallel')
                return (originalKeys, list(self.nodes.values())[0])
            
        elif hasattr(self, 'sequentialGraphs') and self.sequentialGraphs is not None:
            eachSimpleGraph = [graph for graph in self.sequentialGraphs if len(graph.nodes) == 1]
            if len(self.sequentialGraphs) == len(eachSimpleGraph):
                nodeIdList = list(self.nodes.keys())
                self.mergeRelatedNodes((nodeIdList), 'sequential')
                return (nodeIdList, list(self.nodes.values())[0])
            else:
                originalKeys = list(self.nodes.keys())
                for graph in self.sequentialGraphs:
                    if len(graph.nodes) > 1:
                        returnTuple = graph.mergeToNodeRecursive()
                        self.graphStichIn(returnTuple[0], returnTuple[1])
                nodeIdList = list(self.nodes.keys())
                self.mergeRelatedNodes((nodeIdList), 'sequential')
                return (originalKeys, list(self.nodes.values())[0])   
        else:
            raise Exception("Sequential and Parallel Graphs do not exist, somehow")
                

    def mergeNodeIds(self, nodeList:list[str]):
        """Given a list of node IDs, Creates a new node merged from each node, and is
        'stiched' into the graph to preserve continuity. NodeList should contain node IDs
        that are able to be merged

        Args:
            nodeList (list[str]): A list of node ids to be merged
        """        
        if self.isMergeNodesValid(nodeList):
            if len(nodeList) > 1:
                # A graph made up of only the nodes
                # nodes = [self.nodes[n] for n in nodeList]
                nodeIdGraph = self.createSubGraph(nodeList)
                # nodeIdGraph.partition()

                # Makes a one node graph of those nodes
                returnTuple = nodeIdGraph.mergeToNodeRecursive()

                self.graphStichIn(returnTuple[0], returnTuple[1])

    def multiMergeNodeIds(self, listOfNodeLists:list[list[str]]):
        """Same as mergeNodeIds, but simultaneous merges

        Args:
            listOfNodeLists (list[list[str]]): A list of the list of nodes to be merged

        Raises:
            Exception: Empty list provided
            Exception: Nodes in set of merges must contain unique nodes
        """        
        if len(listOfNodeLists) > 1:
            startPos = 0
            while startPos != len(listOfNodeLists):
                toCompare = listOfNodeLists[startPos]
                for nodeListPos in range(startPos+1, len(listOfNodeLists)):
                    if not toCompare or not listOfNodeLists[nodeListPos]:
                        raise Exception('Empty list provided. Cannot merge empty lists')
                    intersection = set(listOfNodeLists[nodeListPos]).intersection(set(toCompare))
                    if intersection:
                        raise Exception("Nodes in set of merges must contain unique nodes")
                startPos+=1

        for nodeList in listOfNodeLists:
            self.mergeNodeIds(nodeList)

    def mergeNodes(self, nodeList:list[NODE]):
        """Given a list of nodes , Creates a new node merged from each node, and is
        'stiched' into the graph to preserve continuity. NodeList should contain nodes
        that are able to be merged

        Args:
            nodeList (list[NODE]): THe list of node objects to be merged
        """        
        toCheck = [x.id_ for x in nodeList]
        if self.isMergeNodesValid(toCheck):
            if len(nodeList) > 1:
                # A graph made up of only the nodes
                nodeIdGraph = self.createSubGraphNodes(nodeList)

                # Makes a one node graph of those nodes
                returnTuple = nodeIdGraph.mergeToNodeRecursive()

                self.graphStichIn(returnTuple[0], returnTuple[1])

    def mergeEverything(self):
        """
        Merges every node into one node
        """
        self.mergeNodeIds(list(self.nodes.keys()))

    def createSubGraph(self, nodeList:list[str]):
        """Returns a GRAPH object with nodes and edges made up of all node IDs given.

        Args:
            nodeList (list[str]): The list of node objects to be used to create the new graph

        Returns:
            GRAPH: A new graph built from the provided nodes, and edges provided from this graph
        """        
        subNodes = {}

        for node in nodeList:
            subNodes[node] = self.nodes[node]
        # Below gets all of its edges too
        subEdges = [edge for edge in self.edges if (edge.parent in nodeList and edge.child in nodeList)]

        subG = GRAPH(nodes=subNodes, edges=subEdges, alpha=self.alpha)
        return subG

    def getValidSubGraphs(self, stichSubGraphs=False):
        """Returns every valid subGraphs in the graph
        
        Args:
            stichSubGraphs (bool, optional): Whether all subGraphs should automatically be stiched or not. Defaults to False.

        Returns:
            list[str]: A list of all valid subGraphs. Some may be tuples, or nested
        """     
        
        subGraphList = self.getValidSubGraphsRecursive()
        
        topNodes = self.getNodesNoParents()
        bottomNodes = self.getNodesNoChildren()
        
        # topNodeChildren = set()
        # bottomNodeParents = set()
        # if not self.parallelGraphs:
        #     for node in topNodes:
        #         topNodeChildren.extend(self.getChildren(node))
        
        # regex = '([ab][ab])|(c[abd])|([abc]d)' creates an N-Graph because of one of the combinations. 
        # Must have to do with merging across graphs, or not sharing any ancestors
        generationSets = self.getGenerationalSets()
        if len(topNodes) > 1:
            if topNodes not in subGraphList:
                subGraphList.append(self.createSubGraph(topNodes))
            if len(topNodes) > 2:
                if len(generationSets[1]) < 2:
                    for i in range (2, len(topNodes)):
                        topNodeCombs = list(combinations(topNodes, i))
                        for nodeComb in topNodeCombs:
                            topCombGraph = self.createSubGraph(list(nodeComb))
                            subGraphList.append(topCombGraph)
        if len(bottomNodes) > 1:
            if bottomNodes not in subGraphList:
                subGraphList.append(self.createSubGraph(bottomNodes))
            if len(bottomNodes) > 2:
                if len(generationSets[len(generationSets)-2]) == 1:
                    for i in range (2, len(bottomNodes)):
                        bottomNodeCombs = list(combinations(bottomNodes, i))
                        for nodeComb in bottomNodeCombs:
                            bottomCombGraph = self.createSubGraph(list(nodeComb))
                            subGraphList.append(bottomCombGraph)
            
        if stichSubGraphs and isinstance(subGraphList, list):
            for i in range(len(subGraphList)-1):
                if isinstance(subGraphList[i], list) or isinstance(subGraphList[i], tuple):
                    subGraphList[i] = self.mergeValidSubGraphs(subGraphList[i])
        
        # Debug Lines below (excl. return)
        if isinstance(subGraphList, list):
            print("Nodes", len(self.nodes), "VSG", len(subGraphList))
        return subGraphList
    
    def getValidSubGraphsRecursive(self):
        """Returns every valid subGraph in the graph, resursive method

        Returns:
            list[str]: (Recursive Return) returns all the valid subgraphs in this graph. Some may be tuples
        """        
        if len(self.nodes) == 1:
            return None
        else:
            parGraphSubGraphs = []
            seqGraphSubGraphs = []
            
            if self.parallelGraphs:
                parGraphLength = len(self.parallelGraphs)
                parGraphSubGraphs.append(self)
                if parGraphLength > 2:
                    for i in range (2, parGraphLength):
                        parGraphSubGraphs.extend(combinations(self.parallelGraphs, i))
                belowGraphs = [g.getValidSubGraphsRecursive() for g in self.parallelGraphs]
                for subGraph in belowGraphs:
                    if subGraph:
                        parGraphSubGraphs.extend(subGraph)
                return parGraphSubGraphs
            else:
                seqGraphLength = len(self.sequentialGraphs)
                seqGraphSubGraphs.append(self)
                if seqGraphLength > 2:
                    for i in range (2, seqGraphLength):
                        for start in range((seqGraphLength-(i))+1):
                            seqGraphSubGraphs.append(self.sequentialGraphs[start:start+(i)])
                belowGraphs = [g.getValidSubGraphsRecursive() for g in self.sequentialGraphs]
                for subGraph in belowGraphs:
                    if subGraph:
                        seqGraphSubGraphs.extend(subGraph)
                return seqGraphSubGraphs
            
    def mergeValidSubGraphs(self, subGraphList:list):
        """Creates a new graph object of sub graphs "stiched" together

        Args:
            subGraphList (list[GRAPH]): The list of graph objects to be merged

        Raises:
            Exception: The subGraphList is not a list

        Returns:
            GRAPH: The subGraphs stiched together
        """        
        
        if isinstance(subGraphList, list) or isinstance(subGraphList, tuple):
            nodeIds = []
            for graph in subGraphList:
                nodeIds.extend(graph.nodes.keys())
            return self.createSubGraph(nodeIds)
        else:
            raise Exception("subGraphList is not a list or tuple")

    def reducePhi(self):
        returnVal = self.phiReduction()
        while returnVal:
            returnVal = self.phiReduction()
            
    def phiReduction(self):
        subGraphList = self.getValidSubGraphs(stichSubGraphs=True)
        generationalSets = self.getGenerationalSets()
        setToMerge = ''
        if isinstance(subGraphList, list):
            graphToMergePos = -99
            smallestPhi = -99
            for i in range(len(subGraphList)-1):
                testGraph = self.deepCopy()
                keyList = [key for key in subGraphList[i].nodes]
                if not self.willNgraphAppear(keyList):
                    print(keyList)
                    testGraph.mergeNodeIds(keyList)
                    # '0 <' Because work needs to be done with node cardinality
                    # The 'everything merge' returns a negative cardinality on normal to big regexes
                    # Maxwrapping
                    if (smallestPhi == -99 and 0 < testGraph.phi < self.phi) or 0 < testGraph.phi < smallestPhi:
                        smallestPhi = testGraph.phi
                        setToMerge = "SGL"
                        graphToMergePos = i
            
            for i in range(len(generationalSets)):
                testGraph = self.deepCopy()
                nodeIdList = [element for element in generationalSets[i]]
                if len(nodeIdList) > 1:
                    if not self.willNgraphAppear(nodeIdList):
                        testGraph.mergeNodeIds(nodeIdList)
                        # '0 <' Because work needs to be done with node cardinality
                        # The 'everything merge' returns a negative cardinality on normal to big regexes maxwrapping
                        if (smallestPhi == -99 and 0 < testGraph.phi < self.phi) or 0 < testGraph.phi < smallestPhi:
                            smallestPhi = testGraph.phi
                            setToMerge = "GS"
                            graphToMergePos = i
                
            
            testGraph = self.deepCopy()
            testGraph.squishAllGenerationSets()
            if (smallestPhi == -99 and 0 < testGraph.phi < self.phi) or 0 < testGraph.phi < smallestPhi:
                setToMerge = 'squish'
            
            if graphToMergePos != -99:
                if setToMerge == "GS":
                    self.mergeNodeIds([element for element in generationalSets[graphToMergePos]])
                elif setToMerge == 'SGL':
                    self.mergeNodeIds([key for key in subGraphList[graphToMergePos].nodes])
                else:
                    self.squishAllGenerationSets()
            
            return graphToMergePos != -99
        
    def squishAllGenerationSets(self):
        for genSet in self.getGenerationalSets():
            self.mergeNodeIds([element for element in genSet])
                

    def partition(self):
        """Partitions the graph recursively into either sequential or parallel graphs. Should be run after instantiating a graph via regex.
        Potent method.

        Raises:
            Exception: Method will not run if an N-graph is present
        """
                
        if self.doesNgraphExist:
            raise Exception("Graph is invalid, cointains N-graph")

        self.removeStraightShots()
        self.simplify()
        
        if len(self.nodes) == 1:
            self.sequentialGraphs = self.parallelGraphs = None
            return

        else:
            self.parallelPartition()
            if self.parallelGraphs:  # is not none
                [g.partition() for g in self.parallelGraphs]
            else:
                self.sequentialPartition()
                [g.partition() for g in self.sequentialGraphs]

    def sequentialPartition(self):
        """Partitions the graph in a sequential manner if sequentialGraphs are present
        """        

        self.sequentialGraphs:list[GRAPH] = []
        cutSets = self.getCutSets()

        for nodeSet in cutSets:
            self.sequentialGraphs.append(self.createSubGraph(nodeSet))
        if len(self.sequentialGraphs) == 1:
            self.sequentialGraphs = None
        return
    
    def parallelPartition(self):
        """Partitions the graph in a parallel manner if parallelGraphs are present
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
        """Reduces each node in the graph, i.e. abc -> a-c
        """        
        for node in self.nodes.values():
            node.reduce()

    def createVector(self):
        """Creates a vector for the given Graph.
            Raises:
        Exception: No regex or nodes present
        """
        if not self.regex and not self.nodes:
            raise Exception("No regex or nodes")
        regex = self.outRegex
        vector = np.zeros(128, dtype=int)

        if len(regex) == 1:
            vector[ord(regex)] = 1

        elif regex == '\d':
            vector[48:58] = 1

        elif regex == '\D':
            vector[31:47] = 1
            vector[59:127] = 1

        elif regex == '\w':
            vector[48:58] = 1
            vector[65:91] = 1
            vector[95] = 1
            vector[97:123] = 1

        elif regex == '\W':
            vector[31:48] = 1
            vector[58:65] = 1
            vector[91:95] = 1
            vector[96] = 1
            vector[123:127] = 1

        elif '[' in regex:
            pieces = partitionClass(regex)
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

    def checkAlternatePath(self, upStreamNode:str, nodeInQuestion:str, recursionDepth:int):
        """Checks for any alternate path besides the direct edge that connects two nodes

        Args:
            upStreamNode (str): Node id of the upstream node
            nodeInQuestion (str): Node id of the node in question
            recursionDepth (int): How many times you have recursed into this function

        Returns:
            bool: True or false depending on whether an alternate path exists
        """        
        if not self.getChildren(upStreamNode):
            return False
        else:
            results = []
            for child in self.getChildren(upStreamNode):
                if child == nodeInQuestion:
                    if recursionDepth:
                        return True
                    else:
                        results.append(True)
                        continue
                results.append(self.checkAlternatePath(child, nodeInQuestion, recursionDepth+1))
            
            trueCheck = -1
            for result in results:
                if result:
                    if recursionDepth:
                        return True
                    trueCheck += 1
                    if trueCheck:
                        return True
            return False
        
    def checkAlternatePathMultiNodesQuestion(self, upStreamNode:str, nodesInQuestion:list[str], recursionDepth:int):
        """Checks for any alternate path besides the direct edge that connects two nodes

        Args:
            upStreamNode (str): Node id of the upstream node
            nodesInQuestion (list[str]):List of node ids of the nodes in question
            recursionDepth (int): How many times you have recursed into this function


        Returns:
            bool: True or false depending on whether an alternate path exists
        """        
        if not self.getChildren(upStreamNode):
            return False
        else:
            results = []
            for child in self.getChildren(upStreamNode):
                if child in nodesInQuestion:
                    if recursionDepth:
                        return True
                    else:
                        results.append(True)
                        continue
                results.append(self.checkAlternatePathMultiNodesQuestion(child, nodesInQuestion, recursionDepth+1))
            
            trueCheck = -1
            for result in results:
                if result:
                    if recursionDepth:
                        return True
                    trueCheck += 1
                    if trueCheck:
                        return True
            return False
        
    def checkAlternatePathMultiNodesUpStream(self, upStreamNodes:list[str], nodeInQuestion:str, recursionDepth:int):
        """Checks for any alternate path besides the direct edge that connects two nodes

        Args:
            upStreamNode (list[str]): Node ids of the upstream nodes
            nodesInQuestion (str):List of node ids of the nodes in question
            recursionDepth (int): How many times you have recursed into this function

        Returns:
            bool: True or false depending on whether an alternate path exists
        """        
        childrenOfUpStream = []
        for node in upStreamNodes:
            nodeChildren = self.getChildren(node)
            if nodeChildren:
                childrenOfUpStream.extend(nodeChildren)
        results = []
        for child in childrenOfUpStream:
            if child == nodeInQuestion:
                if recursionDepth:
                    return True
                else:
                    results.append(True)
                    continue
            results.append(self.checkAlternatePathMultiNodesUpStream([child], nodeInQuestion, recursionDepth+1))
        
        trueCheck = -1
        for result in results:
            if result:
                if recursionDepth:
                        return True
                trueCheck += 1
                if trueCheck:
                    return True
        return False
    
    def checkAlternatePathMultiNodes(self, upStreamNodes:list[str], nodesInQuestion:list[str], recursionDepth:int):
        """Checks for any alternate path besides the direct edge that connects two nodes

        Args:
            upStreamNode (list[str]): Node ids of the upstream nodes
            nodesInQuestion (list[str]):List of node ids of the nodes in question
            recursionDepth (int): How many times you have recursed into this function

        Returns:
            bool: True or false depending on whether an alternate path exists
        """        
        childrenOfUpStream = []
        for node in upStreamNodes:
            nodeChildren = self.getChildren(node)
            if nodeChildren:
                childrenOfUpStream.extend(nodeChildren)
        results = []
        for child in childrenOfUpStream:
            if child in nodesInQuestion:
                if recursionDepth:
                    return True
                else:
                    results.append(True)
                    continue
            results.append(self.checkAlternatePathMultiNodes([child], nodesInQuestion, recursionDepth+1))
        
        trueCheck = -1
        for result in results:
            if result:
                if recursionDepth:
                        return True
                trueCheck += 1
                if trueCheck:
                    return True
        return False

    @property
    def doesNgraphExist(self):
        """Checks whether an N-graph is already present in a graph

        Returns:
            bool: True or False on whether an n-graphs exists
        """        
        for child in self.nodes.keys():
            parentsList = self.getParents(child)
            if len(parentsList) > 1:
                for parent in parentsList:
                    childrenList = self.getChildren(parent)
                    if len(childrenList) > 1: # If the node has multiple parents, and that parent has multiple children
                            # Check if it's a straight shot (exeption to the multiple children rule)
                        result = self.checkAlternatePath(parent, child, 0)
                        if not result:
                            return True
        return False

    @property
    def doesStraightShotExist(self):
        """Checks the current graph for any straight shots

        Returns:
            bool: Whether the straight shot exists or not
        """   
        for node in self.nodes.keys():
            nodeChildren = self.getChildren(node)
            if nodeChildren:
                if len(nodeChildren) > 1:
                    for kid in nodeChildren:
                        kidDescendants = self.getNodeDescendants(kid)
                        descendantChildrenIntersection = list(set(kidDescendants).intersection(nodeChildren))
                        if descendantChildrenIntersection:
                            return self.getEdge(node, descendantChildrenIntersection[0])
        return False

    @property
    def cardinality(self):
        """Returns the cardinality, how many strings are able to satisfy the regex of the graph.

        Returns:
            int: The cardinality of the graph
        """        
        if hasattr(self, 'parallelGraphs') and self.parallelGraphs:
            return sum([g.cardinality for g in self.parallelGraphs])
        elif hasattr(self, 'sequentialGraphs') and self.sequentialGraphs:
            return int(np.prod([g.cardinality for g in self.sequentialGraphs]))
        else:
            k = list(self.nodes.keys())[0]
            return self.nodes[k].cardinality

    @property
    def log2Cardinality(self):
        """Helper method for calculating the base 2 logarithm for any size cardinality

        Returns:
            float: base2 logarithm of the cardinality
        """        
        if hasattr(self, 'parallelGraphs') and self.parallelGraphs:
            return np.log2(sum([g.cardinality for g in self.parallelGraphs]))
        elif hasattr(self, 'sequentialGraphs') and self.sequentialGraphs:
            return round(sum([np.log2(g.cardinality) for g in self.sequentialGraphs]), 4)
        else:
            k = list(self.nodes.keys())[0]
            return self.nodes[k].cardinality

    @property
    def entropy(self):
        """Returns the Information-Theoretic entropy of the given graph's regex.

        Returns:
            float: The entropy of the graph's regex
        """        
        return round(np.log2(self.cardinality), 4)

    @property
    def K(self):
        """Returns the length of the given graph's regex.

        Returns:
            int: The graph's regex length
        """        
        return len(self.outRegex)

    def outRegexRecursive(self):
        """Returns the joined regexes of each individual subNode,
        based on parallel or sequential relationship.

        Returns:
            str: (Recursive return) Returns the outRegex
        """        
        if hasattr(self, 'parallelGraphs') and self.parallelGraphs:
            # if hasattr(self, 'parallelGraphs') and self.parallelGraphs is not None:
            toReturn = "(" + "|".join([g.outRegexRecursive() for g in self.parallelGraphs]) + ")"
        elif hasattr(self, 'sequentialGraphs') and self.sequentialGraphs:
            # elif hasattr(self, 'sequentialGraphs') and self.sequentialGraphs is not None:
            toReturn = "".join([g.outRegexRecursive() for g in self.sequentialGraphs])
        else:
            k = list(self.nodes.keys())[0]
            toReturn = self.nodes[k].regex
        return toReturn

    @property
    def outRegex(self):
        """Returns the regex of the given graph based on node relationships

        Returns:
            str: The graph's regex
        """        
        toReturn = self.outRegexRecursive()
        if hasOuterParentheses(toReturn):
            return toReturn[1:-1]
        else:
            return toReturn

    @property
    def phi(self):
        """Returns the phi of the graph, being the entropy of the graph added to the product
        of the alpha (weight parameter) and its K value.

        Returns:
            float: The graph's phi value
        """        
        return round(self.log2Cardinality, 4) + self.alpha * self.K

    @property
    def random(self):
        """Returns a random string that satisfies its regex

        Returns:
            str: A string that satisfies the regex
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
