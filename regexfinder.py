import copy
import numpy as np
import pandas as pd
import re
from graphviz import Digraph
from itertools import combinations
import more_itertools as mit
import random

from utils import *


class NODE:
    def __init__(self, regex=None, vector=None, replaced=False, simple=False, alpha=1, quantifier=None):

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
        return not ('(' in self.regex or '|' in self.regex or len(self.classQuantList) > 1)

    @property
    def getQuantifier(self):
        if self.simple:
            return self.classQuantList[0]['quantifier']
        else:
            return False
        
    @property
    def getQuantifierMin(self):
        if self.simple:
            return self.classQuantList[0]['min']
        else:
            return False    
        
    @property
    def getQuantifierMax(self):
        if self.simple:
            return self.classQuantList[0]['max']
        else:
            return False
        
    def mergeQuantifiers(self, nodeList):
        nodeList.append(self)
        lowestLow = min(lowestMin.getQuantifierMin for lowestMin in nodeList)
        highestHigh = max(highestMax.getQuantifierMax for highestMax in nodeList)
        
        if lowestLow == highestHigh:
            return lowestLow
        else:
            return (lowestLow, highestHigh)
        

    @property
    def topOrExists(self):
        return topOrExists(self.regex)

    def removeOuterParentheses(self):
        while self.regex[0] == '(' and self.regex[-1] == ')':
            self.regex = self.regex[1:-1]

    def createVector(self):

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
        if not self.simple:
            raise Exception('Node is not simple. A node cannot be reduced if it is not simple.')
        else:
            if not self.vector:
                self.createVector()
            else:
                pass
            self.vector.reduce()
            self.regex = self.vector.regex
            self.reduced = True

    @property
    def cardinality(self):
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
        return self.getQuantifierMin == self.getQuantifierMax

    @property
    def entropy(self):
        return round(np.log2(self.cardinality), 4)

    @property
    def K(self):
        return len(self.regex)

    @property
    def phi(self):
        return self.entropy + self.alpha * self.K

    def match(self, inString, boolean=False):
        matches = re.finditer(self.regex, inString)
        if boolean:
            return bool(list(matches))
        else:
            return matches

    @property
    def random(self):
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
        if words is None:
            words = []
        self.parent = parent
        self.child = child
        self.words = words


class VECTOR:
    def __init__(self, vector, alpha=None):
        self.v = vector
        self.alpha = alpha

    @property
    def regex(self):

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
        return [list(group) for group in mit.consecutive_groups(self.support)]

    @property
    def minIndex(self):
        return np.min(np.where(self.v))

    @property
    def maxIndex(self):
        return np.max(np.where(self.v))

    @property
    def ent(self):
        return round(np.log2(np.sum(self.v)), 4)

    @property
    def support(self):
        return np.where(self.v)[0]

    @property
    def K(self):
        return len(self.regex)

    @property
    def phi(self):
        return self.ent + self.alpha * self.K

    @property
    def random(self):
        return chr(random.sample(list(np.where(self.v)[0]), 1)[0])


class ALPHABET:
    def __init__(self, alphabetList):
        self.alphabetList = alphabetList

    def sample(self, count=1):
        return ''.join(random.choices(self.alphabetList, k=count))


class WORDCOLLECTION:
    def __init__(self, words):
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
        # returns a dictionary grouping the input dictionary keys by same value
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
        self.regex = regex
        self.wordList = wordList
        self.alpha = alpha

        self.nodes = {}
        self.edges = []

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
            self.edges = edges

        else:
            pass

        if self.regex:
            self.startNode = NODE(self.regex)
            self.nodes.update({self.startNode.id_: self.startNode})
        else:
            pass

    def deepCopy(self):
        return copy.deepcopy(self)


    def addNode(self, node, edges=[]):
        if node.id_ in self.nodes.keys():
            raise Exception('Node key already exists')
        
        self.nodes[node.id_] = node
        
        if edges:
            for currEdge in edges:
                self.edges.append(currEdge)
        else:
            pass
        
        
        
        
    def removeNodeAndEdges(self, nodeList):
        # upperEdges = []
        # lowerEdges = []
        # nodeIdList = [node.id_ for node in nodeList]
        
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
        
        # [self.edges.remove(x) for x in edgesToRemove]
        for edge in edgesToRemove:
            self.edges.remove(edge)
        
        for node in nodeList:
            del self.nodes[node]
        
                
        
        return [upperAffectedNodes, lowerAffectedNodes]
    

    def addEdge(self, edge):
        for gEdge in self.edges:
            if edge == gEdge:
                raise Exception("Edge already exists")
        self.edges.append(edge)

    def getEdge(self, parent, child):
        matches = [edge for edge in self.edges if (edge.parent == parent and edge.child == child)]
        if len(matches) > 1:
            raise Exception('More than one edge with same parent/child.')
        elif len(matches) == 1:
            return matches[0]
        else:
            return False

    def removeEdge(self, parent, child):
        toRemove = [edge for edge in self.edges if (edge.parent == parent and edge.child == child)]

        if toRemove:
            [self.edges.remove(edge) for edge in toRemove]
    
    def removeEdgesList(self, edgeList):
        [self.edges.remove(edge) for edge in edgeList]

    def getParents(self, id_):
        return [x.parent for x in self.edges if x.child == id_]

    def getChildren(self, id_):
        return [x.child for x in self.edges if x.parent == id_]

    def getSiblings(self, id_):
        if not self.getParents(id_):
            return self.getNodesNoParents()
        else:
            return sorted(list(set(flatten([self.getChildren(parent) for parent in self.getParents(id_)]))))

    def getNodesNoChildren(self):
        return [x.id_ for x in self.nodes.values() if not self.getChildren(x.id_)]

    def getNodesNoParents(self):
        return [x.id_ for x in self.nodes.values() if not self.getParents(x.id_)]
    
    def getLonelyNodes(self):
        setOfNoParents = set(self.getNodesNoParents())
        setOfNoChildren = set(self.getNodesNoChildren())
        
        return list(setOfNoChildren.intersection(setOfNoParents))

    def getNotSimple(self):
        return [id_ for id_, node in self.nodes.items() if (not node.replaced and not node.simple)]

    # Should only run if a graph is simplified
    def partition(self):

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

    def simplify(self):
        while self.getNotSimple():
            self.process(self.getNotSimple()[0])
        self.nodes = dict([(name, node) for name, node in self.nodes.items() if node.simple])

    def getNodeEqClasses(self):
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
        toReturn = []
        for id_ in inList:
            toReturn += self.getNodeAncestors(id_)
        return toReturn

    def getNodeDescendantsList(self, inList):
        toReturn = []
        for id_ in inList:
            toReturn += self.getNodeDescendants(id_)
        toReturn = list(set(toReturn))
        return toReturn

    def getNodeAncestors(self, id_):
        ancestors = []
        parents = self.getParents(id_)
        while parents:
            parent = parents.pop()
            parents += self.getParents(parent)
            ancestors.append(parent)
        return sorted(list(set(ancestors)))

    def getNodeDescendants(self, id_):
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
        eqClasses = self.getNodeEqClasses()
        return [sorted(eqClass) for eqClass in eqClasses if self.testCutSet(eqClass)]

    def getNextCutSet(self, id_):
        if not id_ in self.nodes.keys():
            raise Exception('Node id not not valid.')
        children = self.getChildren(id_)
        while not (children in self.getCutSets()) and children:
            newchildren = self.getChildren(children[0])
            children = newchildren
        return children

    def getNodesBetweenCutSets(self, cutSetUpper, cutSetLower):

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

        # for id_ in self.getNodesNoParents():
        #   dot.edge('',id_)

        for edge in self.edges:
            dot.edge(edge.parent, edge.child)

        dot.render(view=True)

    def process(self, id_):

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

            n = NODE(regex=toString(cQList[0]), simple=True)
            self.addNode(n)
            previous = n.id_
            for parent in parents:
                self.addEdge(EDGE(parent, n.id_))
                self.removeEdge(parent, self.nodes[id_].id_)
            for cQ in cQList[1:]:
                n = NODE(regex=toString(cQ), simple=True)
                self.addNode(n)
                self.addEdge(EDGE(previous, n.id_))
                previous = n.id_

            children = self.getChildren(self.nodes[id_].id_)
            for child in children:
                self.removeEdge(self.nodes[id_].id_, child)
                self.addEdge(EDGE(previous, child))

    def getSharedDecendantSets(self):

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


    def isMergeNodesValid(self, nodeList):
        
        if self.getLonelyNodes():
            if any(node in self.getLonelyNodes() for node in nodeList):
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
                    nodeAncestorsList.append(self.getNodeAncestors(n))

            for currNodeDescItem in currNodeDescendants:
                if currNodeDescItem not in nodeDescendantsList:
                    nodeDescendantsList.append(self.getNodeDescendants(n))

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
        
                # for intersectedNode in intersectTABD:
        #     if intersectedNode not in nodeList:
        #         return False
            

        # # list comprehension, use any()
        # for intersectedNode in intersectTDBA:
        #     if intersectedNode not in nodeList:
        #         return False
        
        # return True

        for topNode in topNodes:
            [topNodeAncestors.append(topNodeAncest) for topNodeAncest in self.getNodeAncestors(topNode) if topNodeAncest not in topNodeAncestors]

            [topNodeDescendants.append(topNodeDesc) for topNodeDesc in self.getNodeDescendants(topNode) if topNodeDesc not in topNodeDescendants]

        for bottomNode in bottomNodes:
            [bottomNodeAncestors.append(bottomNodeAncest) for bottomNodeAncest in self.getNodeAncestors(bottomNode) if bottomNodeAncest not in bottomNodeAncestors]

            [bottomNodeDescendants.append(bottomNodeDesc) for bottomNodeDesc in self.getNodeDescendants(bottomNode) if bottomNodeDesc not in bottomNodeDescendants]

        setOfTopNodeAncest = set(topNodeAncestors)
        setOfTopNodeDesc = set(topNodeDescendants)
        setOfBottomNodeAncest = set(bottomNodeAncestors)
        setOfbottomNodeDesc = set(bottomNodeDescendants)

        intersectTABD = list(setOfTopNodeAncest.intersection(setOfbottomNodeDesc))
        intersectTDBA = list(setOfTopNodeDesc.intersection(setOfBottomNodeAncest))

        if not (intersectTABD and intersectTDBA):
            return True
        else:
            areNodesInNodeList1 = any([node in intersectTABD for node in nodeList])
            areNodesInNodeList2 = any([node in intersectTDBA for node in nodeList])
            if not (areNodesInNodeList1 or areNodesInNodeList2):
                return False
            else:
                return True

    def createMergedNodes(self, nodeList, nodeRelationship):
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
        if any(self.nodes[nodeId].getQuantifier for nodeId in nodeList):
            lowestLow = min(self.nodes[nodeId].getQuantifierMin for nodeId in nodeList if self.nodes[nodeId].getQuantifierMin)
            highestHigh = max(self.nodes[nodeId].getQuantifierMax for nodeId in nodeList if self.nodes[nodeId].getQuantifierMax)
            
            if lowestLow == highestHigh:
                return lowestLow
            else:
                # I am making the return a string b/c
                # this return should be used to initialize a node,
                # which will run getClassQuantList and that code
                # handles a tuple as a string, for now
                return str((str(lowestLow) + "," + str(highestHigh)))
        else:
            return None
        
    def createMergedSequentialNodesQuantifier(self, nodeList):
        if any(self.nodes[nodeId].getQuantifier for nodeId in nodeList):
            lowestLows = [self.nodes[nodeId].getQuantifierMin for nodeId in nodeList if self.nodes[nodeId].getQuantifierMin]
            highestHighs = [self.nodes[nodeId].getQuantifierMax for nodeId in nodeList if self.nodes[nodeId].getQuantifierMax]
            
            sumLowestLows = sum(lowestLows)
            sumHighestHighs = sum(highestHighs)
            
            if sumLowestLows == sumHighestHighs:
                return sumHighestHighs
            else:
                return str((str(sumLowestLows) + "," + str(sumHighestHighs)))
        else:
            return len(nodeList)
        
        
    def mergeNodes(self, nodeList, nodeRelationship):
        if not set(nodeList).issubset(set(self.nodes.keys())):
            raise Exception('Node list includes invalid node.')
        
        if isinstance(nodeRelationship, str):
            if not ((nodeRelationship.lower() == "sequential") or (nodeRelationship.lower() == "parallel")):
                raise Exception("Node relationship is either 'sequential' or 'parallel'")
        else:
            raise Exception("Node relationship should be a string 'sequential' or 'parallel'")
    
        mergedNode = self.createMergedNodes(nodeList, nodeRelationship)
        
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
            raise Exception("nodeList was not able to be merged")

    def createSubGraph(self, nodeList):
        subNodes = {}
        subEdges = []

        for node in nodeList:
            subNodes[node] = self.nodes[node]
        subEdges = [edge for edge in self.edges if (edge.parent in nodeList and edge.child in nodeList)]

        subG = GRAPH(nodes=subNodes, edges=subEdges, alpha=self.alpha)
        return subG

    def sequentialPartition(self):
        sequentialGraphsNodes = []
        cutSets = self.getCutSets()
        noParents = self.getNodesNoParents()

        currentSet = noParents

        nextCutSet = self.getNextCutSet(currentSet[0])
        descendantList = self.getNodeDescendantsList(currentSet[0])
        if not nextCutSet and descendantList:
            if currentSet in cutSets:
                sequentialGraphsNodes.append(currentSet)
                sequentialGraphsNodes.append(descendantList)
            else:
                sequentialGraphsNodes.append(currentSet + descendantList)


        else:

            if currentSet in cutSets:
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
            for nodeSet in parallel:
                self.parallelGraphs = [self.createSubGraph(nodeSet) for nodeSet in parallel]

    @property
    def cardinality(self):
        if hasattr(self, 'parallelGraphs') and self.parallelGraphs is not None:
            return sum([g.cardinality for g in self.parallelGraphs])
        elif hasattr(self, 'sequentialGraphs') and self.sequentialGraphs is not None:
            return np.prod([g.cardinality for g in self.sequentialGraphs])
        else:
            k = list(self.nodes.keys())[0]
            return self.nodes[k].cardinality

    @property
    def entropy(self):
        return round(np.log2(self.cardinality), 4)

    @property
    def K(self):
        return len(self.outRegex)

    @property
    def outRegexRecursive(self):
        if hasattr(self, 'parallelGraphs') and self.parallelGraphs is not None:
            toReturn = "(" + "|".join([g.outRegexRecursive for g in self.parallelGraphs]) + ")"
        elif hasattr(self, 'sequentialGraphs') and self.sequentialGraphs is not None:
            toReturn = "".join([g.outRegexRecursive for g in self.sequentialGraphs])
        else:
            k = list(self.nodes.keys())[0]
            toReturn = self.nodes[k].regex
        return toReturn

    @property
    def outRegex(self):
        toReturn = self.outRegexRecursive
        if toReturn[0] == '(' and toReturn[-1] == ')':
            return toReturn[1:-1]
        else:
            return toReturn

    @property
    def phi(self):
        return round(np.log2(self.cardinality), 4) + self.alpha * self.K

    @property
    def random(self):
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
