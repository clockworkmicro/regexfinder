import re
from functools import reduce


def randFactors(n):    
    factorNum = 50
    while factorNum > 1:
        if n%factorNum == 0:
            return factorNum
        else:
            factorNum+=1

def strNumberGenerator():
     number = 0
     while True:
         yield str(number)
         number += 1
         
strCounter = strNumberGenerator()

def topOrExists(inString):
    topORs = getTopOR(inString)
    return bool(topORs)

def getTopOR(inString):
    # find or symbols '|' that are not contained within any parentheses
    matches = re.finditer(r"\|" ,inString)
    matches = list(matches)
    orLocations = []
    for match in matches:
        orLocations.append(match.start())
    
    p,c = outerParentheses(inString)
    inParentheses = [list(range(x[0],x[1])) for x in p]
    topOrs = [x for x in orLocations if not any([x in r for r in inParentheses])]
    return topOrs

def getTopOrSegments(inString):
    topORs = getTopOR(inString)
    if not topORs:
        return [inString]
    else:
        t1 = [0] + [x +1 for x in topORs]
        t2 = topORs + [len(inString)]
        endPoints = list(zip(t1,t2))
        return [inString[x:y] for x,y in endPoints]
    
def outerParentheses(inString):
    v = 0
    out = []
    for s in inString:
        if s == '(':
            v += 1
        elif s == ')':
            v -= 1
        else:
            pass
        out.append(v)
    starts = [i + 1 for i in range(len(out)-1) if (out[i]==0 and out[i+1]==1)]

    if out[0] == 1:
        starts = [0] + starts
        
    ends = [i +1 for i in range(1,len(out)) if (out[i-1]==1 and out[i]==0)]

    t1 = [0] + ends
    t2 = starts + [len(inString)]
    complements = list(zip(t1,t2))
    #print('starts:' ,starts)
    #print('ends: ',ends)
    if not len(starts) == len(ends):
       raise Exception('len(starts) does not equal len(ends)')
    parentheses = list(zip(starts,ends))
    
    parentheses = [p for p in parentheses if p[0]<p[1]]
    complements = [c for c in complements if c[0]<c[1]]
    return parentheses,complements

def hasOuterParentheses(inString):
    if inString[0] == '(':
        for s in range(len(inString)-1):
            if inString[s] == ')' and s == len(inString)-1:
                return True
    return False
    

def getParenthesesSegments(inString):
    p,c = outerParentheses(inString)
    r = p+c
    r.sort(key = lambda x:x[1])
    segments = [inString[x[0]:x[1]] for x in r]
    segments = [removeOuterParenthesesOrBrackets(s) for s in segments]
    return segments

def removeOuterParenthesesOrBrackets(inString):
    if (inString[0] == '(' and inString[-1] == ')') or (inString[0] == '[' and inString[-1] == ']') :
        return inString[1:-1]
    else:
        return inString
    
def getClassQuantList(inString):
    '''
    Returns a list of quantifier data, being 'class', 'quantifier', 'min' and 'max'.
    '''
    matches = re.finditer(r"(\[[^]]+\]|\\d|\\w|\\W|.)(\?|\+|\*|\{\d?,?\d?\})?" ,inString)
    toReturn = []
    for match in matches:
        groups = list(match.groups())
        if not len(groups) == 2:
            raise Exception('Expected 2 groups. Received ',len(groups))
        

        entry = {'class':groups[0]}
        if groups[1]:         
            if groups[1] != '{1}':   
                entry['quantifier'] = groups[1]
                matches2 = re.finditer(r"\d+" ,groups[1])
                vals = [match2.group() for match2 in matches2]
        
                if not vals:
                    entry['min'] = None
                    entry['max'] = None
                elif len(vals) == 1:
                    entry['min'] = int(vals[0])
                    entry['max'] = int(vals[0])
                elif len(vals) == 2:
                    if int(vals[0]) > int(vals[1]):
                        raise Exception("Min and Max quantifier values are in the wrong place")
                    entry['min'] = int(vals[0])
                    entry['max'] = int(vals[1])
            else:
                entry['quantifier'] = None
                entry['min'] = None
                entry['max'] = None
        else:
            entry['quantifier'] = None
            entry['min'] = None
            entry['max'] = None
            
        toReturn.append(entry)
    return toReturn


def setClassQuantList(inString:str, quantifier):
    '''
    Returns a list of quantifier data, being 'class', 'quantifier', 'min' and 'max', built from a given quantifier,
    as well as an updated inString to couple with the new quantifier.
    Quantifier must be an integer or string containing integers. Quantifier must have a bounded maximum.
    '''
    if isinstance(quantifier, int):
        if quantifier < 1:
            raise Exception('Quantifier must be greater than 0')
    elif isinstance(quantifier, str):
        quantifier = removeOuterParenthesesOrBrackets(quantifier)
        if not quantifier:
            raise Exception('Quantifier field is empty')
        elif not any(re.match('\d', char) for char in [*quantifier]):
            raise Exception('Invalid Quantifier input')
        elif isinstance(quantifier, tuple):
            quantifier = str(quantifier)
    else:
        raise Exception('Invalid quantifier input')
        
    matches = re.finditer(r"(\[[^]]+\]|\\d|\\w|\\W|.)(\?|\+|\*|\{\d?,?\d?\})?" ,inString)
    toReturn = []
    for match in matches:
        groups = list(match.groups())
        if not len(groups) == 2:
            raise Exception('Expected 2 groups. Received ',len(groups))

        entry = {'class':groups[0]}
        quantIsOne = False
        if quantifier == 1:
            quantIsOne = True

        if quantIsOne:
            entry['quantifier'] = None
        else:
            entry['quantifier'] = '{' + str(quantifier) + '}'
            # entry['quantifier'] = quantifier

        if isinstance(quantifier, int):
            if quantIsOne:
                entry['min'] = None
                entry['max'] = None
            else:
                entry['min'] = quantifier
                entry['max'] = quantifier
        else:
            quantifierCharacters = [*quantifier]
            if not re.match('\d', quantifierCharacters[0]):
                entry['quantifier'] = '{' + removeOuterParenthesesOrBrackets(entry['quantifier'][1:-1]) + '}'
                quantifier = removeOuterParenthesesOrBrackets(quantifier)
                quantifierCharacters = [*quantifier]
            
            if len(quantifierCharacters) == 1:
                quant = int(quantifierCharacters[0])
                entry['max'] = quant
                entry['min'] = quant
            elif quantifierCharacters[0] == ',':
                entry['min'] = 0
                max = quantifierCharacters[1]
                i = 2
                while i < len(quantifierCharacters):
                    max += quantifierCharacters[i]
                    i+=1
                max = int(max)
                entry['max'] = max
            elif quantifierCharacters[len(quantifierCharacters)-1] == ',':
                raise Exception("Quantifier must have a finite/bounded maximum")
            else:
                min = quantifierCharacters[0]
                i = 1
                while not quantifierCharacters[i] == ',':
                    min += quantifierCharacters[i]
                    i+=1
                min = int(min)
                entry['min'] = min
                i+=1
                max = quantifierCharacters[i]
                i+=1
                properLen = len(quantifierCharacters)
                while i < properLen:
                    max += quantifierCharacters[i]
                    i+=1
                max = int(max)
                entry['max'] = max
                
        if entry['quantifier'] != None:
            updatedInString = inString + entry['quantifier']
        else:
            updatedInString = inString
        toReturn.append(entry)
    return [toReturn, updatedInString]


def partitionClass(inString):
    # partitions a class , i.e. [....] into pieces
    #20220630
    #matches = re.finditer(r"([^\[\]]-[^\[\]])|[^\[\]]",inString)
    matches = re.finditer(r"(\\d|\\D|\\w|\\W)|([^\[\]]-[^\[\]])|(?<!-)[^-\[\]](?!-)",inString)
    matches = list(matches)
    return [match.group() for match in matches]

def flatten(x):
    return [y for z in x for y in z]      
