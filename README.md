# regexfinder

## Introduction

Regexfinder is a set of tools to create and manipulate regular expressions. In particular, the package can:
- determine a regular expression given a set of strings
- optimize a size/complexity tradeoff for a regular expression
- visualize a regular expression as a graph


## Size/complexity tradeoff

A regular expression defines a set of strings. Our general intuition is that we want the regex to include as few strings as possible (i.e. be as small as possible) but also not be too long or complex. Regexfinder formalizes this tradeoff by using two ideas from information theory. The size of the regular expression is the [information theoretic entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) of a distribution on the regular expression (we start with the uniform distribution). And the complexity of the regular expression is the length of the regular expression which is motivated by [Kolmogorov Complexity](https://en.wikipedia.org/wiki/Kolmogorov_complexity). 

Using a weight parameter *&alpha;*, *ent* for entropy and *K* for complexity, regexfinder minimizes

![](/images/equation1.png)

over regular expressions that contain a provided set of strings.

For example, if the set of strings all have length 1 and contain the digits 0,1,2,4,5,7,8,9, then an admissible regular expression is RE1 = [01245789]. If &alpha; = 1, then 

&phi;(RE1) = log2(8) + 10 = 13. 

Another regular expression that contains all of these strings is RE2 = \d, and 

&phi;(RE2) = log2(10) + 2 = 5.52. 

So, in this case, \d would be chosen over [01245789].

## Graph Representation

The tool creates a graph representation of a regular expression. For example, the regular expression 
```
r'\d{2}([a-m]{2}[n-z]{2}|[A-Z]{4})\d'
```
is represented as:

![](/images/exampleGraph1.png)




