# regexfinder

## Introduction

The regexfinder package offers to create and manipulate regular expressions. In particular, the package can:
- determine a regular expression given a set of words
- optimize a size/complexity tradeoff for a regular expression


## Size/complexity tradeoff

A regular expression defines a set of strings. Our general intuition is that we want the regex to include as few strings as possible (i.e. be as small as possible) but also not be too long or complex. Regexfinder formalizes this tradeoff by using two ideas from information theory. The size of the regular expression is the [information theoretic entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) of a distribution on the regular expression (we start with the uniform distribution). And the complexity of the regular expression is the length of the length of the regular expressionwhich is motivated by [Kolmogorov Complexity](https://en.wikipedia.org/wiki/Kolmogorov_complexity). 

Using a weight parameter &alpha;


## Graph Representation

The tool creates a graph representation of a regular expression. For example, the regular expression 
```
r'\d{2}([a-m]{2}[n-z]{2}|[A-Z]{4})\d'
```
is represented as:
![](/images/exampleGraph1.png)
