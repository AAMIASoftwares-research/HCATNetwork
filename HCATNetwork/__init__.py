"""HCATNetwork
Heart Coronary Artery Tree Network is a library based on NetworkX
which defines standard data structures for nodes, edges, and graphs
(or better, it defines their attributes) for standardardised use across multiple project.
It is kinda like a header file with some added functionality, such as graph saving and reading from file.

Access stuff with (example):
    import HCATNetwork
    print(HCATNetwork.edge.BasicEdge_KeysList)

As of now, supported node, edge and graph features include, beside the standard data-types supported by NetworkX and GML file format:
- nested lists of whatever complexity
- numpy's n-dimensional arrays
Note that, for these eobjects, the holded data type, if numerical, will be automatically converted to "float".
Conversion makes use of the "json" package. See "graph" module for more.

Future improvements:
Use TypedDict: https://peps.python.org/pep-0589/ , from typing import TypedDict
    In this was, data retrieval from files will be much more precise.
"""

welcome = "Welcome to HCATNetwork - Heart Coronary Artery Tree Network"

# When the module gets imported
from . import node, edge, graph, draw
__all__ = ["graph", "edge", "node", "draw"]
