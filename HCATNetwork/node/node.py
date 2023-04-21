"""Node
This file defines standard data structures for node attributes/features.
In this context, "node attribute" is any object associated (contained) in a node, while node features implies
that each node attribute is a single float, a sort of "unwinding" of node attributes.
A node feature must be such that it can be then encoded, along all other features, in a feature matrix.
Here we define many standard dictionaries used in the context of Heart Coronary Artery Tree mapping.
Depending on the kind of node, a node can have some attributes. Each attribute is defined as a dictionary entry.
The defined dictionaries are meant to be used together with NetworkX, which means that NetworkX must accept the
here-defined dictionaries as node features.

IMPORTANT:
Node ids MUST be strings when creating nodes for NetworkX. They can be floats or any hashable python object, but in reading/writing for
saving the file, all gets lost. To ensure continuity between a graph created on the fly and another one
opened from a file, use strings (even better if str(i), where i is an integer).
"""
import itertools
from enum import Enum, auto
import numpy
from ..core.core import CoreDict

####################
# Common utilities
####################


##############
# Vertex Node
##############
"""
A vertex node is a node defined by just its label and x, y, z positions
"""
class VertexNode(CoreDict):
    x: float
    y: float 
    z: float

    def getVertexList(self) -> list[float]:
        return [self.__getitem__("x"), self.__getitem__("y"), self.__getitem__("z")]

    def getVertexNumpyArray(self) -> numpy.ndarray:
        return numpy.array(self.getVertexList())

    def setVertex(self, v: float | list | numpy.ndarray) -> None:
        if isinstance(v, float):
            v = [v]
        if len(v) == 0 or len(v) > 3:
            raise RuntimeError(f"Unsupported input vertex length: {len(v)}")
        if isinstance(v, numpy.ndarray):
            v = v.flatten()
        for key, new_val in itertools.zip_longest(["x", "y", "z"], v, fillvalue=0.0):
            self.__setitem__(key, float(new_val))
        


##########################################
# Other intermidiate nodes could be built
##########################################


#############################
# Simple HCA Centerline Node
#############################
"""
This node stores just the most basic information about the geometric centerline,
with no added complexity.
- "topology_class": An enum from the list ["o", "s", "i", "e"]:
    o: coronary ostium/starting point of the left or right tree
    s: segment (a point with 2 connections)
    i: intersection (a point with more than two connections)
    e: endpoint
- x, y, z, t, r: The cartesian and temporal coordinates of the node, as well as
    r, which is the radius of the circle with area equivalent to the area of the coronary lumen at that point.
- "tree" must be one of the string literals defined in the following enum: ["r", "l", "b"], where "b" stands for "both"
    (there are some heart structures in which the coronary arteries from left and right side branches merge together)
"""
class ArteryPointTopologyClass(Enum):
    OSTIUM = auto()
    SEGMENT = auto()
    INTERSECTION = auto()
    ENDPOINT = auto()

class ArteryPointTree(Enum):
    RIGHT = auto()
    LEFT = auto()
    RL = auto()

class SimpleCenterlineNode(VertexNode):
    topology_class: ArteryPointTopologyClass
    t: float
    r: float
    arterial_tree: ArteryPointTree

    def getVertexRadiusList(self):
        return self.getVertexList().extend(self.__getitem__("r"))

    def getVertexRadiusNumpyArray(self):
        return numpy.array(self.getVertexRadiusList())

    def setVertexRadius(self, v: float | list | numpy.ndarray):
        if isinstance(v, float):
            v = [v]
        if len(v) == 0 or len(v) > 4:
            raise RuntimeError(f"Unsupported input vertex length: {len(v)}")
        if isinstance(v, numpy.ndarray):
            v = v.flatten()
        for key, new_val in itertools.zip_longest(["x", "y", "z", "r"], v, fillvalue=0.0):
            self.__setitem__(key, float(new_val))



############
# HCAT Node
############
"""
This is the "most complete" node, with everything that is needed and that might be needed in the future.
This is the only node actively maintained and that will be used in the future.
"""

class HeartCoronaryArteryNode(CoreDict): # ["this is the most complete dict you can think of"]
    everything: any

if __name__ == "__main__":
    print("Running 'HCATNetwork.node' module")
    d = SimpleCenterlineNode()
    print(d)

    d["arterial_tree"] = ArteryPointTree.RIGHT
    print(d, d["arterial_tree"].value)