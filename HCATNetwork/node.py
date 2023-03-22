"""
In this file it is defined a series of standard nodes, each defined as a data structure.
"""
import numpy
import networkx

class Node(object):
    ID: int
    def __init__(self, id: int):
        self.ID = id

    def __hash__(self):
        # https://docs.python.org/3/glossary.html#term-hashable
        return self.ID

class NodeLabelled(Node):
    label: str

class NodeVertex3(NodeLabelled):
    vertex: numpy.ndarray(shape=(3,1), dtype=numpy.float32)
    vertex_projective: numpy.ndarray(shape=(4,1), dtype=numpy.float32)

class NodeTimedVertex3(NodeVertex3):
    t: float

class NodeVertex3Radius(NodeVertex3):
    radius: float

class NodeArtery(NodeTimedVertex3, NodeVertex3Radius):
    branch: enumerate()## "l", "r", "lr"
     