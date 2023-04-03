"""
In this file it is defined a series of standard nodes, each defined as a data structure.

The intended use is to use them in the context of the NetworkX library (see intro: https://networkx.org/documentation/stable/reference/introduction.html).
To do so, each node must be a hashable object (more infos here: https://docs.python.org/3/glossary.html#term-hashable)
Objects which are instances of user-defined classes are hashable by default.
They all compare unequal (except with themselves), and their hash value is derived from their id().
The id(object_name) is an integer which is guaranteed to be unique and constant for this object during its lifetime.
Two objects with non-overlapping lifetimes may have the same id() value.
CPython implementation detail: This is the address of the object in memory.

The decorator dataclass is used to define nodes, as it has some useful functionalities
to it: https://www.dataquest.io/blog/how-to-use-python-data-classes/


"""
from dataclasses import dataclass
import sys
import time, random
import numpy

__NODES_ID_POOL__ = []

@dataclass
class Node(object):
    __ID: int
    def __get_id(self):
        return time.time_ns() #random.randint(a=0, b=int(2**63))
    
    def __get_unique_id(self):
        id = self.__get_id()
        while id in __NODES_ID_POOL__:
            id = self.__get_id()
        return id
            
    def __init__(self, id: int | None = None):
        self.__ID = id if ((id is not None) and (id not in __NODES_ID_POOL__)) else self.__get_unique_id()
    
    def ID(self):
        # I want ID to be read-only for a node
        return self.__ID()

@dataclass
class NodeLabelled(Node):
    label: str
    def __init__(self, id: int | None = None, label: str | None = None):
        super().__init__(id)
        self.label = label if label is not None else ""

@dataclass
class NodeVertex3(NodeLabelled):
    vertex: numpy.ndarray(shape=(3,1), dtype=numpy.float32)
    vertex_projective: numpy.ndarray(shape=(4,1), dtype=numpy.float32)
    def __init__(self, id: int | None = None,
                 label: str | None = None,
                 vertex: numpy.ndarray | list |  tuple | None = None ):
        super().__init__(id, label)
        if vertex is None:
            vertex = numpy.zeros(shape=(3,1), dtype=numpy.float32)
        if not isinstance(vertex, numpy.ndarray):
            if len(vertex) != 3:
                raise RuntimeError("vertex must be of length 3")
            vertex = numpy.array(vertex, dtype=numpy.float32).reshape((3,1))
        if vertex.shape != (3,1):
            vertex = vertex.reshape((3,1))
        self.vertex = vertex.copy()
        self.vertex_projective = numpy.append(self.vertex, 1).reshape((4,1))
    
    def __eq__(self, other):
        return (self.vertex == other.vertex).all()

@dataclass
class NodeTimedVertex3(NodeVertex3):
    t: float
    def __init__(self, id: int | None = None,
                 label: str | None = None,
                 vertex: numpy.ndarray | list |  tuple | None = None,
                 t: float | None = 0.0):
        super().__init__(id, label, vertex)
        self.t = t
    
    def __eq__(self, other):
        return super().__eq__(other) and self.t == other.t

@dataclass
class NodeVertex3Radius(NodeVertex3):
    radius: float
    def __init__(self, id: int | None = None,
                 label: str | None = None,
                 vertex: numpy.ndarray | list |  tuple | None = None,
                 radius: float | None = 0.0):
        super().__init__(id, label, vertex)
        self.radius = radius
    
    def __eq__(self, other):
        return super().__eq__(other) # two identical points with different radiuses make no sense in this case


# The following is the big fat class holding every imaginable information in itself
# Previous classes are small, standalone classes which have their use, but are 
# inappropriate to fully represent a point in an arterial tree

class CoronaryArteryNode(NodeVertex3):
    """Thorough explanation for DOCS
    """
    def __init__(self, id: int | None = None,
                 label: str | None = None,
                 vertex: numpy.ndarray | list |  tuple | None = None
                 ):
        super().__init__(id, label, vertex)



if __name__ == "__main__":
    print("Running 'HCATNetwork.node' module")
    print(sys.maxsize)
    n = Node()
    print(n)

    n2 = NodeLabelled(label="Matteo")
    print(n2)
    n2.label="Giovanni"
    print(n2)

    n3 = NodeVertex3(label="V3")
    print(n3)

    print(time.time_ns())
    print(time.time_ns())