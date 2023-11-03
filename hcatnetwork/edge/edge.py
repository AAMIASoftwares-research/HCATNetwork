"""Edge
This file defines standard data structures for edge attributes/features.
In this context, "edge attribute" is any object associated (contained) in a edge, while edge features implies
that each edge attribute is a single float, a sort of "unwinding" of edge attributes.
An edge feature must be such that it can be then encoded, along all other features, in a feature matrix.
Here we define many standard dictionaries used in the context of Heart Coronary Artery Tree mapping.
Depending on the kind of edge, an edge can have some attributes. Each attribute is defined as a dictionary entry.
The defined dictionaries are meant to be used together with NetworkX, which means that NetworkX must accept the
here-defined dictionaries as edge features.

An edge in NetworkX should have the "weight" parameter, which is used in many algorithms
as the "standard" parameter of the edge.
In NetworkX, the weight can assume any meaning. Whichever meaning you assign to it, there should also exist
a dictionary key for the specific meaning.
For example, if the weight is intended as a node-to-node distance in 3D space, then the dict should contain
both "weight" and "distance" or "distance_euclidean" or whatever.
"""
from ..core.core import CoreDict, TYPE_NAME_TO_TYPE_DICT

####################
# Common utilities
####################


##############
# Basic Edge
##############
"""
In this basic edge, only the euclidean distance between its nodes is conserved as a feature.
The weight standard parameter is set equal to the signed distance.

To run as a module, activate the venv, go inside the HCATNetwork parent directory,
and use: python -m hcatnetwork.edge.edge
"""

class SimpleCenterlineEdgeAttributes(CoreDict):
    weight: float
    euclidean_distance: float

    def update_weight_from_euclidean_distance(self) -> None:
        if self["euclidean_distance"] is not None:
            self.__setitem__("weight", abs(self.__getitem__("euclidean_distance")) )
        else:
            raise ValueError("euclidean_distance is currently None")
    
    def update_euclidean_distance_from_weight(self) -> None:
        if self["weight"] is not None:
            self.__setitem__("weight", abs(self.__getitem__("weight")) )
        else:
            raise ValueError("weight is currently None")
    

###########
# Add types
###########

TYPE_NAME_TO_TYPE_DICT["SimpleCenterlineEdgeAttributes"] = SimpleCenterlineEdgeAttributes


if __name__ == "__main__":
    print("Running 'hcatnetwork.edge' module")

    d = SimpleCenterlineEdgeAttributes()
    d["weight"] = 0.9
    print(d.is_full())
    print(d, type(d))