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
from ..core.core import CoreDict

####################
# Common utilities
####################


##############
# Basic Edge
##############
"""
In this basic edge, the idea is to have positive distances for edges connecting
a proximal point on the centerline to a more distal point with respect to the 
coronary ostium, while having the same distance, but negative, in the connection
between a distal point to the proximal point.
Thus, it is possible to encounter negative distances.
"""
class BasicEdge(CoreDict):
    weight: float
    signed_distance: float

    def setPositiveSignedDistance(self) -> None:
        if self["weight"] is not None:
            self.__setitem__("signed_distance", abs(self.__getitem__("weight")) )
        else:
            raise ValueError("weight is currently None")
    
    def setNegativeSignedDistance(self) -> None:
        if self["weight"] is not None:
            self.__setitem__("signed_distance", -abs(self.__getitem__("weight")) )
        else:
            raise ValueError("weight is currently None")



if __name__ == "__main__":
    print("Running 'HCATNetwork.edge' module")

    d = BasicEdge_FeatureDict()
    d["weight"] = 0.9
    print(d.isValid())
    d.setNegativeSignedDistance()
    print(d.isValid())
    print(d, type(d))