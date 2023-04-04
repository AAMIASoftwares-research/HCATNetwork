"""
This file has the same aim of a header file in C/C++: to define standard data structures for edge attributes.
The concept is the same as explained in the "node" file.

Following are defined, each with its use cases and explanation, some lists of keys which define
the dictionary entries.
To create a dictionary with the predefined keys and values, use:
    dict.fromkeys(some_KeysList)
        or
    node_dict = getDictFromKeyList(some_KeysList)
which will initialise each dictionary field to None. Of course, you should populate all of it, otherwise errors might occur.
For each edge/dictionary type, a series of functions are predefined to be applicable to that specific kind of dict.

The steps then are:
    1. Create an uninitialised dict starting from one of the provided templates.
    2. Populate ALL fields of the dict, by also using the utilities provided here.
    3. check for dictionary integrity with assertNodeValidity().
"""
####################
# Common utilities
####################
def assertEdgeValidity(dictionary: dict):
    for v in dictionary.values():
        if v is None:
            return False
    return True

def getEdgeDictFromKeyList(key_list: dict):
    """Just a wrapper function with a more memorable name"""
    return dict.fromkeys(key_list)



##############
# Basic Edge
##############
"""
An edge in NetworkX should have the "weight" parameter, which is used in many algorithms
as the standard parameter of the edge.
Here, the weight can assuma any meaning. Whichever meaning you assign to it, there should also exist
a dictionary key for the specific meaning.
    For example, if the weight is intended as a node distance in 3D space, then the dict should contain
    both "weight" and "distance" or "distance_euclidean" or wathever.

In this basic edge, the idea is to have positive distances for edges connecting a proximal point on the cnetelrine
to a more distal point with respect to the coronary ostium, while having the same distance, but negative, in the connection
between a distal point to the proximal point. Thus, it is possible to encounter negative distances.
"""
BasicEdge_KeysList = ["weight", "distance"]


if __name__ == "__main__":
    print("Running 'HCATNetwork.edge' module")