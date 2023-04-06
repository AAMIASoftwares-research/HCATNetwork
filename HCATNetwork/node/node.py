"""
This file has the same aim of a header file in C/C++: to define standard data structures for node attributes.
The aim is to create a data-only struct, but in form of a dictionary.
Here we define many standard dictionaries used in the context of Heart Coronary Artery Tree mapping.
Depending on the kind of node, a node can have some features. Each feature is defined as a dictionary entry.
A node feature must be such that it can be then encoded, along all other features, in a feature matrix.

Following are defined, each with its use cases and explanation, some lists of keys which define
the dictionary entries.
To create a dictionary with the predefined kays and values, use:
    dict.fromkeys(some_KeysList)
        or
    node_dict = getDictFromKeyList(some_KeysList)
which will initialise each dictionary field to None. Of course, you should populate all of it, otherwise errors might occur.
For each node/dictionary type, a series of functions are predefined to be applicable to that specific kind of dict.

The steps then are:
    1. Create an uninitialised dict starting from one of the provided templates
    2. Populate ALL fields of the dict
    3. check for dictionary integrity with assertNodeValidity()

IMPORTANT:
Node ids MUST be strings. They can be floats or any hashable python object, but in reading/writing for
saving the file, all gets lost. To ensure continuity between a graph created on the fly and another one
opened from a file, use strings (even better if str(i), where i is an integer).
"""
import itertools
import numpy

####################
# Common utilities
####################
def assertNodeValidity(dictionary: dict):
    for v in dictionary.values():
        if v is None:
            return False
    return True

def getNodeDictFromKeyList(key_list: dict):
    """Just a wrapper function with a more memorable name"""
    return dict.fromkeys(key_list)



##############
# Vertex Node
##############
"""
A vertex node is a node defined by just its label and x, y, z positions
"""
VertexNode_KeysList = ["class", "x", "y", "z"]

def getListVertexFromVertexNode(d: dict):
    return [d["x"], d["y"], d["z"]]

def getNumpyVertexFromVertexNode(d: dict):
    return numpy.array(getListVertexFromVertexNode(d))

def setVertexNodeVertex(d: dict, v: list | numpy.ndarray):
    if len(v) == 0 or len(v) > 3:
        raise RuntimeError(f"Unsupported vertex length: {len(v)}")
    if isinstance(v, numpy.ndarray):
        v = v.flatten()
    for key, new_val in itertools.zip_longest(["x", "y", "z"], v, fillvalue=0.0):
        d[key] = float(new_val)
        


##########################################
# Other intermidiate nodes could be built
##########################################


#############################
# Simple HCA Centerline Node
#############################
"""
This node stores just the most basic information about the geometric centerline,
with no added complexity.
- "class": A literal from the list ["o", "s", "i", "e"]:
    o: coronary ostium/starting point of the left or right tree
    s: segment (a point with 2 connections)
    i: intersection (a point with more than two connections)
    e: endpoint
- x, y, z, t, r: The cartesian and temporal coordinates of the node, as well with
    r, which is the radius of the circle with area equivalent to the area of the coronary lumen at that point.
- "tree" must be one of the string literals defined in the following list: ["r", "l", "b"] or numeric [0, 1, 2]
    where "r" (0) stands for right, "l" (1) for left, "b" (2) for both
    (there are some heart structures in which the coronary arteries from left and right side branches merge together)
"""
SimpleCenterlineNode_KeysList        = ["class", "x", "y", "z", "t", "r", "tree"]
SimpleCenterlineNode_class_ValueList = ["o", "s", "i", "e"]
SimpleCenterlineNode_tree_ValueList  = ["r", "l", "b"]

def getListVertexFromSimpleCenterlineNode(d: dict):
    return [d["x"], d["y"], d["z"]]

def getNumpyVertexFromSimpleCenterlineNode(d: dict):
    return numpy.array(getListVertexFromSimpleCenterlineNode(d))

def setSimpleCenterlineNodeVertex(d: dict, v: list | numpy.ndarray):
    if len(v) == 0 or len(v) > 3:
        raise RuntimeError(f"Unsupported vertex length: {len(v)}")
    if isinstance(v, numpy.ndarray):
        v = v.flatten()
    for key, new_val in itertools.zip_longest(["x", "y", "z"], v, fillvalue=0.0):
        d[key] = float(new_val)

def getListVertexRadiusFromSimpleCenterlineNode(d: dict):
    return [d["x"], d["y"], d["z"], d["r"]]

def getNumpyVertexRadiusFromSimpleCenterlineNode(d: dict):
    return numpy.array(getListVertexRadiusFromSimpleCenterlineNode(d))

def setSimpleCenterlineNodeVertexRadius(d: dict, v: list | numpy.ndarray):
    if len(v) != 4:
        raise RuntimeError(f"Input object should be exactly of length 4, instead it is: {len(v)}")
    if isinstance(v, numpy.ndarray):
        v = v.flatten()
    for key, new_val in itertools.zip_longest(["x", "y", "z", "r"], v, fillvalue=0.0):
        d[key] = float(new_val)



############
# HCAT Node
############
"""
This is the "most complete" node, with everything that is needed and that might be needed in the future.
This is the only node actively maintained and that will be used in the future.
"""

HeartCoronaryArteryNode_KeysList = ["this is the most complete dict you can think of"]

if __name__ == "__main__":
    print("Running 'HCATNetwork.node' module")
    a = numpy.ones((1,))
    d = getNodeDictFromKeyList(VertexNode_KeysList)
    print(a, d)

    setVertexNodeVertex(d, a)
    print(d)