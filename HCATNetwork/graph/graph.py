"""Graph
This file defines standard data structures for graph attributes/features.
In this context, "graph attribute" is any object associated (contained) in a graph, while graph features implies
that each graph attribute is a single float, a sort of "unwinding" of node attributes.
A graph feature must be such that it can be then encoded, along all other features, in a feature matrix.
Here we define many standard dictionaries used in the context of Heart Coronary Artery Tree mapping.
Depending on the kind of graph, a graph can have some attributes. Each attribute is defined as a dictionary entry.
The defined dictionaries are meant to be used together with NetworkX, which means that NetworkX must accept the
here-defined dictionaries as node features.

A graph can hold, other than nodes and edges, also all sorts of information, which are stored in graph attributes.
Here, the information is encoded in standard containers, depending on the type of graph.

Loading and Saving:
The GML file tipe is chosen to hold graph data, as it is widely supported and does not rely
on insecure python libraries as other do. NetworkX is used as a load/save interface.
"""
import os, sys, copy, json
from enum import Enum, auto
import numpy
import networkx

from ..core.core import CoreDict

###################################
# LOADING and SAVING to text files
##################################

def saveGraph(
        graph: networkx.classes.graph.Graph|
               networkx.classes.digraph.DiGraph|
               networkx.classes.multigraph.MultiGraph|
               networkx.classes.multidigraph.MultiDiGraph,
        file_path: str):
    """saveGraph in GML file format
    Saves the graph in GML format using the networkx interface.
    If some data are:
    - numpy arrays
    - lists or multidimensional lists
    data are converted into json strings with json.dumps() before 
    saving the graph in GML format.
    """
    # Make a deep copy of the graph, otherwise the graph will be modified even outside this function
    graph = copy.deepcopy(graph)
    # Setup conversion directives lists
    node_features_conversion_k,  node_features_conversion_v   = [], []
    edge_features_conversion_k,  edge_features_conversion_v   = [], []
    graph_features_conversion_k, graph_features_conversion_v  = [], []
    # Convert any node data into a json string
    for n in graph.nodes.values():
        for k in n:
            if isinstance(n[k], numpy.ndarray):
                if not k in node_features_conversion_k:
                    node_features_conversion_k.append(k)
                    node_features_conversion_v.append("numpy.ndarray")
                n[k] = n[k].tolist()
            if isinstance(n[k], list):
                if not k in node_features_conversion_k:
                    node_features_conversion_k.append(k)
                    node_features_conversion_v.append("list")
                n[k] = json.dumps(n[k])
    node_features_conversion_dict = {k: v for k, v in zip(node_features_conversion_k, node_features_conversion_v)}
    # Convert any edge data into a json string
    for e in graph.edges.values():
        for k in e:
            if isinstance(e[k], numpy.ndarray):
                if not k in edge_features_conversion_k:
                    edge_features_conversion_k.append(k)
                    edge_features_conversion_v.append("numpy.ndarray")
                e[k] = e[k].tolist()
            if isinstance(e[k], list):
                if not k in edge_features_conversion_k:
                    edge_features_conversion_k.append(k)
                    edge_features_conversion_v.append("list")
                e[k] = json.dumps(e[k])
    edge_features_conversion_dict = {k: v for k, v in zip(edge_features_conversion_k, edge_features_conversion_v)}
    # Convert any graph data into a json string
    for k in graph.graph:
        if isinstance(graph.graph[k], numpy.ndarray):
            if not k in graph_features_conversion_k:
                graph_features_conversion_k.append(k)
                graph_features_conversion_v.append("numpy.ndarray")
            graph.graph[k] = graph.graph[k].tolist()
        if isinstance(graph.graph[k], list):
            if not k in graph_features_conversion_k:
                graph_features_conversion_k.append(k)
                graph_features_conversion_v.append("list")
            graph.graph[k] = json.dumps(graph.graph[k])
    graph_features_conversion_dict = {k: v for k, v in zip(graph_features_conversion_k, graph_features_conversion_v)}    
    # Save data conversion info
    graph.graph["node_features_conversion_dict"] = node_features_conversion_dict
    graph.graph["edge_features_conversion_dict"] = edge_features_conversion_dict
    graph.graph["graph_features_conversion_dict"] = graph_features_conversion_dict
    # safe to save
    networkx.write_gml(graph, file_path)
    # Cleanup deepcopied graph that was useful just for saving it
    del graph

def loadGraph(file_path: str) ->    networkx.classes.graph.Graph|\
                                    networkx.classes.digraph.DiGraph|\
                                    networkx.classes.multigraph.MultiGraph|\
                                    networkx.classes.multidigraph.MultiDiGraph:
    """Loads the graph from gml format using the networkx interface.
    The loading function will attempt to recover the original data type of the
    attributes of edges, nodes, and graph.
    No sign of the conversion is left behind on the loaded graph.
    """
    graph = networkx.read_gml(file_path)
    # Node data
    if "node_features_conversion_dict" in graph.graph:
        if graph.graph["node_features_conversion_dict"]:
            for n in graph.nodes.values():
                for k in n:
                    if k in graph.graph["node_features_conversion_dict"]:
                        if graph.graph["node_features_conversion_dict"][k] == "numpy.ndarray":
                            n[k] = numpy.array(json.loads(n[k]))
                        elif graph.graph["node_features_conversion_dict"][k] == "list":
                            n[k] = json.loads(n[k])
        del graph.graph["node_features_conversion_dict"] 
    # Edge data
    if "edge_features_conversion_dict" in graph.graph:
        if graph.graph["edge_features_conversion_dict"]:
            for e in graph.nodes.values():
                for k in e:
                    if k in graph.graph["edge_features_conversion_dict"]:
                        if graph.graph["edge_features_conversion_dict"][k] == "numpy.ndarray":
                            e[k] = numpy.array(json.loads(e[k]))
                        elif graph.graph["edge_features_conversion_dict"][k] == "list":
                            e[k] = json.loads(e[k])
        del graph.graph["edge_features_conversion_dict"] 
    # Graph data
    if "graph_features_conversion_dict" in graph.graph:
        if graph.graph["graph_features_conversion_dict"]:
            for k in graph.graph:
                if k in graph.graph["graph_features_conversion_dict"]:
                    if graph.graph["graph_features_conversion_dict"][k] == "numpy.ndarray":
                        graph.graph[k] = numpy.array(json.loads(graph.graph[k]))
                    elif graph.graph["graph_features_conversion_dict"][k] == "list":
                        graph.graph[k] = json.loads(graph.graph[k])    
        del graph.graph["graph_features_conversion_dict"]
    # Done
    return graph


########################
# Basic Centerline Graph
########################
"""The basic centerline graph
"""

class BasicCenterlineGraph(CoreDict):
     image_id: str
     are_left_right_disjointed: bool


############################
# Coronary Artery Tree Graph
############################
"""Coronary Artery Tree Graph
This is the most complete graph, holding everything needed for coronary artery tree.
Both trees (l and r) are stored in the same graph.
In some patients, it could happen that the left and right subgraphs are not disjointed, hence the need to have just one graph.
For the future.
"""

# Heart dominance is described by which coronary artery branch gives off the posterior descending artery and supplies the inferior wall, and is characterized as left, right, or codominant
class HeartDominance(Enum):
    LEFT = auto()
    RIGHT= auto()
    CODOMINANT = auto()

if __name__ == "__main__":
    print("Running 'HCATNetwork.graph' module")

    # example: save a graph with nodes holding a random ndarray, and see what happens
    attr_dict = BasicCenterlineGraph
    attr_dict["image_id"] = "nessuna immagine"
    attr_dict["are_left_right_disjointed"] = 1
    g = networkx.Graph(**attr_dict)
    for i in range(5):
        ndarray = numpy.random.randn(i+1, i+2, i+4)
        g.add_node(str(i), numpy_array=ndarray) #json.dumps(ndarray.tolist()))
    path = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\HCATNetwork\\HCATNetwork\\test\\prova_grafo_numpy.GML"
    saveGraph(g, path)
    g2 = loadGraph(path)
    saveGraph(g2, path[:-5]+"2.GML")
    g3 = loadGraph(path[:-5]+"2.GML")

    print(g.nodes["0"], g2.nodes["0"], g3.nodes["0"], "\n\n")
    quit()
    for n in g2.nodes.values():
        n["numpy_array"] = json.loads(n["numpy_array"])
    print([(type(n[1]["numpy_array"]), n[1]["numpy_array"]) for n in g2.nodes.items()][0])
    print([(type(n[1]["numpy_array"][0][0][0]),n[1]["numpy_array"][0][0][0]) for n in g2.nodes.items()][0])

    g.node


    