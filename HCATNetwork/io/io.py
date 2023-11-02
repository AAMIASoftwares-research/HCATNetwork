"""IO module for HCATNetwork.

Loading and Saving:
The GML file tipe is chosen to hold graph data, as it is widely supported and does not rely
on insecure python libraries as other do. NetworkX is used as a basic load/save interface.

To run as a module, activate the venv, go inside the HCATNetwork parent directory,
and use: python -m hcatnetwork.graph.graph
"""
import os, copy, json
from datetime import datetime, timezone
from enum import Enum

import numpy
import networkx

from ..node.node import SimpleCenterlineNodeAttributes, ArteryNodeTopology, ArteryNodeSide
from ..edge.edge import SimpleCenterlineEdgeAttributes
from ..graph.graph import SimpleCenterlineGraph

#########
# SAVING 
#########

def save_graph(
        graph: networkx.classes.graph.Graph|
               networkx.classes.digraph.DiGraph|
               networkx.classes.multigraph.MultiGraph|
               networkx.classes.multidigraph.MultiDiGraph|
               SimpleCenterlineGraph,
        file_path: str):
    """Saves the graph in GML format using the networkx interface.

    If some graph, node or edge features are:
        * numpy arrays
        * lists or multidimensional lists of basic python types

    Data are converted into json strings with json.dumps() before saving the graph in GML format.
    The file also contains everything needed to convert back the json strings into the original data types.
    
    Parameters
    ----------
    graph : networkx.classes.graph.Graph|
            networkx.classes.digraph.DiGraph|
            networkx.classes.multigraph.MultiGraph|
            networkx.classes.multidigraph.MultiDiGraph|
            SimpleCenterlineGraph
        The graph to be saved.
    file_path : str
        The path to the file where the graph will be saved as GML.

    Raises
    ------
    ValueError
        If any feature of the graph, node or edge is None.
    
    """
    # Make a deep copy of the graph, otherwise the graph will be modified even outside this function
    graph = copy.deepcopy(graph)
    # Date-time info to save inside file
    dt_dict = {"timezone": "UTC", "datetime": str(datetime.now(timezone.utc))}
    graph.graph["creation_datetime"] = dt_dict
    # Setup conversion directives lists
    node_features_conversion_k,  node_features_conversion_v   = [], []
    edge_features_conversion_k,  edge_features_conversion_v   = [], []
    graph_features_conversion_k, graph_features_conversion_v  = [], []
    # Convert any node data into a json string
    for n in graph.nodes.values():
        for k in n:
            if n[k] is None:
                # The field was not filled although being initialised (which means required)
                raise ValueError(f"Feature {k} in graph node {n} is None (was not filled). All features must be filled. Abort saving graph.")
            if isinstance(n[k], bool):
                if not k in node_features_conversion_k:
                    node_features_conversion_k.append(k)
                    node_features_conversion_v.append("bool")
            if isinstance(n[k], int):
                if not k in node_features_conversion_k:
                    node_features_conversion_k.append(k)
                    node_features_conversion_v.append("int")
            if isinstance(n[k], float):
                if not k in node_features_conversion_k:
                    node_features_conversion_k.append(k)
                    node_features_conversion_v.append("float")
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
            if isinstance(n[k], ArteryNodeTopology):
                if not k in node_features_conversion_k:
                    node_features_conversion_k.append(k)
                    node_features_conversion_v.append("hcatnetwork.node.ArteryNodeTopology")
                n[k] = str(n[k].name)
            if isinstance(n[k], ArteryNodeSide):
                if not k in node_features_conversion_k:
                    node_features_conversion_k.append(k)
                    node_features_conversion_v.append("hcatnetwork.node.ArteryNodeSide")
                n[k] = str(n[k].name)
    node_features_conversion_dict = {k: v for k, v in zip(node_features_conversion_k, node_features_conversion_v)}
    # Convert any edge data into a json string
    for e in graph.edges.values():
        for k in e:
            if e[k] is None:
                # The field was not filled although being initialised (which means required)
                raise ValueError(f"Feature {k} in graph edge {e} is None (was not filled). All features must be filled. Abort saving graph.")
            if isinstance(n[k], bool):
                if not k in edge_features_conversion_k:
                    edge_features_conversion_k.append(k)
                    edge_features_conversion_v.append("bool")
            if isinstance(n[k], int):
                if not k in edge_features_conversion_k:
                    edge_features_conversion_k.append(k)
                    edge_features_conversion_v.append("int")
            if isinstance(n[k], float):
                if not k in edge_features_conversion_k:
                    edge_features_conversion_k.append(k)
                    edge_features_conversion_v.append("float")
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
        if graph.graph[k] is None:
            # The field was not filled although being initialised (which means required)
            raise ValueError(f"Graph feature {k} is None (was not filled). All features must be filled. Abort saving graph.")
        if isinstance(n[k], bool):
            if not k in graph_features_conversion_k:
                graph_features_conversion_k.append(k)
                graph_features_conversion_v.append("bool")
        if isinstance(n[k], int):
            if not k in graph_features_conversion_k:
                graph_features_conversion_k.append(k)
                graph_features_conversion_v.append("int")
        if isinstance(n[k], float):
            if not k in graph_features_conversion_k:
                graph_features_conversion_k.append(k)
                graph_features_conversion_v.append("float")
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
        if isinstance(graph.graph[k], HeartDominance):
            if not k in node_features_conversion_k:
                node_features_conversion_k.append(k)
                node_features_conversion_v.append("hcatnetwork.node.HeartDominance")
            graph.graph[k] = str(n[k].name)
    graph_features_conversion_dict = {k: v for k, v in zip(graph_features_conversion_k, graph_features_conversion_v)}    
    # Save data conversion info
    graph.graph["node_features_conversion_dict"] = node_features_conversion_dict
    graph.graph["edge_features_conversion_dict"] = edge_features_conversion_dict
    graph.graph["graph_features_conversion_dict"] = graph_features_conversion_dict
    # safe to save
    networkx.write_gml(graph, file_path)
    # Cleanup deepcopied graph that was useful just for saving it
    del graph


#########
# LOADING 
#########

def load_enums(node, key, enum_class) -> Enum:
    out = None
    for d_ in enum_class:
        if d_.name == node[key]:
            out = d_
            break
    if out is None:
        raise ValueError(f"Error in loading graph nodes data: {type(enum_class)} does not have {node[key]} member.")
    return out

def load_graph_networkx_output(file_path: str) -> networkx.classes.graph.Graph:
    """Loads the graph from gml format using the networkx interface.
    
    The loading function will attempt to recover the original data types of the
    attributes of edges, nodes, and graph.
    No sign of the conversion is left behind on the loaded graph, as well as the conversion dictionaries.

    Parameters
    ----------
    file_path : str
        The path to the file where the graph is saved as gml.

    Returns
    -------
    networkx.classes.graph.Graph
        The loaded graph.
    
    Raises
    ------
    ValueError
        If any feature of the graph, node or edge is None.

    See Also
    --------
    networkx.read_gml()
    """
    graph = networkx.read_gml(file_path)
    del graph.graph["creation_datetime"]
    # Node data
    if "node_features_conversion_dict" in graph.graph:
        if graph.graph["node_features_conversion_dict"]:
            for n in graph.nodes:
                for k in graph.nodes[n]:
                    if k in graph.graph["node_features_conversion_dict"]:
                        if graph.graph["node_features_conversion_dict"][k] == "bool":
                            graph.nodes[n][k] = bool(int(n[k]))
                        elif graph.graph["node_features_conversion_dict"][k] == "int":
                            graph.nodes[n][k] = int(n[k])
                        elif graph.graph["node_features_conversion_dict"][k] == "float":
                            graph.nodes[n][k] = float(n[k])
                        elif graph.graph["node_features_conversion_dict"][k] == "numpy.ndarray":
                            graph.nodes[n][k] = numpy.array(json.loads(n[k]))
                        elif graph.graph["node_features_conversion_dict"][k] == "list":
                            graph.nodes[n][k] = json.loads(n[k])
                        elif graph.graph["node_features_conversion_dict"][k] == "hcatnetwork.node.ArteryNodeTopology":
                            graph.nodes[n][k] = load_enums(graph.nodes[n], k, ArteryNodeTopology)
                        elif graph.graph["node_features_conversion_dict"][k] == "hcatnetwork.node.ArteryNodeSide":
                            graph.nodes[n][k] = load_enums(graph.nodes[n], k, ArteryNodeSide)
        del graph.graph["node_features_conversion_dict"]
    # Edge data
    if "edge_features_conversion_dict" in graph.graph:
        if graph.graph["edge_features_conversion_dict"]:
            for es, et in graph.edges:
                for k in graph.edges[es, et]:
                    if k in graph.graph["edge_features_conversion_dict"]:
                        if graph.graph["edge_features_conversion_dict"][k] == "bool":
                            graph.edges[es, et][k] = bool(int(graph.edges[es, et][k]))
                        elif graph.graph["edge_features_conversion_dict"][k] == "int":
                            graph.edges[es, et][k] = int(graph.edges[es, et][k])
                        elif graph.graph["edge_features_conversion_dict"][k] == "float":
                            graph.edges[es, et][k] = float(graph.edges[es, et][k])
                        elif graph.graph["edge_features_conversion_dict"][k] == "numpy.ndarray":
                            graph.edges[es, et][k] = numpy.array(json.loads(graph.edges[es, et][k]))
                        elif graph.graph["edge_features_conversion_dict"][k] == "list":
                            graph.edges[es, et][k] = json.loads(graph.edges[es, et][k])
        del graph.graph["edge_features_conversion_dict"] 
    # Graph data
    if "graph_features_conversion_dict" in graph.graph:
        if graph.graph["graph_features_conversion_dict"]:
            for k in graph.graph:
                if k in graph.graph["graph_features_conversion_dict"]:
                    if graph.graph["graph_features_conversion_dict"][k] == "bool":
                        graph.graph[n][k] = bool(int(n[k]))
                    elif graph.graph["graph_features_conversion_dict"][k] == "int":
                        graph.graph[n][k] = int(n[k])
                    elif graph.graph["graph_features_conversion_dict"][k] == "float":
                        graph.graph[n][k] = float(n[k])
                    elif graph.graph["graph_features_conversion_dict"][k] == "numpy.ndarray":
                        graph.graph[k] = numpy.array(json.loads(graph.graph[k]))
                    elif graph.graph["graph_features_conversion_dict"][k] == "list":
                        graph.graph[k] = json.loads(graph.graph[k])
                    elif graph.graph["graph_features_conversion_dict"][k] == "hcatnetwork.node.HeartDominance":
                        graph.graph[k] = load_enums(graph.graph, k, HeartDominance)
        del graph.graph["graph_features_conversion_dict"]
    # Done
    return graph

def load_graph_simple_centerline_graph_output(file_path: str) -> SimpleCenterlineGraph:
    """Loads the graph from GML format using the networkx interface.
    
    The loading function will attempt to recover the original data types of the
    attributes of edges, nodes, and graph.
    No sign of the conversion is left behind on the loaded graph, as well as the conversion dictionaries.

    Parameters
    ----------
    file_path : str
        The path to the file where the graph is saved as GML.

    Returns
    -------
    SimpleCenterlineGraph
        The loaded graph.

    Raises
    ------
    ValueError
        If any feature of the graph, node or edge is None.

    See Also
    --------
    hcatnetwork.io.io.load_graph_networkx_output()
    """
    graph = load_graph_networkx_output(file_path)
    graph = SimpleCenterlineGraph.from_networkx_graph(graph)
    return graph

_load_graph_output_type_function_overload_map = {
    networkx.classes.graph.Graph: load_graph_networkx_output,
    SimpleCenterlineGraph: load_graph_simple_centerline_graph_output
}

def load_graph(file_path: str, output_type: type) -> (networkx.classes.graph.Graph| SimpleCenterlineGraph):
    """Loads the graph from GML format using the networkx interface.
    
    The loading function will attempt to recover the original data types of the
    attributes of edges, nodes, and graph.
    No sign of the conversion is left behind on the loaded graph, as well as the conversion dictionaries.

    Parameters
    ----------
    file_path : str
        The path to the file where the graph is saved as GML.
    output_type : type
        The type of graph to be returned.

    Returns
    -------
    networkx.classes.graph.Graph|
    networkx.classes.digraph.DiGraph|
    networkx.classes.multigraph.MultiGraph|
    networkx.classes.multidigraph.MultiDiGraph|
    SimpleCenterlineGraph
        The loaded graph.
    
    Raises
    ------
    ValueError
        If any feature of the graph, node or edge is None.

    See Also
    --------
    networkx.read_gml()
    """
    if not output_type in _load_graph_output_type_function_overload_map:
        raise TypeError(f"Error in loading graph: {output_type} is not a supported output type.")
    return _load_graph_output_type_function_overload_map[output_type](file_path)
    











if __name__ == "__main__":
    print("Running hcatnetwork.io.io")

    def mia(a: int):
        print(a, "is an integer")

    def mia(b: float):
        print(b, "is a float")

    def mia(c: str):
        print(c, "is a string")

    mia(1)
    mia(1.0)
    mia("1")    