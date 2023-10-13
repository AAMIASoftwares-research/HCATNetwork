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

To run as a module, activate the venv, go inside the HCATNetwork parent directory,
and use: python -m HCATNetwork.graph.graph
"""
import os, sys, copy, json
from datetime import datetime, timezone
from enum import Enum, auto
import numpy
import networkx

from ..core.core import CoreDict
from ..node.node import ArteryPointTopologyClass, ArteryPointTree

###################################
# LOADING and SAVING to text files
###################################

def saveGraph(
        graph: networkx.classes.graph.Graph|
               networkx.classes.digraph.DiGraph|
               networkx.classes.multigraph.MultiGraph|
               networkx.classes.multidigraph.MultiDiGraph,
        file_path: str):
    """saveGraph in GML file format
    Saves the graph in GML format using the networkx interface.

    If some graph, node or edge features are:
        * numpy arrays
        * lists or multidimensional lists of basic python types

    data are converted into json strings with json.dumps() before 
    saving the graph in GML format.
    The file also contains everything needed to convert back the json strings into the original data types.
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
            if isinstance(n[k], ArteryPointTopologyClass):
                if not k in node_features_conversion_k:
                    node_features_conversion_k.append(k)
                    node_features_conversion_v.append("HCATNetwork.node.ArteryPointTopologyClass")
                n[k] = str(n[k].name)
            if isinstance(n[k], ArteryPointTree):
                if not k in node_features_conversion_k:
                    node_features_conversion_k.append(k)
                    node_features_conversion_v.append("HCATNetwork.node.ArteryPointTree")
                n[k] = str(n[k].name)
    node_features_conversion_dict = {k: v for k, v in zip(node_features_conversion_k, node_features_conversion_v)}
    # Convert any edge data into a json string
    for e in graph.edges.values():
        for k in e:
            if e[k] is None:
                # The field was not filled although being initialised (which means required)
                raise ValueError(f"Feature {k} in graph edge {e} is None (was not filled). All features must be filled. Abort saving graph.")
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
                node_features_conversion_v.append("HCATNetwork.node.HeartDominance")
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

def loadEnums(node, key, clss) -> Enum:
    out = None
    for d_ in clss:
        if d_.name == node[key]:
            out = d_
            break
    if out is None:
        raise ValueError(f"Error in loading graph nodes data: {type(clss)} does not have {node[key]} member.")
    return out

def loadGraph(file_path: str) -> networkx.classes.graph.Graph|\
                                    networkx.classes.digraph.DiGraph|\
                                    networkx.classes.multigraph.MultiGraph|\
                                    networkx.classes.multidigraph.MultiDiGraph:
    """Loads the graph from gml format using the networkx interface.
    The loading function will attempt to recover the original data types of the
    attributes of edges, nodes, and graph.
    No sign of the conversion is left behind on the loaded graph, as well as the conversion dictionaries.
    """
    graph = networkx.read_gml(file_path)
    del graph.graph["creation_datetime"]
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
                        elif graph.graph["node_features_conversion_dict"][k] == "HCATNetwork.node.ArteryPointTopologyClass":
                            n[k] = loadEnums(n, k, ArteryPointTopologyClass)
                        elif graph.graph["node_features_conversion_dict"][k] == "HCATNetwork.node.ArteryPointTree":
                            n[k] = loadEnums(n, k, ArteryPointTree)
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
                    elif graph.graph["node_features_conversion_dict"][k] == "HCATNetwork.node.HeartDominance":
                        graph.graph[k] = loadEnums(n, k, HeartDominance)
        del graph.graph["graph_features_conversion_dict"]
    # Done
    return graph


########################
# Basic Centerline Graph
########################

class BasicCenterlineGraph(CoreDict):
    """The basic centerline graph static dictionary.

    There is no way of hard-imposing it in python, but the graph should contain:
        * nodes of type HCATNetwork.node.SimpleCenterlineNode,
        * edges of type HCATNetwork.edge.BasicEdge.

    Both trees (left and right) are stored in the same NetworkX.Graph structure.
    In some patients, it could happen that the left and right subgraphs are not disjointed, hence the need to have just one graph.
    BasicCenterlineGraph, and all its connected algorithm, assumes that the coronary ostia are disjointed, meaning that the 
    coronary ostia should not be overlapping or too near to each other. No control is actively performed to check this condition.
    
    Keys
    ----
    image_id : str
        The image id of the image from which the graph was extracted.
        This string has no fixed format, and can be anything, but it should be clear enough
        to identify the image, and possibly be consistent for images coming from the same source/dataset.
    
    are_left_right_disjointed : bool
        ``True`` if the left and right coronary trees are disjointed, ``False`` otherwise.

    See Also
    --------
    HCATNetwork.core.CoreDict
    """
    image_id : str
    are_left_right_disjointed : bool
    
    @staticmethod
    def getCoronaryOstiumNodeIdRelativeToNode(graph: networkx.classes.graph.Graph, node_id: str) -> tuple[str]:
        """Get the coronary ostium node id relative to the node with id node_id
        Output: a 1-tuple or 2-tuple of strings, depending on whether the node is associated with one or both arterial trees.
                The 2-tuple always contains the left ostium node id as the first element, and the right ostium node id as the second element.
        """
        if not node_id in [id for id in graph.nodes]:
            raise ValueError(f"Node with id \"{node_id}\" is not in graph.")
        node = graph.nodes[node_id]
        # Node is a coronary ostium
        if node['topology_class'].value == ArteryPointTopologyClass.OSTIUM.value:
            return tuple([node_id])
        # Node is not a coronary ostium
        # The node could be associated with either one or both arterial trees.
        # There should be no nodes asssociated with no artrial trees.
        if node['arterial_tree'].value != ArteryPointTree.RL.value:
            # The node is associated with a single tree
            for n in graph.nodes:
                if graph.nodes[n]['arterial_tree'].value == node['arterial_tree'].value and graph.nodes[n]['topology_class'].value == ArteryPointTopologyClass.OSTIUM.value:
                    return tuple([n])
        else:
            # The node is associated with both arterial trees
            count_hits_ = 0
            left_ostium_n, right_ostium_n = None, None
            for n in graph.nodes:
                if graph.nodes[n]['topology_class'].value == ArteryPointTopologyClass.OSTIUM.value:
                    if graph.nodes[n]['arterial_tree'].value == ArteryPointTree.LEFT.value:
                        left_ostium_n = n
                    elif graph.nodes[n]['arterial_tree'].value == ArteryPointTree.RIGHT.value:
                        right_ostium_n = n
                    else:
                        raise RuntimeError(f"Node {n} is a coronary ostium associated with no arterial tree (nor left, nor right).")
                    count_hits_ += 1
                    if count_hits_ == 2:
                        return tuple([left_ostium_n, right_ostium_n])
        # If the code reaches this point, it means that the node is not associated with any arterial tree
        raise RuntimeError(f"Node {n} is a coronary ostium associated with no arterial tree (nor left, nor right).")

    @staticmethod
    def get_coronary_ostia_node_id(graph: networkx.classes.graph.Graph) -> tuple():
        """Gets the left and right coronary ostia node ids.
        Returns a  2-tuple of strings, where the first element is the left ostium node id, and the second element is the right ostium node id.
        If an ostium cannot be found, the element will be set to None.
        """
        count_hits_ = 0
        left_ostium_n, right_ostium_n = None, None
        for n in graph.nodes:
            if graph.nodes[n]['topology_class'].value == ArteryPointTopologyClass.OSTIUM.value:
                if graph.nodes[n]['arterial_tree'].value == ArteryPointTree.LEFT.value:
                    left_ostium_n = n
                elif graph.nodes[n]['arterial_tree'].value == ArteryPointTree.RIGHT.value:
                    right_ostium_n = n
                else:
                    raise RuntimeError(f"Node {n} is a coronary ostium associated with no arterial tree (nor left, nor right).")
                count_hits_ += 1
                if count_hits_ == 2:
                    return tuple([left_ostium_n, right_ostium_n])
        # If the code reaches this point, it means that the graph does not have two ostia, so return the tuple with a None element
        return tuple([left_ostium_n, right_ostium_n])
    
    @staticmethod
    def get_segments(graph: networkx.classes.graph.Graph) -> list[tuple[str]]:
        """Gets the segments of the graph, starting from the coronary ostia.

        The segments are returned as a list of tuples, each containing the start and end node id of a segment delimited by either an ostium, intersection, or endpoint.
        
        Parameters
        ----------
        graph : networkx.classes.graph.Graph
            The graph to be walked.

        Returns
        -------
        list[tuple[str]]
            A list of tuples, each containing the start and end node id of a segment delimited by either an ostium, intersection, or endpoint.
        
        """
        segments = []
        for start_node_id in BasicCenterlineGraph.get_coronary_ostia_node_id(graph):
            nodes_distances_from_start_node = networkx.single_source_dijkstra_path_length(graph, start_node_id)
            breadth_first_search_successors_from_ostium = networkx.bfs_successors(graph, start_node_id)
            # Walk the graph from the ostium to the next landmark (intersection or endpoint)
            current_node_id = start_node_id
            def recursive_walk(segment_start_node_id_next, segments):
                ###  TO DOOOOOO
                # SOMEHOW YOU CAN DO IT WITH RECURSION
                # JUST GET TO THE NEXT LANDMARK AND THEN RECURSIVELY CALL THE FUNCTION
                # BEFORE RECURSIVELY CALLING THE FUNCTION, ADD THE SEGMENT TO THE LIST
                next_nodes = breadth_first_search_successors_from_ostium[segment_start_node_id_next]
                # do not know if following code works, copilot did it but I do not understand it
                if len(next_nodes) > 1:
                    segment_end_node_id = next_nodes[0]
                    segments.append((segment_start_node_id_next, segment_end_node_id))
                    recursive_walk(current_node_id, segment_start_node_id_next, segment_end_node_id)
                elif len(next_nodes) == 1:
                    segment_end_node_id = next_nodes[0]
                    segments.append((segment_start_node_id_next, segment_end_node_id))
                    recursive_walk(current_node_id, segment_start_node_id_next, segment_end_node_id)
                else:
                    pass

            

            

            '''
            while True:
                # Get the neighbors of the current node
                neighbors = list(graph.neighbors(current_node_id))
                # If the current node is an endpoint or an intersection, add the segment and move to the next landmark
                if len(neighbors) > 2 or len(neighbors) == 1:
                    segments.append((current_node_id, neighbors[0]))
                    if len(neighbors) > 1:
                        current_node_id = neighbors[1]
                    else:
                        break
                # If the current node is an ostium, add the segment and move to the next ostium
                elif graph.nodes[current_node_id]['topology_class'].value == ArteryPointTopologyClass.OSTIUM.value:
                    segments.append((current_node_id, neighbors[0]))
                    break
                # If the current node is a regular node, add the segment and move to the next node
                else:
                    segments.append((current_node_id, neighbors[0]))
                    current_node_id = neighbors[0]
            '''
        return segments

    @staticmethod
    def get_simplified_landmarks_graph(graph):
        pass
        
    @staticmethod
    def resample_coronary_artery_tree_disjointed(graph: networkx.classes.graph.Graph, mm_between_nodes: float = 0.5):
        """Resamples the coronary artery tree so that two connected points are on average mm_between_nodes millimeters apart.

        NOTE: This function only works correctly with coronary artery trees which are disjointed, meaning that the left and right trees are not connected. 

        The tree is resampled so that the absolute position of coronary ostia, intersections and endpoints is preserved.
        The position of the nodes between these landmarks can vary, and so can radius data, which is interpolated (linear).
        
        Parameters
        ----------
        graph : networkx.classes.graph.Graph
            The graph to be resampled.
        mm_between_nodes : float, optional
            The average distance between two connected points, in millimeters, by default 0.5.
        
        """
        pass
        # Get the two coronary ostia node ids
        left_ostium_n, right_ostium_n = BasicCenterlineGraph.get_coronary_ostia_node_id(graph)
        # Get all nodes distances from the left ostium
        distances_from_left_ostium = networkx.single_source_dijkstra_path_length(graph, left_ostium_n)
        # Get all nodes distances from the right ostium
        distances_from_right_ostium = networkx.single_source_dijkstra_path_length(graph, right_ostium_n)
        #########################
        #########################
        #########################
        #########################
        #########################
        
        # Walk recursively to the next landmark (intersection or endpoint) and resample the segment in between

            
        


############################
# Coronary Artery Tree Graph
############################
"""Coronary Artery Tree Graph
This is the most complete graph, holding everything needed for representing a coronary artery tree.
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
    
    # Load a coronary artery tree graph
    f_prova = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\CAT08\\CenterlineGraphs_FromReference\\dataset00.GML"
    g_ = loadGraph(f_prova)
    segments = BasicCenterlineGraph.get_segments(g_)
    print(segments)



    