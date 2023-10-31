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
from ..node.node import SimpleCenterlineNode, ArteryPointTopologyClass, ArteryPointTree
from ..edge.edge import BasicEdge
from ..utils.slicer import numpy_array_to_open_curve_json, numpy_array_to_fiducials_json

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

    Data are converted into json strings with json.dumps() before saving the graph in GML format.
    The file also contains everything needed to convert back the json strings into the original data types.
    
    Parameters
    ----------
    graph : networkx.classes.graph.Graph|
            networkx.classes.digraph.DiGraph|
            networkx.classes.multigraph.MultiGraph|
            networkx.classes.multidigraph.MultiDiGraph
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
    def get_anatomic_segments_ids(graph: networkx.classes.graph.Graph) -> list[tuple[str]]:
        """Gets the segments of the graph, starting from the coronary ostia.

        The segments are returned as a list of tuples, each containing the start and end node id of
        a segment delimited by either an ostium, intersection, or endpoint.
        A segment is a piece of graph connecting an ostium or intersection to an intersection or an endpoint,
        without any other landmark (ostium, intersection, endpoint) in between.
        
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
            # Fore each node, get the successor of the node starting from the ostium
            breadth_first_search_successors_from_ostium = {key: val for (key, val) in networkx.bfs_successors(graph, start_node_id)}
            next_buffer_list: list = breadth_first_search_successors_from_ostium[start_node_id]
            start_buffer_list: list = [start_node_id]
            while len(next_buffer_list) != 0:
                # Get the next node
                current_node_id = next_buffer_list[0]
                # Since you considered this, remove it from the list
                next_buffer_list.pop(0)
                # Walk up to the next landmark (intersection, or endpoint)
                stay_in_loop_ = current_node_id in breadth_first_search_successors_from_ostium # endpoint check
                if stay_in_loop_:
                    stay_in_loop_ = stay_in_loop_ and len(breadth_first_search_successors_from_ostium[current_node_id]) == 1 # intersection check
                while stay_in_loop_:
                    # Jump to the next node
                    current_node_id = breadth_first_search_successors_from_ostium[current_node_id][0]
                    # Check if the node is a landmark (intersection, or endpoint)
                    stay_in_loop_ = current_node_id in breadth_first_search_successors_from_ostium # endpoint check
                    if stay_in_loop_:
                        stay_in_loop_ = stay_in_loop_ and len(breadth_first_search_successors_from_ostium[current_node_id]) == 1 # intersection check
                # This node should be a landmark (intersection, or endpoint)
                # - add the segment
                segments.append((start_buffer_list[0], current_node_id))
                # - the start node was considered, so remove it from the start list
                start_buffer_list.pop(0)
                # - set up data to walk to next landmark(s)
                if current_node_id in breadth_first_search_successors_from_ostium:
                    if len(breadth_first_search_successors_from_ostium[current_node_id]) != 0:
                        for new_next_node in breadth_first_search_successors_from_ostium[current_node_id]:
                            start_buffer_list.append(current_node_id)
                            next_buffer_list.append(new_next_node)
        return segments

    @staticmethod
    def get_anatomic_subgraph(graph: networkx.classes.graph.Graph) -> networkx.classes.graph.Graph:
        """Gets the anatomic subgraph of the graph, meaning the graph of just coronary ostia, intersections and endpoints.

        In the returned graph, the distance between nodes is not the euclidean distance between the nodes,
        but the euclidean length of the segment.

        Parameters
        ----------
        graph : networkx.classes.graph.Graph
            The graph to be simplified.

        Returns
        -------
        networkx.classes.graph.Graph
            The simplified graph.
        
        """
        # Create the subgraph, copying the info from the original graph
        subgraph = networkx.Graph(**graph.graph)
        # Get the segments
        segments = BasicCenterlineGraph.get_anatomic_segments_ids(graph)
        # Add the nodes
        for segment in segments:
            if not segment[0] in subgraph.nodes:
                subgraph.add_node(segment[0], **graph.nodes[segment[0]])
            if not segment[1] in subgraph.nodes:
                subgraph.add_node(segment[1], **graph.nodes[segment[1]])
        # Add the edges
        for segment in segments:
            edge_features = BasicEdge()
            edge_features["euclidean_distance"] = networkx.algorithms.shortest_path_length(graph, segment[0], segment[1], weight="euclidean_distance")
            edge_features.updateWeightFromEuclideanDistance()
            subgraph.add_edge(segment[0], segment[1], **edge_features)
        # Done
        return subgraph
        
    @staticmethod
    def resample_coronary_artery_tree(graph: networkx.classes.graph.Graph, mm_between_nodes: float = 0.5) -> networkx.classes.graph.Graph:
        """Resamples the coronary artery tree so that two connected points are on average mm_between_nodes millimeters apart.

        The tree is resampled so that the absolute position of coronary ostia, intersections and endpoints is preserved.
        The position of the nodes between these landmarks can vary, and so can radius data, which is interpolated (linear).
        
        Parameters
        ----------
        graph : networkx.classes.graph.Graph
            The graph to be resampled.
        mm_between_nodes : float, optional
            The average distance between two connected points, in millimeters, by default 0.5.

        Returns
        -------
        networkx.classes.graph.Graph
            The resampled graph.
        
        """
        # Create the new graph, copying the info from the original graph
        graph_new = networkx.Graph(**graph.graph)
        # Get the anatomic segments of the original graph
        segments = BasicCenterlineGraph.get_anatomic_segments_ids(graph)
        # - consider each segment only once, needed for patients with non-disjointed left and right trees
        # - we do not want to resample the same segment twice
        segments = list(set(segments)) 
        untouchable_node_ids = [a for (a,b) in segments] + [b for (a,b) in segments]
        # Resample each segment
        node_id_counter = 0
        for n0, n1 in segments:
            # Get the number of nodes to put into this segment (counting also n0 and n1)
            # Therefore, the number of nodes is always at least 2.
            length = networkx.algorithms.shortest_path_length(graph, n0, n1, weight="euclidean_distance")
            n_nodes = max(
                [2, int(length / mm_between_nodes)]
            )
            # Resample the segment
            if n_nodes == 2:
                # Just add the two nodes
                if not n0 in graph_new.nodes:
                    graph_new.add_node(n0, **graph.nodes[n0])
                if not n1 in graph_new.nodes:
                    graph_new.add_node(n1, **graph.nodes[n1])
                # Add the edge
                # Here, the edge's property "euclidean_distance" is the actual distance between the nodes.
                if not graph_new.has_edge(n0, n1):
                    edge_features = BasicEdge()
                    n0_p = numpy.array([graph.nodes[n0]["x"], graph.nodes[n0]["y"], graph.nodes[n0]["z"]])
                    n1_p = numpy.array([graph.nodes[n1]["x"], graph.nodes[n1]["y"], graph.nodes[n1]["z"]])
                    edge_features["euclidean_distance"] = numpy.linalg.norm(n0_p - n1_p)
                    edge_features.updateWeightFromEuclideanDistance()
                    graph_new.add_edge(n0, n1, **edge_features)
            else:
                distances_to_sample = numpy.linspace(0, length, n_nodes)
                nodes_ids_to_connect_in_sequence_list = []
                # First and last node will be n0 and n1, respectively
                # Add the first node
                if not n0 in graph_new.nodes:
                    graph_new.add_node(n0, **graph.nodes[n0])
                nodes_ids_to_connect_in_sequence_list.append(n0)
                # Add the middle nodes
                # - get all nodes in the segment
                nodes_in_segment = networkx.algorithms.shortest_path(graph, n0, n1)
                nodes_in_segment_distances_from_n0 = {n__: networkx.algorithms.shortest_path_length(graph, n0, n__, weight="euclidean_distance") for n__ in nodes_in_segment}
                for d in distances_to_sample[1:-1]:
                    # Find node before and after the distance d
                    node_before_, node_ = None, None
                    for n in nodes_in_segment:
                        if nodes_in_segment_distances_from_n0[n] <= d:
                            node_before_ = n
                        elif nodes_in_segment_distances_from_n0[n] > d:
                            node_ = n
                            break
                    # Interpolate the position and radius of the node
                    p_n_ = numpy.array([graph.nodes[node_]["x"], graph.nodes[node_]["y"], graph.nodes[node_]["z"]])
                    p_n_b_ = numpy.array([graph.nodes[node_before_]["x"], graph.nodes[node_before_]["y"], graph.nodes[node_before_]["z"]])
                    proportion_ = (d - nodes_in_segment_distances_from_n0[node_before_]) / (nodes_in_segment_distances_from_n0[node_] - nodes_in_segment_distances_from_n0[node_before_])
                    position_new_ = p_n_b_ + (p_n_ - p_n_b_) * proportion_
                    radius_new_ = graph.nodes[node_before_]["r"] + (graph.nodes[node_]["r"] - graph.nodes[node_before_]["r"]) * proportion_
                    # Add the node to the graph and to the list to then connect
                    while (str(node_id_counter) in graph_new.nodes) or (str(node_id_counter) in untouchable_node_ids):
                        # make sure no new nodes have the same id
                        node_id_counter += 1
                    node_features = SimpleCenterlineNode()
                    node_features.setVertex(position_new_)
                    node_features["r"] = radius_new_
                    node_features["t"] = 0.0
                    node_features["topology_class"] = ArteryPointTopologyClass.SEGMENT
                    node_features["arterial_tree"] = graph.nodes[node_before_]["arterial_tree"]
                    graph_new.add_node(str(node_id_counter), **node_features)
                    nodes_ids_to_connect_in_sequence_list.append(str(node_id_counter))
                # Add the last node
                if not n1 in graph_new.nodes:
                    graph_new.add_node(n1, **graph.nodes[n1])
                nodes_ids_to_connect_in_sequence_list.append(n1)
                # Connect the nodes
                for i in range(len(nodes_ids_to_connect_in_sequence_list) - 1):
                    n0 = nodes_ids_to_connect_in_sequence_list[i]
                    n1 = nodes_ids_to_connect_in_sequence_list[i + 1]
                    if not graph_new.has_edge(n0, n1):
                        edge_features = BasicEdge()
                        n0_p = numpy.array([graph_new.nodes[n0]["x"], graph_new.nodes[n0]["y"], graph_new.nodes[n0]["z"]])
                        n1_p = numpy.array([graph_new.nodes[n1]["x"], graph_new.nodes[n1]["y"], graph_new.nodes[n1]["z"]])
                        edge_features["euclidean_distance"] = numpy.linalg.norm(n0_p - n1_p)
                        edge_features.updateWeightFromEuclideanDistance()
                        graph_new.add_edge(n0, n1, **edge_features)
        return graph_new
    
    @staticmethod
    def convert_to_3dslicer_opencurve(graph: networkx.classes.graph.Graph, save_directory: str, affine_transformation_matrix: numpy.ndarray | None = None) -> None:
        """This function converts each segment, from ostium to endpoint, into an open curve
        that can be loaded directly in 3D Slicer.
        
        The 3D Slicer curve control points coordinate system is the RAS (Right, Anterior, Superior) coordinate system.
        
        Parameters
        ----------
        graph : networkx.classes.graph.Graph
            The graph to be converted.
        save_directory : str
            The directory where the curves will be saved.
            If the directory is just a name, it will be created inside the current working directory.
        affine_transformation_matrix : numpy.ndarray, optional
            The affine transformation matrix to apply to the points, by default None.
            If None, the identity transformation is applied.
        
        Raises
        ------
        FileNotFoundError
            If the save_directory does not exist or cannot be created.
        """
        # Directory handling
        if not os.path.exists(save_directory):
            try:
                os.mkdir(save_directory)
            except:
                raise FileNotFoundError(f"Directory {save_directory} does not exist and cannot be created.")
        # Affine transformation matrix handling
        if affine_transformation_matrix is None:
            affine_transformation_matrix = numpy.identity(4)
        else:
            if affine_transformation_matrix.shape != (4, 4):
                raise ValueError(f"Affine transformation matrix must be a 4x4 matrix, not {affine_transformation_matrix.shape}.")
        # Cycle through all endpoints
        for n in graph.nodes:
            if graph.nodes[n]['topology_class'] == ArteryPointTopologyClass.ENDPOINT:
                endpoint_node_id = n
                # get coronary ostium node id that is connected to this endpoint
                ostia_node_id = BasicCenterlineGraph.getCoronaryOstiumNodeIdRelativeToNode(graph, endpoint_node_id)
                for ostium_node_id in ostia_node_id:
                    # continue if the returned ostium is None
                    if ostia_node_id is None:
                        continue
                    # get the path from ostium to endpoint
                    path = networkx.algorithms.shortest_path(graph, ostium_node_id, endpoint_node_id)
                    # create the 3D Slicer open curve file content in json format
                    arr_ = numpy.array(
                        [[graph.nodes[n]['x'], graph.nodes[n]['y'], graph.nodes[n]['z']] for n in path]
                    )
                    # - transform the points according to the transformation matrix
                    arr_ = numpy.concatenate((arr_, numpy.ones((arr_.shape[0], 1))), axis=1).T # 4 x N
                    arr_ = numpy.matmul(affine_transformation_matrix, arr_).T
                    arr_ = arr_[:, :3]
                    # - get other data
                    labels_ = [n for n in path]
                    descriptions_ = [f"{graph.nodes[n]['arterial_tree'].name} {graph.nodes[n]['topology_class'].name}" for n in path]
                    # - make the json string through this utility function
                    file_content_str = numpy_array_to_open_curve_json(arr_, labels_, descriptions_)
                    # create the file
                    if graph.nodes[ostium_node_id]['arterial_tree'] == ArteryPointTree.LEFT:
                        tree = "left"
                    if graph.nodes[ostium_node_id]['arterial_tree'] == ArteryPointTree.RIGHT:
                        tree = "right"
                    f_name = f"{tree}_arterial_segment_{ostium_node_id}_to_{endpoint_node_id}.SlicerOpenCurve.mkr.json"
                    f_path = os.path.join(save_directory, f_name)
                    f = open(f_path, "w")
                    # write the file
                    f.write(file_content_str)
                    f.close()

    @staticmethod
    def convert_to_3dslicer_fiducials(graph: networkx.classes.graph.Graph, save_filename: str, affine_transformation_matrix: numpy.ndarray | None = None) -> None:
        """This function converts the whole graph into a fiducial object (a list of markers)
        that can be loaded directly in 3D Slicer.

        The 3D Slicer fiducials coordinate system is the RAS (Right, Anterior, Superior) coordinate system.
        
        Parameters
        ----------
        graph : networkx.classes.graph.Graph
            The graph to be converted.
        save_filename : str
            The file where the fiducials will be saved.
            It must end with ".SlicerFiducial.mkr.json", else everything after the first "." will be replaced by
            the correct extension.
        affine_transformation_matrix : numpy.ndarray, optional
            The affine transformation matrix to apply to the points, by default None.
            If None, the identity transformation is applied.
        
        Raises
        ------
        FileNotFoundError
            If the save_filename does not exist or cannot be created.
        """
        # Directory handling
        dir_, f_ = os.path.split(save_filename)
        if not os.path.exists(dir_):
            try:
                os.mkdir(dir_)
            except:
                raise FileNotFoundError(f"Directory {dir_} does not exist and cannot be created.")
        # Affine transformation matrix handling
        if affine_transformation_matrix is None:
            affine_transformation_matrix = numpy.identity(4)
        else:
            if affine_transformation_matrix.shape != (4, 4):
                raise ValueError(f"Affine transformation matrix must be a 4x4 matrix, not {affine_transformation_matrix.shape}.")
        # Handle file name
        if not f_.endswith(".SlicerFiducial.mkr.json"):
            f_ = f_.split(".")[0]
            f_ += ".SlicerFiducial.mkr.json"
            save_filename = os.path.join(dir_, f_)
        # Create the 3D Slicer fiducials file content in json format
        arr_ = numpy.array(
            [[graph.nodes[n]['x'], graph.nodes[n]['y'], graph.nodes[n]['z']] for n in graph.nodes]
        )
        # - transform the points according to the transformation matrix
        arr_ = numpy.concatenate((arr_, numpy.ones((arr_.shape[0], 1))), axis=1).T # 4 x N
        arr_ = numpy.matmul(affine_transformation_matrix, arr_).T
        arr_ = arr_[:, :3]
        # - get other data
        labels_ = [n for n in graph.nodes]
        descriptions_ = [f"{graph.nodes[n]['arterial_tree'].name} {graph.nodes[n]['topology_class'].name}" for n in graph.nodes]
        file_content_str = numpy_array_to_fiducials_json(arr_, labels_, descriptions_)
        # create and write the file
        f = open(save_filename, "w")
        f.write(file_content_str)
        f.close()
                    
                    









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






##################
##################
##################

if __name__ == "__main__":
    print("Running 'HCATNetwork.graph' module")
    
    # Load a coronary artery tree graph
    from ..draw.draw import drawCenterlinesGraph2D
    f_prova = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\CAT08\\CenterlineGraphs_FromReference\\dataset00.GML"
    g_ = loadGraph(f_prova)
    
    # Get the anatomic segments
    if 0:
        segments = BasicCenterlineGraph.get_anatomic_segments_ids(g_)
        drawCenterlinesGraph2D(segments)
    
    # Get the anatomic subgraph
    if 0:
        subgraph = BasicCenterlineGraph.get_anatomic_subgraph(g_)
        drawCenterlinesGraph2D(subgraph)

    # Resample the graph
    if 0:
        reampled_graph = BasicCenterlineGraph.resample_coronary_artery_tree(
            graph=g_,
            mm_between_nodes=0.5
        )
        drawCenterlinesGraph2D(reampled_graph)

    # Convert to 3D Slicer open curve

    if 1:
        # Graph
        g_prova = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\ASOCA\\normal_prova\\CTCA\\Normal_01_0.5mm.GML"
        g_ = loadGraph(g_prova)
        #drawCenterlinesGraph2D(g_)

        # Image
        import SimpleITK as sitk
        image_path = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\ASOCA\\normal_prova\\CTCA\\Normal_1.nrrd"
        itkimage = sitk.ReadImage(image_path)
        spacing = itkimage.GetSpacing()
        spacing = numpy.array(spacing)
        origin = itkimage.GetOrigin()
        [sizex, sizey, _] = itkimage.GetSize()

        folder = "C:\\Users\\lecca\\Desktop\\test__slicer_hcatnetwork_"+g_.graph["image_id"].replace("/", "_")
        fname_fiducials= os.path.join(folder, "fiducials_"+g_.graph["image_id"].replace("/", "_")+"_.ext")
        
        '''
        affine_cat08 = numpy.array(
            [  -1.0,    0.0,    0.0,   -origin[0],
                0.0,   -1.0,    0.0,   -origin[1],
                0.0,    0.0,    1.0,    origin[2],
                0.0,    0.0,    0.0,    1.0 ]
        ).reshape((4,4))
        '''
        affine_asoca = numpy.array(
            [  -1.0,    0.0,    0.0,    -origin[0]*2,
                0.0,   -1.0,    0.0,    -origin[1]*2,
                0.0,    0.0,    1.0,    0.0,
                0.0,    0.0,    0.0,    1.0 ]
        ).reshape((4,4))


        if 1:
            BasicCenterlineGraph.convert_to_3dslicer_opencurve(
                graph=g_,
                save_directory=folder,
                affine_transformation_matrix=affine_asoca
            )
        # Convert to 3D Slicer fiducials
        if 1:
            BasicCenterlineGraph.convert_to_3dslicer_fiducials(
                graph=g_,
                save_filename=fname_fiducials,
                affine_transformation_matrix=affine_asoca
            )
    

    