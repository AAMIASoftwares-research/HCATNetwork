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
and use: python -m hcatnetwork.graph.graph
"""
from __future__ import annotations

from enum import Enum, auto
import numpy
import networkx

from ..core.core import CoreDict, TYPE_NAME_TO_TYPE_DICT
from ..node.node import SimpleCenterlineNodeAttributes, ArteryNodeTopology, ArteryNodeSide
from ..edge.edge import SimpleCenterlineEdgeAttributes

###################################
# LOADING and SAVING to text files
###################################



#########################
# Simple Centerline Graph
#########################

class SimpleCenterlineGraphAttributes(CoreDict):
    """SimpleCenterlineGraphAttributes

    This is the dictionary type for the SimpleCenterlineGraph attributes.

    Keys
    ----
    image_id : str
        The image id of the image from which the graph was extracted.
        This string has no fixed format, and can be anything, but it should be clear enough
        to identify the image, and possibly be consistent for images coming from the same source/dataset.
    are_left_right_disjointed : bool
        ``True`` if the left and right coronary trees are disjointed, ``False`` otherwise.
    """
    image_id: str
    are_left_right_disjointed: bool
                    
class SimpleCenterlineGraph(networkx.classes.graph.Graph):

    def __init__(self, attributes_dict: SimpleCenterlineGraphAttributes | dict | None = None, **attributes_kwargs):
        """The basic centerline graph, child of NetworkX.Graph.

        The graph should contain:
            * nodes of type hcatnetwork.node.SimpleCenterlineNodeAttributes
            * edges of type hcatnetwork.edge.SimpleCenterlineEdgeAttributes
        
        The graph performs automatic type checking upon graph initialization adding of nodes and edges, so that only valid nodes and edges can be added to the graph.

        Both trees (left and right) are stored in the same Graph structure.
        In some patients, it could happen that the left and right subgraphs are not disjointed, hence the need to have just one graph.
        
        SimpleCenterlineGraph, and all its algorithms and methods, assumes that the coronary ostia are disjointed, meaning that the 
        coronary ostia should not be overlapping or too near to each other. No control is actively performed to check this condition.

        Parameters
        ----------
        attr : **dict or key=value pairs, same contained in SimpleCenterlineGraphAttributes. Mandatory attributes are:
            image_id : str
                The image id of the image from which the graph was extracted.
                This string has no fixed format, and can be anything, but it should be clear enough
                to identify the image, and possibly be consistent for images coming from the same source/dataset.
            
            are_left_right_disjointed : bool
                ``True`` if the left and right coronary trees are disjointed, ``False`` otherwise.
        
        See Also
        --------
        NetworkX.Graph, hcatnetwork.graph.SimpleCenterlineGraph, hcatnetwork.node.SimpleCenterlineNodeAttributes, hcatnetwork.edge.SimpleCenterlineEdgeAttributes

        Examples
        --------
        Call the class ike this:
        > g_dict = SimpleCenterlineGraphAttributes()
        > g_dict["image_id"] = f"example/name"
        > g_dict["are_left_right_disjointed"] = True
        > graph = SimpleCenterlineGraph(g_dict)
        > # or
        > graph = SimpleCenterlineGraph(**g_dict)

        passing a ``dict`` or a ``**dict`` is equivalent.

        Or like this
        > dict_ = {}
        > dict_["image_id"] = f"example/name"
        > dict_["are_left_right_disjointed"] = True
        > graph = SimpleCenterlineGraph(g_dict)

        Or like this
        > dict_ = {}
        > dict_["image_id"] = f"example/name"
        > dict_["are_left_right_disjointed"] = True
        > dict_ = SimpleCenterlineGraphAttributes(**dict_)
        > graph = SimpleCenterlineGraph(g_dict)

        # Or like this
        > graph = SimpleCenterlineGraph(
        >     image_id="name of image",
        >    are_left_right_disjointed=True
        > )

        While this gives an ERROR
        > g_dict = SimpleCenterlineGraphAttributes()
        > g_dict["image_id"] = 5                      # error here
        > g_dict["are_left_right_disjointed"] = "yes" # error here
        > graph = SimpleCenterlineGraph(g_dict)

        also here
        >g_dict = {"image_id": 55, "are_left_right_disjointed": True}
        >graph = SimpleCenterlineGraph(g_dict)

        also here
        >graph = SimpleCenterlineGraph(image_id="name of image", are_left_right_disjointed="yes")

        # also this
        > g_dict = SimpleCenterlineGraphAttributes()
        > g_dict["image_id"] = f"example/name"             # correct
        > g_dict["are_left_right_disjointed"] = True       # correct
        > graph = SimpleCenterlineGraph(g_dict)            # correct
        > graph.graph["invalid key"] = 5                   # error here
        > graph.graph["are_left_right_disjointed"] = "yes" # error here

        """
        # ################################
        # Input handling and type checking
        # ################################
        # In this way, a user can pass a dictionary, or key=value pairs, or a **dict
        if attributes_dict is None and len(attributes_kwargs) == 0:
            raise ValueError(f"Please provide an input, which can be a fully-populated SimpleCenterlineGraphAttributes dictionary, or a **dict, or key=value pairs corresponding to SimpleCenterlineGraphAttributes. Available attributes are: \n{SimpleCenterlineGraphAttributes.__annotations__}")
        if attributes_dict is None:
            # Let SimpleCenterlineGraphAttributes do the type checking (thanks to the CoreDict class)
            attributes_dict = SimpleCenterlineGraphAttributes(**attributes_kwargs)
        if isinstance(attributes_dict, dict):
            # If attributes_dict is a dict, it must be casted to a SimpleCenterlineGraphAttributes
            if not isinstance(attributes_dict, SimpleCenterlineGraphAttributes):
                attributes_dict = SimpleCenterlineGraphAttributes(**attributes_dict)
        if not isinstance(attributes_dict, SimpleCenterlineGraphAttributes):
            # If attributes_dict is not None, it must be a SimpleCenterlineGraphAttributes
            raise TypeError(f"attributes_dict must be of type SimpleCenterlineGraphAttributes or dict or None, not {type(attributes_dict)}.")
        # Now, any provided input is a SimpleCenterlineGraphAttributes
        # Check for completeness of the attributes
        if not attributes_dict.is_full():
            raise ValueError(f"attributes_dict must be a valid SimpleCenterlineGraphAttributes dictionary. provided attributes are {attributes_dict}. Mandatory attributes and types are: \n{SimpleCenterlineGraphAttributes.__annotations__}")
        
        # ##############
        # Make the graph
        # ##############
        # Since type checking already occurred, we can safely pass the attributes_dict to the super class
        super().__init__(**attributes_dict)

        # #####################
        # On-line type checking
        # #####################
        # Check mandatory attributes of graph
        # by making the self.graph dictionary (originally of type dict) a SimpleCenterlineGraphAttributes dictionary, so that it can inherit the setitem from CoreDict
        # self.graph is created by super().__init__()
        self._attributes_type = SimpleCenterlineGraphAttributes()
        self._attributes_keys_list = [k for k in self._attributes_type.keys()]
        self._attributes_types_dict = {k:v if isinstance(v, type) else TYPE_NAME_TO_TYPE_DICT[v] for k,v in self._attributes_type.__annotations__.items()}
        self.graph = SimpleCenterlineGraphAttributes(**self.graph) # incorporated type checking
        # Check mandatory attributes of nodes
        self._simple_centerline_node_attributes_type = SimpleCenterlineNodeAttributes()
        self._simple_centerline_node_attributes_keys_list = [k for k in self._simple_centerline_node_attributes_type.keys()]
        self._simple_centerline_node_attributes_types_dict = self._simple_centerline_node_attributes_type.__annotations__
        # Check mandatory attributes of edges
        self._simple_centerline_edge_attributes_type = SimpleCenterlineEdgeAttributes()
        self._simple_centerline_edge_attributes_keys_list = [k for k in self._simple_centerline_edge_attributes_type.keys()]
        self._simple_centerline_edge_attributes_types_dict = self._simple_centerline_edge_attributes_type.__annotations__
        
    def _check_node_id_type(self, node_id) -> None:
        """Check if the node id is of the correct type
        In this case, the correct type is a string.
        """
        if not isinstance(node_id, str):
            raise ValueError(f"Node id {node_id} must be a string, not a {isinstance(node_id)}.")

    def add_node(self, node_for_adding: str, attributes_dict: SimpleCenterlineNodeAttributes | dict | None = None, **attributes_kwargs):
        """Add a single node node_for_adding and update node attributes.

        The node's attributes dictionary should be of type hcatnetwork.node.SimpleCenterlineNodeAttributes or a dictionary of
        attributes must be provided that has all attributes of a hcatnetwork.node.SimpleCenterlineNodeAttributes.

        Parameters
        ----------
        node_for_adding : str
            The node id.
        attr : **dict or **SimpleCenterlineNodeAttributes or key=value pairs.
            Dictionary of node attributes. Dictionary must have all attributes of a hcatnetwork.node.SimpleCenterlineNodeAttributes.
            Alternatively, it can be a **dict, where dict is of type hcatnetwork.node.SimpleCenterlineNodeAttributes.
        """
        # Node id type checking
        self._check_node_id_type(node_for_adding)
        # Node attributes type checking
        if attributes_dict is None and len(attributes_kwargs) == 0:
            raise ValueError(f"Please provide an input, which can be a fully-populated SimpleCenterlineNodeAttributes dictionary, or a dict, **dict or key=value pairs corresponding to SimpleCenterlineNodeAttributes.\nAvailable attributes are:\n{SimpleCenterlineNodeAttributes.__annotations__}")
        if attributes_dict is None:
            # Let SimpleCenterlineNodeAttributes do the type checking (thanks to the CoreDict class)
            attributes_dict = SimpleCenterlineNodeAttributes(**attributes_kwargs)
        if isinstance(attributes_dict, dict):
            # If attributes_dict is a dict, it must be casted to a SimpleCenterlineNodeAttributes
            if not isinstance(attributes_dict, SimpleCenterlineNodeAttributes):
                attributes_dict = SimpleCenterlineNodeAttributes(**attributes_dict)
        if not isinstance(attributes_dict, SimpleCenterlineNodeAttributes):
            # If attributes_dict is not None, it must be a SimpleCenterlineNodeAttributes
            raise TypeError(f"attributes_dict must be of type SimpleCenterlineNodeAttributes or dict or None, not {type(attributes_dict)}.")
        # Now, any provided input is a SimpleCenterlineNodeAttributes
        # Check for completeness of the attributes
        if not attributes_dict.is_full():
            raise ValueError(f"attributes_dict must be a valid SimpleCenterlineNodeAttributes dictionary. provided attributes are {attributes_dict}. Mandatory attributes and types are: \n{SimpleCenterlineNodeAttributes.__annotations__}")
        # All checks passed, add the node
        super().add_node(node_for_adding, **attributes_dict)
    
    def add_edge(self, u_of_edge: str, v_of_edge: str, attributes_dict: SimpleCenterlineEdgeAttributes | dict | None = None, **attributes_kwargs):
        """Add an edge between u_of_edge and v_of_edge.

        The edge should be of type hcatnetwork.edge.SimpleCenterlineEdgeAttributes or a dictionary of
        attributes must be provided that has all attributes of a hcatnetwork.edge.SimpleCenterlineEdgeAttributes.

        Parameters
        ----------
        u_of_edge : str
            The id of the first node of the edge.
        v_of_edge : str
            The id of the second node of the edge.
        attr : **dict or **SimpleCenterlineEdgeAttributes or key=value pairs.
            Dictionary of edge attributes. Dictionary must have all attributes of a hcatnetwork.edge.SimpleCenterlineEdgeAttributes.
            Alternatively, it can be a **dict, where dict is of type hcatnetwork.edge.SimpleCenterlineEdgeAttributes.
        """
        # Edge source and target nodes id type checking
        self._check_node_id_type(u_of_edge)
        self._check_node_id_type(v_of_edge)
        # Edge attributes type checking
        if attributes_dict is None and len(attributes_kwargs) == 0:
            raise ValueError(f"Please provide an input, which can be a fully-populated SimpleCenterlineEdgeAttributes dictionary, or a dict, **dict or key=value pairs corresponding to SimpleCenterlineEdgeAttributes.\nAvailable attributes are:\n{SimpleCenterlineEdgeAttributes.__annotations__}")
        if attributes_dict is None:
            # Let SimpleCenterlineEdgeAttributes do the type checking (thanks to the CoreDict class)
            attributes_dict = SimpleCenterlineEdgeAttributes(**attributes_kwargs)
        if isinstance(attributes_dict, dict):
            # If attributes_dict is a dict, it must be casted to a SimpleCenterlineEdgeAttributes
            if not isinstance(attributes_dict, SimpleCenterlineEdgeAttributes):
                attributes_dict = SimpleCenterlineEdgeAttributes(**attributes_dict)
        if not isinstance(attributes_dict, SimpleCenterlineEdgeAttributes):
            # If attributes_dict is not None, it must be a SimpleCenterlineEdgeAttributes
            raise TypeError(f"attributes_dict must be of type SimpleCenterlineEdgeAttributes or dict or None, not {type(attributes_dict)}.")
        # Now, any provided input is a SimpleCenterlineEdgeAttributes
        # Check for completeness of the attributes
        if not attributes_dict.is_full():
            raise ValueError(f"attributes_dict must be a valid SimpleCenterlineEdgeAttributes dictionary. provided attributes are {attributes_dict}. Mandatory attributes and types are: \n{SimpleCenterlineEdgeAttributes.__annotations__}")
        # All checks passed, add the edge
        super().add_edge(u_of_edge, v_of_edge, **attributes_dict)

    def get_relative_coronary_ostia_node_id(self, node_id: str) -> tuple[str] | tuple[str, str]:
        """Get the coronary ostium node id relative to the node with id node_id.

        Parameters
        ----------
        node_id : str
            The id of the node for which the relative ostium is to be found.

        Returns
        -------
            tuple[str] or tuple[str, str]
                Depending on whether the node is associated with one or both arterial trees.
                The 2-tuple always contains the left ostium node id as the first element, and the right ostium node id as the second element.
        """
        if not node_id in [id for id in self.nodes]:
            raise ValueError(f"Node with id \"{node_id}\" is not in graph.")
        node = self.nodes[node_id]
        # Node is a coronary ostium
        if node['topology'].value == ArteryNodeTopology.OSTIUM.value:
            return tuple([node_id])
        # Node is not a coronary ostium
        # The node could be associated with either one or both arterial trees.
        # There should be no nodes asssociated with no artrial trees.
        if node['side'].value != ArteryNodeSide.RL.value:
            # The node is associated with a single tree
            for n in self.nodes:
                if self.nodes[n]['side'].value == node['side'].value and self.nodes[n]['topology'].value == ArteryNodeTopology.OSTIUM.value:
                    return tuple([n])
        else:
            # The node is associated with both arterial trees
            count_hits_ = 0
            left_ostium_n, right_ostium_n = None, None
            for n in self.nodes:
                if self.nodes[n]['topology'].value == ArteryNodeTopology.OSTIUM.value:
                    if self.nodes[n]['side'].value == ArteryNodeSide.LEFT.value:
                        left_ostium_n = n
                    elif self.nodes[n]['side'].value == ArteryNodeSide.RIGHT.value:
                        right_ostium_n = n
                    else:
                        raise RuntimeError(f"Node {n} is a coronary ostium associated with no arterial tree (nor left, nor right).")
                    count_hits_ += 1
                    if count_hits_ == 2:
                        return tuple([left_ostium_n, right_ostium_n])
        # If the code reaches this point, it means that the node is not associated with any arterial tree
        raise RuntimeError(f"Node {n} is a coronary ostium associated with no arterial tree (nor left, nor right).")

    def get_coronary_ostia_node_id(self) -> tuple[str, str]:
        """Gets the left and right coronary ostia node ids.

        Returns
        -------
        tuple[str, str]:
            2-tuple of strings, where the first element is the left ostium node id, and the second element is the right ostium node id.
            If an ostium cannot be found, the element will be set to None. This should never happen.
        """
        count_hits_ = 0
        left_ostium_n, right_ostium_n = None, None
        for n in self.nodes:
            if self.nodes[n]['topology'].value == ArteryNodeTopology.OSTIUM.value:
                if self.nodes[n]['side'].value == ArteryNodeSide.LEFT.value:
                    left_ostium_n = n
                elif self.nodes[n]['side'].value == ArteryNodeSide.RIGHT.value:
                    right_ostium_n = n
                else:
                    raise RuntimeError(f"Node {n} is a coronary ostium associated with no arterial tree (nor left, nor right).")
                count_hits_ += 1
                if count_hits_ == 2:
                    return tuple([left_ostium_n, right_ostium_n])
        # If the code reaches this point, it means that the graph does not have two ostia, so return the tuple with a None element
        return tuple([left_ostium_n, right_ostium_n])
    
    def get_anatomic_segments(self) -> list[tuple[str]]:
        """Gets the segments of the graph, starting from the coronary ostia.

        The segments are returned as a list of tuples, each containing the start and end node id of
        a segment delimited by either an ostium, intersection, or endpoint.
        A segment is a piece of graph connecting an ostium or intersection to an intersection or an endpoint,
        without any other landmark (ostium, intersection, endpoint) in between.

        Returns
        -------
        list[tuple[str]]
            A list of tuples, each containing the start and end node id of a segment delimited by either an ostium, intersection, or endpoint.
        
        """  
        segments = []
        for start_node_id in self.get_coronary_ostia_node_id():
            # Fore each node, get the successor of the node starting from the ostium
            breadth_first_search_successors_from_ostium = {key: val for (key, val) in networkx.bfs_successors(self, start_node_id)}
            next_buffer_list: list = breadth_first_search_successors_from_ostium[start_node_id]
            start_buffer_list: list = [start_node_id]
            while len(next_buffer_list) != 0:
                # Get the next node
                current_node_id = next_buffer_list[0]
                # Since you considered this, remove it from the list
                next_buffer_list.pop(0)
                # Walk up to the next landmark (intersection, or endpoint)
                stay_in_loop_ = current_node_id in breadth_first_search_successors_from_ostium # endpoint check
                stay_in_loop_ = stay_in_loop_ and len(breadth_first_search_successors_from_ostium[current_node_id]) == 1 # intersection check
                while stay_in_loop_:
                    # Jump to the next node
                    current_node_id = breadth_first_search_successors_from_ostium[current_node_id][0]
                    # Check if the node is a landmark (intersection, or endpoint)
                    stay_in_loop_ = current_node_id in breadth_first_search_successors_from_ostium # endpoint check
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

    def get_anatomic_subgraph(self) -> networkx.classes.graph.Graph:
        """Gets the anatomic subgraph of the graph, meaning the graph of just coronary ostia, intersections and endpoints.

        In the returned graph, the distance between nodes is not the euclidean distance between the nodes,
        but the euclidean length of the segment.

        Returns
        -------
        networkx.classes.graph.Graph
            The simplified graph.
        
        """
        # Create the subgraph, copying the info from the original graph
        subgraph = networkx.Graph(**self.graph)
        subgraph.graph["image_id"] += " - anatomic subgraph"
        # Get the segments
        segments = self.get_anatomic_segments()
        # Add the nodes
        for segment in segments:
            if not segment[0] in subgraph.nodes:
                subgraph.add_node(segment[0], **self.nodes[segment[0]])
            if not segment[1] in subgraph.nodes:
                subgraph.add_node(segment[1], **self.nodes[segment[1]])
        # Add the edges
        for segment in segments:
            edge_features = SimpleCenterlineEdgeAttributes()
            edge_features["euclidean_distance"] = networkx.algorithms.shortest_path_length(self, segment[0], segment[1], weight="euclidean_distance")
            edge_features.update_weight_from_euclidean_distance()
            subgraph.add_edge(segment[0], segment[1], **edge_features)
        # Done
        return subgraph
        
    def resample(self, mm_between_nodes: float = 0.5) -> SimpleCenterlineGraph:
        """Resamples the coronary artery tree so that two connected points are on average mm_between_nodes millimeters apart.

        The tree is resampled so that the absolute position of coronary ostia, intersections and endpoints is preserved.
        The position of the nodes between these landmarks can vary, and so can radius data, which is interpolated (linear).
        
        Parameters
        ----------
        mm_between_nodes : float, optional
            The average distance between two connected points, in millimeters, by default 0.5.

        Returns
        -------
        SimpleCenterlineGraph
            The resampled graph.
        
        """
        # Create the new graph, copying the info from the original graph
        graph_new = SimpleCenterlineGraph(**self.graph)
        graph_new.graph["image_id"] += f" - resampled {mm_between_nodes:3.3f} mm"
        # Get the anatomic segments of the original graph
        segments = self.get_anatomic_segments()
        # - consider each segment only once, needed for patients with non-disjointed left and right trees
        # - we do not want to resample the same segment twice
        segments = list(set(segments)) 
        untouchable_node_ids = [a for (a,b) in segments] + [b for (a,b) in segments]
        # Resample each segment
        node_id_counter = 0
        for n0, n1 in segments:
            # Get the number of nodes to put into this segment (counting also n0 and n1)
            # Therefore, the number of nodes is always at least 2.
            length = networkx.algorithms.shortest_path_length(self, n0, n1, weight="euclidean_distance")
            n_nodes = max(
                [2, int(length / mm_between_nodes)]
            )
            # Resample the segment
            if n_nodes == 2:
                # Just add the two nodes
                if not n0 in graph_new.nodes:
                    graph_new.add_node(n0, **self.nodes[n0])
                if not n1 in graph_new.nodes:
                    graph_new.add_node(n1, **self.nodes[n1])
                # Add the edge
                # Here, the edge's property "euclidean_distance" is the actual distance between the nodes.
                if not graph_new.has_edge(n0, n1):
                    edge_features = SimpleCenterlineEdgeAttributes()
                    n0_p = numpy.array([self.nodes[n0]["x"], self.nodes[n0]["y"], self.nodes[n0]["z"]])
                    n1_p = numpy.array([self.nodes[n1]["x"], self.nodes[n1]["y"], self.nodes[n1]["z"]])
                    edge_features["euclidean_distance"] = numpy.linalg.norm(n0_p - n1_p)
                    edge_features.update_weight_from_euclidean_distance()
                    graph_new.add_edge(n0, n1, **edge_features)
            else:
                distances_to_sample = numpy.linspace(0, length, n_nodes)
                nodes_ids_to_connect_in_sequence_list = []
                # First and last node will be n0 and n1, respectively
                # Add the first node
                if not n0 in graph_new.nodes:
                    graph_new.add_node(n0, **self.nodes[n0])
                nodes_ids_to_connect_in_sequence_list.append(n0)
                # Add the middle nodes
                # - get all nodes in the segment
                nodes_in_segment = networkx.algorithms.shortest_path(self, n0, n1)
                nodes_in_segment_distances_from_n0 = {n__: networkx.algorithms.shortest_path_length(self, n0, n__, weight="euclidean_distance") for n__ in nodes_in_segment}
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
                    p_n_ = numpy.array([self.nodes[node_]["x"], self.nodes[node_]["y"], self.nodes[node_]["z"]])
                    p_n_b_ = numpy.array([self.nodes[node_before_]["x"], self.nodes[node_before_]["y"], self.nodes[node_before_]["z"]])
                    proportion_ = (d - nodes_in_segment_distances_from_n0[node_before_]) / (nodes_in_segment_distances_from_n0[node_] - nodes_in_segment_distances_from_n0[node_before_])
                    position_new_ = p_n_b_ + (p_n_ - p_n_b_) * proportion_
                    radius_new_ = self.nodes[node_before_]["r"] + (self.nodes[node_]["r"] - self.nodes[node_before_]["r"]) * proportion_
                    # Add the node to the graph and to the list to then connect
                    while (str(node_id_counter) in graph_new.nodes) or (str(node_id_counter) in untouchable_node_ids):
                        # make sure no new nodes have the same id
                        node_id_counter += 1
                    node_features = SimpleCenterlineNodeAttributes()
                    node_features.set_vertex(position_new_)
                    node_features["r"] = radius_new_
                    node_features["t"] = 0.0
                    node_features["topology"] = ArteryNodeTopology.SEGMENT
                    node_features["side"] = self.nodes[node_before_]["side"]
                    graph_new.add_node(str(node_id_counter), **node_features)
                    nodes_ids_to_connect_in_sequence_list.append(str(node_id_counter))
                # Add the last node
                if not n1 in graph_new.nodes:
                    graph_new.add_node(n1, **self.nodes[n1])
                nodes_ids_to_connect_in_sequence_list.append(n1)
                # Connect the nodes
                for i in range(len(nodes_ids_to_connect_in_sequence_list) - 1):
                    n0 = nodes_ids_to_connect_in_sequence_list[i]
                    n1 = nodes_ids_to_connect_in_sequence_list[i + 1]
                    if not graph_new.has_edge(n0, n1):
                        edge_features = SimpleCenterlineEdgeAttributes()
                        n0_p = numpy.array([graph_new.nodes[n0]["x"], graph_new.nodes[n0]["y"], graph_new.nodes[n0]["z"]])
                        n1_p = numpy.array([graph_new.nodes[n1]["x"], graph_new.nodes[n1]["y"], graph_new.nodes[n1]["z"]])
                        edge_features["euclidean_distance"] = numpy.linalg.norm(n0_p - n1_p)
                        edge_features.update_weight_from_euclidean_distance()
                        graph_new.add_edge(n0, n1, **edge_features)
        return graph_new

    @staticmethod
    def from_networkx_graph(graph: networkx.classes.graph.Graph) -> SimpleCenterlineGraph:
        """Creates a SimpleCenterlineGraph from a networkx.classes.graph.Graph.
        
        Parameters
        ----------
        graph : networkx.classes.graph.Graph
            The graph to convert.
        
        Returns
        -------
        SimpleCenterlineGraph
            The converted graph.
        """
        # Create the new graph, copying the info from the original graph
        graph_new = SimpleCenterlineGraph(**graph.graph)
        # Add the nodes
        for n in graph.nodes:
            graph_new.add_node(n, **graph.nodes[n])
        # Add the edges
        for n0, n1 in graph.edges:
            graph_new.add_edge(n0, n1, **graph.edges[n0, n1])
        return graph_new





############################
# Coronary Artery Tree Graph
############################
"""Coronary Artery Tree Graph

This is the most complete graph, holding everything needed for representing a coronary artery tree.
Both trees (l and r) are stored in the same graph.
In some patients, it could happen that the left and right subgraphs are not disjointed, hence the need to have just one graph.

For the future. ######################

"""

# Heart dominance is described by which coronary artery branch gives off the posterior descending artery and supplies the inferior wall, and is characterized as left, right, or codominant
class HeartDominance(Enum):
    LEFT = auto()
    RIGHT= auto()
    CODOMINANT = auto()



###########
# Add types
###########

TYPE_NAME_TO_TYPE_DICT["SimpleCenterlineGraph"] = SimpleCenterlineGraph
TYPE_NAME_TO_TYPE_DICT["SimpleCenterlineGraphAttributes"] = SimpleCenterlineGraphAttributes
TYPE_NAME_TO_TYPE_DICT["HeartDominance"] = HeartDominance





##################
##################
##################

if __name__ == "__main__":
    print("Running 'hcatnetwork.graph' module")

    # Create a graph from scratch
    if 0:
        graph = SimpleCenterlineGraph(
            image_id="name of image",
            are_left_right_disjointed=True
        )
        quit()
    
    # Load a coronary artery tree graph and plot it
    import os
    from ..io.io import load_graph
    from ..draw.draw import draw_simple_centerlines_graph_2d

    f_prova = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\CAT08\\CenterlineGraphs_FromReference\\dataset00.GML"
    try:
        g_ = load_graph(f_prova, output_type=SimpleCenterlineGraph)
    except TypeError as e:
        g_ = load_graph(f_prova, output_type=networkx.classes.graph.Graph)
        g_.graph["are_left_right_disjointed"] = bool(int(g_.graph["are_left_right_disjointed"]))
        g_ = SimpleCenterlineGraph.from_networkx_graph(g_)
    
    # Draw the graph
    if 0:
        draw_simple_centerlines_graph_2d(g_)
        draw_simple_centerlines_graph_2d(g_, backend="networkx")
        draw_simple_centerlines_graph_2d(g_, backend="debug")

    # Get the anatomic segments
    if 0:
        segments = g_.get_anatomic_segments()
        print(segments)
    
    # Get the anatomic subgraph
    if 0:
        subgraph = g_.get_anatomic_subgraph()
        draw_simple_centerlines_graph_2d(subgraph)

    # Resample the graph
    if 1:
        import time
        _t_s = time.time()
        reampled_graph = g_.resample_coronary_artery_tree(
            mm_between_nodes=0.5
        )
        _t_e = time.time()
        print(f"Resampling took {_t_e - _t_s} seconds.")
        draw_simple_centerlines_graph_2d(reampled_graph)

    # Convert to 3D Slicer open curve

    if 0:
        from ..utils.slicer import convert_graph_to_3dslicer_fiducials, convert_graph_to_3dslicer_opencurve
        # Graph
        g_prova = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\ASOCA\\normal_prova\\CTCA\\Normal_01_0.5mm.GML"
        g_ = load_graph(g_prova)
        # draw_simple_centerlines_graph_2d(g_)

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
        
        
        affine_cat08 = numpy.array(
            [  -1.0,    0.0,    0.0,   -origin[0],
                0.0,   -1.0,    0.0,   -origin[1],
                0.0,    0.0,    1.0,    origin[2],
                0.0,    0.0,    0.0,    1.0 ]
        ).reshape((4,4))
        
        affine_asoca = numpy.array(
            [  -1.0,    0.0,    0.0,    -origin[0]*2,
                0.0,   -1.0,    0.0,    -origin[1]*2,
                0.0,    0.0,    1.0,    0.0,
                0.0,    0.0,    0.0,    1.0 ]
        ).reshape((4,4))


        if 1:
            convert_graph_to_3dslicer_opencurve(
                graph=g_,
                save_directory=folder,
                affine_transformation_matrix=affine_asoca
            )
        # Convert to 3D Slicer fiducials
        if 1:
            convert_graph_to_3dslicer_fiducials(
                graph=g_,
                save_filename=fname_fiducials,
                affine_transformation_matrix=affine_asoca
            )
    

    