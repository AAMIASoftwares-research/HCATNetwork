"""IO module for HCATNetwork.

Loading and Saving:
The GML file tipe is chosen to hold graph data, as it is widely supported and does not rely
on insecure python libraries as other do. NetworkX is used as a basic load/save interface.

To run as a module, activate the venv, go inside the HCATNetwork parent directory,
and use: python -m hcatnetwork.graph.graph
"""

from ..graph.graph import SimpleCenterlineGraph

########
# SAVING
########

from .io import save_graph as io_save_graph

def save_graph(
        graph: SimpleCenterlineGraph,
        file_path: str):
    """Saves the graph in GML format using the networkx interface.

    Same as hcatnetwork.io.io.save_graph()

    If some graph, node or edge features are:
        * numpy arrays
        * lists or multidimensional lists of basic python types

    Data are converted into json strings with json.dumps() before saving the graph in GML format.
    The file also contains everything needed to convert back the json strings into the original data types.
    
    Parameters
    ----------
    graph : SimpleCenterlineGraph
        The graph to be saved.
    file_path : str
        The path to the file where the graph will be saved as GML.

    Raises
    ------
    ValueError
        If any feature of the graph, node or edge is None.
    
    See Also
    --------
    hcatnetwork.io.io.save_graph()
    """
    io_save_graph(graph, file_path)



#########
# LOADING
#########

from .io import load_enums, load_graph as io_load_graph

def load_graph(file_path: str) -> SimpleCenterlineGraph:
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
    hcatnetwork.io.io.load_graph()
    """
    graph = io_load_graph(file_path)
    graph = SimpleCenterlineGraph.from_networkx_graph(graph)
    return graph

