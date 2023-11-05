# HCATNetwork

Heart Coronary Arterial Tree Network handling library based on NetworkX.

<img src="./assets/images/graph_art_example.png">

## Requirements

Developed and tested on Python 3.11 or later. Absolutely requires python >= 3.9.
It is possible to install multiple python versions on the same machine without having them clashing, check out the official website for more infos.

## Installation

To use this package inside your own personal project, we recommend creating a virtual environment in python in which to install this package.

```sh
cd ~/project/folder/
python -m venv env_name
```

Further instructions about virtual environment can be found [here](https://docs.python.org/3/library/venv.html).

Now, install the package with:

```sh
python -m pip install git+https://github.com/AAMIASoftwares-research/HCATNetwork.git
```

or place the following line in your ```requirements.txt``` file:

```txt
git+https://github.com/AAMIASoftwares-research/HCATNetwork.git
```

Now, you should be able to

```py
import HCATNetwork
print(HCATNetwork.edge.SimpleCenterlineEdgeAttributes_KeysList)
```

just as you would with numpy and other packages.

## Usage

### Import and basic usage

```py
import hcatnetwork
# since hcatnetwork is based on networkx,
# you can use any networkx function on the graph
import networkx
```

To create a graph from scratch:

```py
# CREATE A SIMPLE GRAPH
# First, create the graph attributes.
# You can imagine it as a sort of immutable typed dictionary,
# meaning that you can only modify the attributes that you have,
# and they must be of the predefined type. 
graph_attributes = hcatnetwork.graph.SimpleCenterlineGraphAttributes()
graph_attributes["image_id"] = "my_image_id"
graph_attributes["are_left_right_disjointed"] = True
# then, pass it to the graph constructor
graph = hcatnetwork.graph.SimpleCenterlineGraph(graph_attributes)
# Now, nodes of type SimpleCenterlineNode can be added to the graph
# in the same way you would with a networkx graph, except that
# nodes are required to have a certain set of attributes.
node_attributes = hcatnetwork.node.SimpleCenterlineNodeAttributes()
node_attributes["x"] = 0.0
node_attributes["y"] = 1.1
node_attributes["z"] = 2.2
node_attributes["r"] = 3.3
node_attributes["t"] = 0.0
node_attributes["topology_class"] = hcatnetwork.node.ArteryNodeTopology.OSTIUM
node_attributes["tree"] = hcatnetwork.node.ArteryNodeTree.LEFT
graph.add_node("0", node_attributes) # In SimpleCenterlineGraphs, nodes are identified by a str(int) id.
node_attributes["x"] = 1.0
node_attributes["topology_class"] = hcatnetwork.node.ArteryNodeTopology.SEGMENT
graph.add_node("1", node_attributes)
node_attributes["x"] = 2.0
node_attributes["topology_class"] = hcatnetwork.node.ArteryNodeTopology.ENDPOINT
graph.add_node("2", node_attributes)
# Now, edges of type SimpleCenterlineEdge can be added to the graph
edge_attributes = hcatnetwork.edge.SimpleCenterlineEdgeAttributes()
edge_attributes["euclidean_distance"] = 1.0
edge_attributes["weight"] = 1.0
graph.add_edge("0", "1", edge_attributes)
graph_add_edge("1", "2", edge_attributes)
```

This library is based on NetworkX, but it implements some runtime
internal type-checking on the graph's, nodes' and edges' attributes
to ensure that the graph is always a valid and standard.

Being python, it is still possible to bypass this
if you really want to, but it is not recommended.

You can pass graph, edge and nodes attributes as either keyword arguments
(key=value pairs), as a dictionary (dict), as a unpacked dictionary (**dict),
or (recommended) as a typed dictionary of the graph, edge or node type
as in the example above.

### Loading and saving and casting graphs

```py
# LOAD A GRAPH FROM A FILE
# You can load a graph from a file with the following function.
# Supported type is GML.
graph = hcatnetwork.io.load_graph("path/to/file.GML", graph_type=hcatnetwork.graph.SimpleCenterlineGraph)
# or, if you want to load it in plain networkx format
graph = hcatnetwork.io.load_graph("path/to/file.GML", graph_type=networkx.Graph)

# SAVE A GRAPH TO A FILE
# You can save a graph to a file with the following function.
# Supported type is GML.
hcatnetwork.io.save_graph(graph, "path/to/file.GML")

# CAST A NETWORKX GRAPH TO A HCATNetwork GRAPH
# You can cast a networkx graph to a HCATNetwork graph.
# Type checking is performed to ensure that the graph is a valid HCATNetwork graph.
graph_hcatnetwork = hcatnetwork.graph.SimpleCenterlineGraph.from_networkx_graph(graph_networkx)
```

### Visualization

To view a graph, you can use any networkx function, as ```SimpleCenterlineGraph``` is a subclass of ```networkx.Graph```.

For a more advanced visualization, you can use the ```hcatnetwork.draw``` module.

```py
# To visualize it for debugging
draw_simple_centerlines_graph_2d(graph=g_, backend="debug")
# To visualize ot with a visually refined networkx backend
draw_simple_centerlines_graph_2d(graph=g_, backend="networkx")
# To visualize it with hcatnetwork's own backend
draw_simple_centerlines_graph_2d(graph=g_)
```

Our backend offers interactivity and automatic data exploration capabilities.
