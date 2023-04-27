import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx
import numpy

from ..node import SimpleCenterlineNode, ArteryPointTopologyClass, ArteryPointTree

MRK_C_IN_RCA = "firebrick"
MRK_C_IN_LCA = "navy"
MRK_C_OUT_OSTIUM = "green"
MRK_C_OUT_INTERSECT = "gold"
MRK_C_OUT_ENDPOINT = "red"
MRK_C_OUT_DEFAULT = "grey"

EDGE_C_DEFAULT = "black"

def draw2DCenterlinesGraph(graph):
    """Assumes this kind on dictionaries:
        nodes: HCATNetwork.node.SimpleCenterlineNode
        edges: HCATNetwork.edge.BasicEdge
        graph: HCATNetwork.graph.BasicCenterlineGraph
    """
    ax = plt.subplot(111)
    ax.grid(visible=True, zorder=-1)
    # plot nodes
    c_in = []
    c_out = []
    s_in = 25
    s_out = []
    positions = []
    for n in graph.nodes:
        n_ = SimpleCenterlineNode(**(graph.nodes[n]))
        c_in.append(MRK_C_IN_RCA if n_["arterial_tree"].value == ArteryPointTree.RIGHT.value else MRK_C_IN_LCA)
        if n_["topology_class"].value == ArteryPointTopologyClass.OSTIUM.value:
            c_out.append(MRK_C_OUT_OSTIUM)
            s_out.append(2.5)
        elif n_["topology_class"].value == ArteryPointTopologyClass.ENDPOINT.value:
            c_out.append(MRK_C_OUT_ENDPOINT)
            s_out.append(2.5)
        elif n_["topology_class"].value == ArteryPointTopologyClass.INTERSECTION.value:
            c_out.append(MRK_C_OUT_INTERSECT)
            s_out.append(2)
        else:
            c_out.append(MRK_C_OUT_DEFAULT)
            s_out.append(0.0)
        positions.append(n_.getVertexList())
    # - convert to numpy
    c_in  = numpy.array(c_in)
    c_out = numpy.array(c_out)
    s_out = numpy.array(s_out)
    positions = numpy.array(positions)
    # - plot
    below_idx_ = [i for i in range(len(c_out)) if c_out[i] == MRK_C_OUT_DEFAULT]
    above_idx_ = [i for i in range(len(c_out)) if c_out[i] != MRK_C_OUT_DEFAULT]
    ax.scatter( # - below
        positions[below_idx_,0],
        positions[below_idx_,1],
        c=c_in[below_idx_],
        s=s_in,
        zorder=1.5,
        linewidths=s_out[below_idx_],
        edgecolors=c_out[below_idx_]
    )
    ax.scatter( # - above
        positions[above_idx_,0],
        positions[above_idx_,1],
        c=c_in[above_idx_],
        s=s_in*4,
        zorder=2,
        linewidths=s_out[above_idx_],
        edgecolors=c_out[above_idx_]
    )
    # plot undirected edges
    segs = []
    for u_,v_,a in graph.edges(data=True):
        uu = SimpleCenterlineNode(**(graph.nodes[u_])).getVertexList()
        vv = SimpleCenterlineNode(**(graph.nodes[v_])).getVertexList()
        segs.append(numpy.array([uu[:2],vv[:2]]))
    line_segments = LineCollection(segs, zorder=1, linewidth=0.4, color=EDGE_C_DEFAULT)
    ax.add_collection(line_segments)
    # legend
    legend_elements = [
        Line2D([0], [0], marker='o', markerfacecolor=MRK_C_IN_RCA, color="w",                 markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor=MRK_C_IN_LCA, color="w",                 markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=MRK_C_OUT_OSTIUM,    markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=MRK_C_OUT_INTERSECT, markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=MRK_C_OUT_ENDPOINT,  markersize=10, lw=0)
    ]
    ax.legend(
        legend_elements,
        ["RCA",
         "LCA",
         "OSTIA",
         "INTERSECTIONS",
         "ENDPOINTS"]
    )
    # axis
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')
    ax.set_title(graph.graph["image_id"])
    # out
    plt.tight_layout()
    plt.show()

def draw3DCenterlinesGraph(graph):
    """Assumes this kind on dictionaries:
        nodes: HCATNetwork.node.SimpleCenterlineNode
        edges: HCATNetwork.edge.BasicEdge
        graph: HCATNetwork.graph.BasicCenterlineGraph
    """
    ax = plt.subplot(111, projection="3d")
    # plot nodes
    c_in = []
    c_out = []
    s_in = 25
    s_out = []
    positions = []
    for n in graph.nodes:
        n_ = SimpleCenterlineNode(**(graph.nodes[n]))
        c_in.append(MRK_C_IN_RCA if n_["arterial_tree"].value == ArteryPointTree.RIGHT.value else MRK_C_IN_LCA)
        if n_["topology_class"].value == ArteryPointTopologyClass.OSTIUM.value:
            c_out.append(MRK_C_OUT_OSTIUM)
            s_out.append(2.5)
        elif n_["topology_class"].value == ArteryPointTopologyClass.ENDPOINT.value:
            c_out.append(MRK_C_OUT_ENDPOINT)
            s_out.append(2.5)
        elif n_["topology_class"].value == ArteryPointTopologyClass.INTERSECTION.value:
            c_out.append(MRK_C_OUT_INTERSECT)
            s_out.append(2)
        else:
            c_out.append(MRK_C_OUT_DEFAULT)
            s_out.append(0.0)
        positions.append(n_.getVertexList())
    # - convert to numpy
    c_in  = numpy.array(c_in)
    c_out = numpy.array(c_out)
    s_out = numpy.array(s_out)
    positions = numpy.array(positions)
    # - plot
    below_idx_ = [i for i in range(len(c_out)) if c_out[i] == MRK_C_OUT_DEFAULT]
    above_idx_ = [i for i in range(len(c_out)) if c_out[i] != MRK_C_OUT_DEFAULT]
    ax.scatter( # - below
        positions[below_idx_,0],
        positions[below_idx_,1],
        positions[below_idx_,2],
        c=c_in[below_idx_],
        s=s_in,
        zorder=1.5,
        linewidths=s_out[below_idx_],
        edgecolors=c_out[below_idx_]
    )
    ax.scatter( # - above
        positions[above_idx_,0],
        positions[above_idx_,1],
        positions[above_idx_,2],
        c=c_in[above_idx_],
        s=s_in*4,
        zorder=2,
        linewidths=s_out[above_idx_],
        edgecolors=c_out[above_idx_]
    )
    # plot undirected edges
    segs = []
    for u_,v_,a in graph.edges(data=True):
        uu = SimpleCenterlineNode(**(graph.nodes[u_])).getVertexList()
        vv = SimpleCenterlineNode(**(graph.nodes[v_])).getVertexList()
        segs.append(numpy.array([uu[:3],vv[:3]]))
    line_segments = Line3DCollection(segs, zorder=1, linewidth=0.4, color=EDGE_C_DEFAULT)
    ax.add_collection(line_segments)
    # legend
    legend_elements = [
        Line2D([0], [0], marker='o', markerfacecolor=MRK_C_IN_RCA, color="w",                 markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor=MRK_C_IN_LCA, color="w",                 markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=MRK_C_OUT_OSTIUM,    markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=MRK_C_OUT_INTERSECT, markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=MRK_C_OUT_ENDPOINT,  markersize=10, lw=0)
    ]
    ax.legend(
        legend_elements,
        ["RCA",
         "LCA",
         "OSTIA",
         "INTERSECTIONS",
         "ENDPOINTS"]
    )
    # axis
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.set_zlabel("mm")
    ax.grid(color='gray', linestyle='dashed')
    ax.set_title(graph.graph["image_id"])
    # out
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    f_prova = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\CAT08\\dataset00\dataset00.GML"
    from ..graph import loadGraph
    g_ = loadGraph(f_prova)
    draw2DCenterlinesGraph(graph=g_)
    draw3DCenterlinesGraph(graph=g_)