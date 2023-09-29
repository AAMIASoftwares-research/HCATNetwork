"""Drawing utility for the HCATNetwork module.
Graphs can be statically drawn in 2D or 3D.
plots are interactive, meaning that the user can click on the nodes and edges to get information about them.

Useful development resources:
https://matplotlib.org/stable/users/explain/artists/index.html
https://matplotlib.org/stable/users/explain/figure/event_handling.html
https://mpl-interactions.readthedocs.io/en/stable/
https://matplotlib.org/stable/users/explain/figure/event_handling.html

To run as a module, activate the venv, go inside the HCATNetwork parent directory,
and use: python -m HCATNetwork.draw.draw
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import CirclePolygon
from matplotlib.collections import LineCollection, EllipseCollection, CircleCollection
from matplotlib.backend_bases import Event, MouseEvent
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx
import numpy

from ..node import SimpleCenterlineNode, ArteryPointTopologyClass, ArteryPointTree
from .styles import (COLOR_NODE_IN_LCA, COLOR_NODE_IN_RCA, COLOR_NODE_OUT_DEFAULT, 
                     COLOR_NODE_OUT_END, COLOR_NODE_OUT_CROSS, COLOR_NODE_OUT_START, 
                     COLOR_EDGE_DEFAULT, COLOR_NODE_IN_BOTH)



def drawCenterlinesGraph2D(graph: networkx.Graph):
    """Assumes this kind on dictionaries:
        nodes: HCATNetwork.node.SimpleCenterlineNode
        edges: HCATNetwork.edge.BasicEdge
        graph: HCATNetwork.graph.BasicCenterlineGraph
    """
    FIG_DPI = 120
    fig, ax = plt.subplots(dpi=FIG_DPI)
    ax.set_aspect(aspect='equal', adjustable='datalim', anchor='C')
    ########
    # NODES
    ########
    patch_n_color_map = {
        ArteryPointTree.RIGHT.value: COLOR_NODE_IN_RCA,
        ArteryPointTree.LEFT.value: COLOR_NODE_IN_LCA,
        ArteryPointTree.RL.value: COLOR_NODE_IN_BOTH
    }
    edge_n_color_map = {
        ArteryPointTopologyClass.OSTIUM.value: COLOR_NODE_OUT_START,
        ArteryPointTopologyClass.SEGMENT.value: COLOR_NODE_OUT_DEFAULT,
        ArteryPointTopologyClass.ENDPOINT.value: COLOR_NODE_OUT_END,
        ArteryPointTopologyClass.INTERSECTION.value: COLOR_NODE_OUT_CROSS
    }
    edgewidth_n_map = {
        ArteryPointTopologyClass.OSTIUM.value: 1.5,
        ArteryPointTopologyClass.SEGMENT.value: 0.0,
        ArteryPointTopologyClass.ENDPOINT.value: 1.5,
        ArteryPointTopologyClass.INTERSECTION.value: 1.5
    }
    c_in = []
    c_out = []
    s_out = []
    positions = []
    radii = []
    circle_collection_index_to_node_id_map = {}
    for in_, n in enumerate(graph.nodes):
        n_ = SimpleCenterlineNode(**(graph.nodes[n]))
        positions.append(tuple(n_.getVertexList()[:2]))
        radii.append(n_["r"])
        c_in.append(patch_n_color_map[n_["arterial_tree"].value])
        c_out.append(edge_n_color_map[n_["topology_class"].value])
        s_out.append(edgewidth_n_map[n_["topology_class"].value])
        circle_collection_index_to_node_id_map.update({in_: n})
    # - plot
    millimeters_to_inches = 0.0393701
    nodes_collection_mpl = CircleCollection(
        sizes=(numpy.pi/10*(numpy.array(radii)*millimeters_to_inches*FIG_DPI)**2),
        offsets=positions,
        offset_transform=ax.transData,
        edgecolors=c_out,
        facecolors=c_in,
        linewidths=s_out,
        antialiaseds=True,
        zorder=2,
        picker=True,
        pickradius=2,
    )
    ax.add_collection(nodes_collection_mpl)
    class NodeHoverEffects():
        def __init__(self):
            # Highlighted nodes
            self.node_hover_highlight_circle_obj = None
            # Textbox pointer with node information
            self.node_hover_annotation = None
        def node_hover(self, motion_notify_event: MouseEvent):
            redraw = False
            if self.node_hover_highlight_circle_obj is not None:
                self.node_hover_highlight_circle_obj.remove()
                self.node_hover_highlight_circle_obj = None
                redraw = True
            if self.node_hover_annotation is not None:
                self.node_hover_annotation.remove()
                self.node_hover_annotation = None
                redraw = True
            if motion_notify_event.inaxes == ax:
                cont, ind = nodes_collection_mpl.contains(motion_notify_event)
                if cont:
                    redraw = True
                    # HIGLIGHT NODE
                    cont_idx = ind["ind"][-1]
                    ax_lims_ = motion_notify_event.inaxes.get_xlim()
                    node_hover_circle_mpl = CirclePolygon(
                        xy=positions[cont_idx],
                        radius=1.1*(ax_lims_[1] - ax_lims_[0])/100,
                        resolution=16,
                        color='yellow',
                        linewidth=0,
                        zorder=2.1,
                        alpha=0.8
                    )
                    self.node_hover_highlight_circle_obj = motion_notify_event.inaxes.add_patch(node_hover_circle_mpl)
                    # NODE ANNOTATION
                    if ind["ind"].shape[0] > 1:
                        annotation_text = f"{ind['ind'].shape[0]} nodes\nZoom in and select\none node at a time"
                    else:
                        # get node information
                        # build info text
                        node_id = circle_collection_index_to_node_id_map[cont_idx]
                        node = SimpleCenterlineNode(**(graph.nodes[node_id]))
                        annotation_text  = f"Node \"{node_id}\"\n"
                        annotation_text += f"x [mm]: {node['x']: 7.3f}\ny [mm]: {node['y']: 7.3f}\nz [mm]: {node['z']: 7.3f}\n"
                        annotation_text += f"r [mm]: {node['r']: 7.3f}\nt  [s]: {node['t']: 7.3f}\n"
                        if node['arterial_tree'].value == ArteryPointTree.RIGHT.value:
                            annotation_text += f"Right arterial tree\n"
                        elif node['arterial_tree'].value == ArteryPointTree.LEFT.value:
                            annotation_text += f"Left arterial tree\n"
                        elif node['arterial_tree'].value == ArteryPointTree.RL.value:
                            annotation_text += f"Both arterial trees\n"
                        if node['topology_class'].value == ArteryPointTopologyClass.OSTIUM.value:
                            annotation_text += f"Coronary ostium\n"
                        elif node['topology_class'].value == ArteryPointTopologyClass.SEGMENT.value:
                            annotation_text += f"Arterial segment\n"
                        elif node['topology_class'].value == ArteryPointTopologyClass.INTERSECTION.value:
                            annotation_text += f"Arterial branching\n"
                        elif node['topology_class'].value == ArteryPointTopologyClass.ENDPOINT.value:
                            annotation_text += f"Branch endpoint\n"
                        # - distance from ostium/ostia
                        if not node['arterial_tree'].value == ArteryPointTree.RL.value:
                            for n in graph.nodes:
                                if graph.nodes[n]['arterial_tree'].value == node['arterial_tree'].value and graph.nodes[n]['topology_class'].value == ArteryPointTopologyClass.OSTIUM.value:
                                        distance = networkx.shortest_path_length(graph, source=n, target=node_id, weight="euclidean_distance")
                                        annotation_text += f"Distance from ostium [mm]:\n{distance: 8.3f}"
                                        break
                        else:
                            count_hits_ = 0
                            for n in graph.nodes:
                                if graph.nodes[n]['topology_class'].value == ArteryPointTopologyClass.OSTIUM.value:
                                    distance = networkx.shortest_path_length(graph, source=n, target=node_id, weight="euclidean_distance")
                                    ostium_left_right = "left" if graph.nodes[n]['arterial_tree'].value == ArteryPointTree.LEFT.value else "right"
                                    annotation_text += f"Distance from {ostium_left_right} ostium [mm]: {distance: 8.3f}"
                                    count_hits_ += 1
                                    if count_hits_ == 2:
                                        break
                    self.node_hover_annotation = motion_notify_event.inaxes.annotate(
                        # annotation position
                        xy=positions[cont_idx], xycoords='data',
                        # text
                        # https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text
                        text=annotation_text,
                        xytext=(10, 10), textcoords='axes points',
                        color='#343a40',
                        fontfamily=['monospace', 'arial', 'calibri', 'sans-serif'], fontsize=8.5, fontweight='light',
                        horizontalalignment='left', verticalalignment='bottom',
                        # bbox
                        # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyBboxPatch.html#matplotlib.patches.FancyBboxPatch
                        bbox=dict(
                            boxstyle='round',
                            facecolor='#ced4da',
                            edgecolor='#343a40',
                            linewidth=0.75
                        ),
                        # arrow and end patch
                        arrowprops=dict(
                            # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrowPatch.html#matplotlib.patches.FancyArrowPatch
                            arrowstyle="-",
                            color='#343a40',
                            patchB=node_hover_circle_mpl,
                            shrinkA=0, shrinkB=0
                        ),
                        zorder=2.2
                    )
            if redraw:
                motion_notify_event.canvas.draw_idle()
    node_hover_class = NodeHoverEffects()
    fig.canvas.mpl_connect("motion_notify_event", node_hover_class.node_hover)
    def node_on_pick(artist, mouse_event):
        print(mouse_event.ind)

    



    # plot undirected edges
    segs = []
    for u_,v_,a in graph.edges(data=True):
        uu = SimpleCenterlineNode(**(graph.nodes[u_])).getVertexList()
        vv = SimpleCenterlineNode(**(graph.nodes[v_])).getVertexList()
        segs.append(numpy.array([uu[:2],vv[:2]]))
    line_segments = LineCollection(segs, zorder=1, linewidth=0.4, color=COLOR_EDGE_DEFAULT)
    ax.add_collection(line_segments)
    # legend
    legend_elements = [
        Line2D([0], [0], marker='o', markerfacecolor=COLOR_NODE_IN_RCA, color="w",                 markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor=COLOR_NODE_IN_LCA, color="w",                 markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=COLOR_NODE_OUT_START,    markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=COLOR_NODE_OUT_CROSS, markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=COLOR_NODE_OUT_END,  markersize=10, lw=0)
    ]
    ax.legend(
        legend_elements,
        ["RCA",
         "LCA",
         "OSTIA",
         "INTERSECTIONS",
         "ENDPOINTS"],
         loc="upper right"
    )
    # axis
    ax.grid(visible=True, zorder=-10, color='gray', linestyle='dashed')
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.set_axisbelow(True)
    ax.set_title(graph.graph["image_id"])
    ax.autoscale_view()
    # out
    plt.tight_layout()
    plt.show()

def drawCenterlinesGraph3D(graph):
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
        c_in.append(COLOR_NODE_IN_RCA if n_["arterial_tree"].value == ArteryPointTree.RIGHT.value else COLOR_NODE_IN_LCA)
        if n_["topology_class"].value == ArteryPointTopologyClass.OSTIUM.value:
            c_out.append(COLOR_NODE_OUT_START)
            s_out.append(2.5)
        elif n_["topology_class"].value == ArteryPointTopologyClass.ENDPOINT.value:
            c_out.append(COLOR_NODE_OUT_END)
            s_out.append(2.5)
        elif n_["topology_class"].value == ArteryPointTopologyClass.INTERSECTION.value:
            c_out.append(COLOR_NODE_OUT_CROSS)
            s_out.append(2)
        else:
            c_out.append(COLOR_NODE_OUT_DEFAULT)
            s_out.append(0.0)
        positions.append(n_.getVertexList())
    # - convert to numpy
    c_in  = numpy.array(c_in)
    c_out = numpy.array(c_out)
    s_out = numpy.array(s_out)
    positions = numpy.array(positions)
    # - plot
    below_idx_ = [i for i in range(len(c_out)) if c_out[i] == COLOR_NODE_OUT_DEFAULT]
    above_idx_ = [i for i in range(len(c_out)) if c_out[i] != COLOR_NODE_OUT_DEFAULT]
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
    line_segments = Line3DCollection(segs, zorder=1, linewidth=0.4, color=COLOR_EDGE_DEFAULT)
    ax.add_collection(line_segments)
    # legend
    legend_elements = [
        Line2D([0], [0], marker='o', markerfacecolor=COLOR_NODE_IN_RCA, color="w",                 markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor=COLOR_NODE_IN_LCA, color="w",                 markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=COLOR_NODE_OUT_START,    markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=COLOR_NODE_OUT_CROSS, markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=COLOR_NODE_OUT_END,  markersize=10, lw=0)
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
    f_prova = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\CAT08\\CenterlineGraphs_FromReference\\dataset00.GML"
    from ..graph import loadGraph
    g_ = loadGraph(f_prova)
    drawCenterlinesGraph2D(graph=g_)
    #drawCenterlinesGraph3D(graph=g_)