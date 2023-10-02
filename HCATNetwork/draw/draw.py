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

from ..graph import BasicCenterlineGraph
from ..node import SimpleCenterlineNode, ArteryPointTopologyClass, ArteryPointTree
from .styles import *



def drawCenterlinesGraph2D(graph: networkx.Graph):
    """Assumes this kind on dictionaries:
        nodes: HCATNetwork.node.SimpleCenterlineNode
        edges: HCATNetwork.edge.BasicEdge
        graph: HCATNetwork.graph.BasicCenterlineGraph
    """
    fig, ax = plt.subplots(
        num="Centerlines Graph 2D Viewer | HCATNetwork",
        dpi=FIGURE_DPI
    )
    fig.set_facecolor(FIGURE_FACECOLOR)
    ax.set_facecolor(AXES_FACECOLOR)
    ax.set_aspect(
        aspect=AXES_ASPECT_TYPE,
        adjustable=AXES_ASPECT_ADJUSTABLE,
        anchor=AXES_ASPECT_ANCHOR
    )
    ax.grid(
        visible=True, 
        zorder=-100, 
        color=AXES_GRID_COLOR, 
        linestyle=AXES_GRID_LINESTYLE, 
        alpha=AXES_GRID_ALPHA, 
        linewidth=AXES_GRID_LINEWIDTH
    )
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.set_axisbelow(True)
    ax.set_title(graph.graph["image_id"])
    ########
    # NODES
    ########
    patch_n_color_map = {
        ArteryPointTree.RIGHT.value: COLOR_NODE_FACE_RCA,
        ArteryPointTree.LEFT.value: COLOR_NODE_FACE_LCA,
        ArteryPointTree.RL.value: COLOR_NODE_FACE_BOTH
    }
    edge_n_color_map = {
        ArteryPointTopologyClass.OSTIUM.value: COLOR_NODE_EDGE_START,
        ArteryPointTopologyClass.SEGMENT.value: COLOR_NODE_EDGE_DEFAULT,
        ArteryPointTopologyClass.ENDPOINT.value: COLOR_NODE_EDGE_END,
        ArteryPointTopologyClass.INTERSECTION.value: COLOR_NODE_EDGE_CROSS
    }
    edgewidth_n_map = {
        ArteryPointTopologyClass.OSTIUM.value: 2.2,
        ArteryPointTopologyClass.SEGMENT.value: 0.0,
        ArteryPointTopologyClass.ENDPOINT.value: 1,
        ArteryPointTopologyClass.INTERSECTION.value: 1.8
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
    circle_sizes = (numpy.pi/10*(numpy.array(radii)*MILLIMETERS_TO_INCHES*FIGURE_DPI)**2)
    circle_sizes = (circle_sizes - numpy.min(circle_sizes))/(numpy.max(circle_sizes) - numpy.min(circle_sizes))*40 + 10
    nodes_collection_mpl = CircleCollection(
        sizes=circle_sizes,
        offsets=positions,
        offset_transform=ax.transData,
        edgecolors=c_out,
        facecolors=c_in,
        linewidths=s_out,
        antialiaseds=True,
        zorder=2,
        picker=True,
        pickradius=1
    )
    nodes_mpl_obj = ax.add_collection(nodes_collection_mpl)
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
                        radius=1.05*(ax_lims_[1] - ax_lims_[0])/100,
                        resolution=16,
                        color=COLOR_HIGHLIGHT_NODE,
                        linewidth=0,
                        zorder=2.1,
                        alpha=ALPHA_HIGHLIGHT_NODE
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
                        annotation_text += f"x: {node['x']: 7.3f} mm\ny: {node['y']: 7.3f} mm\nz: {node['z']: 7.3f} mm\n"
                        annotation_text += f"r: {node['r']: 7.3f} mm\nt: {node['t']: 7.3f} s\n"
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
                        ostia = BasicCenterlineGraph.getCoronaryOstiumNodeIdRelativeToNode(graph=graph, node_id=node_id)
                        if len(ostia) == 1:
                            distance = networkx.shortest_path_length(graph, source=ostia[0], target=node_id, weight="euclidean_distance")
                            annotation_text += f"Distance from ostium:\n{distance: 8.3f} mm"
                        elif len(ostia) == 2:
                            # ostia[0] is alwais left, ostia[1] is alwais right
                            distance = networkx.shortest_path_length(graph, source=ostia[0], target=node_id, weight="euclidean_distance")
                            annotation_text += f"Distance from left ostium:\n{distance: 8.3f} mm"
                            distance = networkx.shortest_path_length(graph, source=ostia[1], target=node_id, weight="euclidean_distance")
                            annotation_text += f"Distance from right ostium:\n{distance: 8.3f} mm"
                    self.node_hover_annotation = motion_notify_event.inaxes.annotate(
                        # annotation position
                        xy=positions[cont_idx], xycoords='data',
                        # text
                        # https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text
                        text=annotation_text,
                        xytext=(10, 10), textcoords='axes points',
                        color=COLOR_INFO_TEXT,
                        fontfamily=['monospace'], fontsize=8.5, fontweight='light',
                        horizontalalignment='left', verticalalignment='bottom',
                        # bbox
                        # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyBboxPatch.html#matplotlib.patches.FancyBboxPatch
                        bbox=dict(
                            boxstyle='round',
                            facecolor=COLOR_INFO_BOX_FACE,
                            edgecolor=COLOR_INFO_BOX_EDGE,
                            linewidth=0.75
                        ),
                        # arrow and end patch
                        arrowprops=dict(
                            # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrowPatch.html#matplotlib.patches.FancyArrowPatch
                            arrowstyle="-",
                            color=COLOR_INFO_ARROW,
                            patchB=node_hover_circle_mpl,
                            shrinkA=0, shrinkB=0
                        ),
                        zorder=2.2
                    )
            if redraw:
                motion_notify_event.canvas.draw_idle()
    node_hover_class = NodeHoverEffects()
    fig.canvas.mpl_connect("motion_notify_event", node_hover_class.node_hover)
    # Keep for later reference
    # def node_on_pick(artist, mouse_event):
    #    print(mouse_event.ind)

    ########
    # EDGES
    ########
    line_edge_map = {}
    edge_distance_color_map = []
    segs = []
    for i, (u_,v_,a) in enumerate(graph.edges(data=True)):
        uu = SimpleCenterlineNode(**(graph.nodes[u_])).getVertexList()
        vv = SimpleCenterlineNode(**(graph.nodes[v_])).getVertexList()
        segs.append(numpy.array([uu[:2],vv[:2]]))
        line_edge_map.update({i: (u_,v_, a)}) # a is the edge attribute dictionary
        # find distance of node u_ from ostium. If node u_ is associated with two ostia, then keep the shortest distance
        ostia = BasicCenterlineGraph.getCoronaryOstiumNodeIdRelativeToNode(graph=graph, node_id=u_)
        if len(ostia) == 1:
            edge_distance_color_map.append(
                networkx.shortest_path_length(
                    graph, 
                    source=ostia[0], 
                    target=u_,
                    weight="euclidean_distance")
                )
        else:
            edge_distance_color_map.append(
                min([networkx.shortest_path_length(
                    graph, 
                    source=ostium, 
                    target=u_,
                    weight="euclidean_distance") for ostium in ostia])
                )
    edge_distance_color_map = numpy.array(edge_distance_color_map)
    line_segments = LineCollection(segs, zorder=1, linewidth=0.8, color=COLOR_EDGE_DEFAULT)
    lines_mpl_obj = ax.add_collection(line_segments)
    class EdgeEffects():
        def __init__(self, nodes_artist, node_collection, edges_artist):
            self.nodes_obj = nodes_artist
            self.nodes_collection = node_collection
            self.line_segments_obj = edges_artist
        def toggle_edges(self, event):
            """On a key pressed event, if the pressed key is 'n', then toggle the nodes visibility in the image.
               Only the edges are left, which will assume a different color profile depending on distance from the coronary ostium.
            """
            if event.key == "n":
                if self.nodes_obj.get_visible():
                    self.line_segments_obj.set_cmap("viridis")
                    self.line_segments_obj.set_array(edge_distance_color_map)
                    self.line_segments_obj.set_clim(vmin=numpy.min(edge_distance_color_map), vmax=numpy.max(edge_distance_color_map))
                    #self.line_segments_obj.set
                    self.nodes_obj.set_visible(False)
                else:
                    self.line_segments_obj.set_color(COLOR_EDGE_DEFAULT)
                    self.nodes_obj.set_visible(True)
                event.canvas.draw_idle()
    edge_effect = EdgeEffects(nodes_mpl_obj,nodes_collection_mpl,lines_mpl_obj)
    fig.canvas.mpl_connect('key_press_event', edge_effect.toggle_edges)
    ########
    # legend
    ########
    legend_elements = [
        Line2D([0], [0], marker='o', markerfacecolor=COLOR_NODE_FACE_RCA, color="w",                 markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor=COLOR_NODE_FACE_LCA, color="w",                 markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=COLOR_NODE_EDGE_START,    markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=COLOR_NODE_EDGE_CROSS, markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=COLOR_NODE_EDGE_END,  markersize=10, lw=0)
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
    # axis rescale to fit data
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
        c_in.append(COLOR_NODE_FACE_RCA if n_["arterial_tree"].value == ArteryPointTree.RIGHT.value else COLOR_NODE_FACE_LCA)
        if n_["topology_class"].value == ArteryPointTopologyClass.OSTIUM.value:
            c_out.append(COLOR_NODE_EDGE_START)
            s_out.append(2.5)
        elif n_["topology_class"].value == ArteryPointTopologyClass.ENDPOINT.value:
            c_out.append(COLOR_NODE_EDGE_END)
            s_out.append(2.5)
        elif n_["topology_class"].value == ArteryPointTopologyClass.INTERSECTION.value:
            c_out.append(COLOR_NODE_EDGE_CROSS)
            s_out.append(2)
        else:
            c_out.append(COLOR_NODE_EDGE_DEFAULT)
            s_out.append(0.0)
        positions.append(n_.getVertexList())
    # - convert to numpy
    c_in  = numpy.array(c_in)
    c_out = numpy.array(c_out)
    s_out = numpy.array(s_out)
    positions = numpy.array(positions)
    # - plot
    below_idx_ = [i for i in range(len(c_out)) if c_out[i] == COLOR_NODE_EDGE_DEFAULT]
    above_idx_ = [i for i in range(len(c_out)) if c_out[i] != COLOR_NODE_EDGE_DEFAULT]
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
        Line2D([0], [0], marker='o', markerfacecolor=COLOR_NODE_FACE_RCA, color="w",                 markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor=COLOR_NODE_FACE_LCA, color="w",                 markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=COLOR_NODE_EDGE_START,    markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=COLOR_NODE_EDGE_CROSS, markersize=10, lw=0),
        Line2D([0], [0], marker='o', markerfacecolor="w",          color=COLOR_NODE_EDGE_END,  markersize=10, lw=0)
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