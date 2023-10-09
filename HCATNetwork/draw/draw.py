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
import matplotlib
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

class BasicCenterlineGraphInteractiveDrawer():
    """Draws a BasicCenterlineGraph interactively in 2D using Matplotlib.

    Colors and styles are defined in HCATNetwork.draw.styles.

    Artists layers by zorder:
    * 1.0: edges
    * 2.0: nodes
    * 2.1: nodes highlight
    * 3.0: nodes info textbox (lower left)
    * 4.0: menu textbox (lower right)
           sub-menus will appear on top of the menu with zorders in (4.0;5.0)
    * 5.0: legend (upper right)

    Usage
    -----
    The menu on screen shows some options.
    * n: toggle sone data views. Availabel views are:
        * default view (edges and nodes),
        * edges colored by distance from ostium,
        * edges colored by radius of source node.
    * p: toggle projections. Available projections are:
        * XY (default),
        * XZ,
        * YZ.

    Other basic options are:
    * Double left click on a node to select it and display its info in the textbox.
    * When a node or multiple nodes are selected, use the mouse whell to scroll through the tree.

    See Also
    --------
    HCATNetwork.graph.BasicCenterlineGraph
    HCATNetwork.node.SimpleCenterlineNode
    HCATNetwork.edge.BasicEdge
    HCATNetwork.draw.styles
    """
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    graph: networkx.Graph
    def __init__(self, figure: matplotlib.figure.Figure, axes: matplotlib.axes.Axes, graph: networkx.Graph):
        """Given a NetworkX graph holding a BasicCenterlineGraph, draw it interactively in 2D on the ax Axes contained in the Figure.
        """
        self.fig: matplotlib.figure.Figure = figure
        self.ax: matplotlib.axes.Axes = axes
        self.graph: networkx.Graph = graph
        #######
        # Nodes
        #######
        self.nodes_positions = self.getNodesPositions()
        self.nodes_r = self.getNodesRadii()
        self.nodes_distance_from_ostia = self.getNodesDistanceFromOstia()
        self.nodes_artist_collectionIndex_to_nodeId_map = numpy.array([n for n in self.graph.nodes])
        self.nodes_artist: matplotlib.collections.CircleCollection = self.getNodesArtist()
        self.nodes_artist.set_visible(True)
        self.ax.add_collection(self.nodes_artist)
        #######
        # Edges
        #######
        # Edges - basic
        self.edges_positions = self.getEdgesPositions()
        self.__current_projection_edges_positions = self.edges_positions[:,:,[0,1]] # XY plane
        self.edges_artist_collectionIndex_to_edgeIdAndData_map = numpy.array([(u_,v_,a) for (u_,v_,a) in self.graph.edges(data=True)])
        self.edges_artist_monocrome: matplotlib.collections.LineCollection = self.getEdgesArtistMonocrome()
        self.edges_artist_monocrome.set_visible(True)
        self.ax.add_collection(self.edges_artist_monocrome)
        # Edges - colormapped by distance from ostia of the source node
        self.edges_artist_colormapped_distance: matplotlib.collections.LineCollection = self.getEdgesArtistColormappedDistance()
        self.edges_artist_colormapped_distance.set_visible(False)
        self.ax.add_collection(self.edges_artist_colormapped_distance)
        # Edges - colormapped by radius of the source node
        self.edges_artist_colormapped_radius: matplotlib.collections.LineCollection = self.getEdgesArtistColormappedRadius()
        self.edges_artist_colormapped_radius.set_visible(False)
        self.ax.add_collection(self.edges_artist_colormapped_radius)
        #########
        # Legends
        #########
        # Nodes legend
        self.legend_artist = self.getLegendArtist()
        self.ax.add_artist(self.legend_artist)
        self.legend_artist.set_visible(True)
        # Edges colormap legend: distance from ostia
        self.colorbar_edges_colormapped_distance: matplotlib.axes.Axes = self.getEdgesDistanceColorbar()
        self.colorbar_edges_colormapped_distance.set_visible(False)
        # Edges colormap legend: radius of the connected node
        self.colorbar_edges_colormapped_radius: matplotlib.axes.Axes = self.getEdgesRadiusColorbar()
        self.colorbar_edges_colormapped_radius.set_visible(False)
        ##############
        # highlighting
        ##############
        # Node highlighting
        self.node_highlighted_artist: matplotlib.collections.CircleCollection = self.getNodeHighlightedArtist()
        self.node_highlighted_artist.set_visible(False)
        self.ax.add_collection(self.node_highlighted_artist)
        ############
        # Text Boxes
        ############
        # Info textbox
        self.textbox_info_artist: matplotlib.text.Annotation = self.getInfoTextboxArtist()
        self.ax.add_artist(self.textbox_info_artist)
        self.textbox_info_artist.set_visible(False)
        # Menu textbox
        self.textbox_artist_menu: matplotlib.offsetbox.AnchoredText = self.getMainMenuTextboxArtist()
        self.ax.add_artist(self.textbox_artist_menu)
        self.textbox_artist_menu.set_visible(True)
        #####################
        # Interactive effects
        #####################
        # Nodes and edges drawing styles carousel
        # Key pressed event with key "n" will cycle through the following styles:
        # 0: Colored nodes with mono-chromatic edges
        # 1: No nodes, distance from coronary ostia colormapped edges
        # 2: No nodes, radius colormapped edges
        self.viewstyle_carousel_current_index = 0
        self.viewstyle_carousel_artists_cycler = [
            [self.nodes_artist, self.edges_artist_monocrome],
            [self.edges_artist_colormapped_distance],
            [self.edges_artist_colormapped_radius]
        ]
        self.viewstyle_carousel_legends_cycler = [
            self.legend_artist,
            self.colorbar_edges_colormapped_distance,
            self.colorbar_edges_colormapped_radius
        ]
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press_event_viewstyle_carousel)
        # View plane projection carousel
        # Key pressed event with key "p" will cycle through the following styles:
        # 0: XY plane
        # 1: XZ plane
        # 2: YZ plane
        self.viewplane_carousel_current_index = 0
        self.viewplane_carousel_planes_list = ["XY", "XZ", "YZ"]
        self.viewplane_carousel_planes_list_slices = [[0,1], [0,2], [1,2]]
        self.viewplane_carousel_nodes_artists_list = [self.nodes_artist]
        self.viewplane_carousel_edges_artists_list = [
            self.edges_artist_monocrome,
            self.edges_artist_colormapped_distance,
            self.edges_artist_colormapped_radius
        ]
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press_event_viewplane_carousel)
        # Node highlighting and info textbox
        # Mouse click event on a node will highlight it and show its info in the textbox
        self.node_highlighted_id = None
        self.fig.canvas.mpl_connect("button_press_event", self.on_left_mouse_button_press_event_node_highlighting)
        # Scroll event
        # Mouse scroll event will move the node highlighting up and down
        # the arterial tree
        self.fig.canvas.mpl_connect("scroll_event", self.on_mouse_scroll_event_node_highlighting)


    
    def getNodesPositions(self) -> numpy.ndarray:
        """Returns a numpy.ndarray of shape (N,3) with the n-th node's coordinates on each row.
        """
        # Memorty allocation
        nodes_positions = numpy.zeros((self.graph.number_of_nodes(), 3), dtype="float")
        # Fill the array
        for i, n in enumerate(self.graph.nodes):
            nodes_positions[i,:] = self.graph.nodes[n]["x"], self.graph.nodes[n]["y"], self.graph.nodes[n]["z"]
        return nodes_positions
    
    def getNodesRadii(self) -> numpy.ndarray:
        """Returns a numpy.ndarray of shape (n_nodes, ) of nodes radii.
        """
        nodes_radii = numpy.zeros((self.graph.number_of_nodes(),), dtype="float")
        for i, n in enumerate(self.graph.nodes):
            nodes_radii[i] = self.graph.nodes[n]["r"]
        return nodes_radii
    
    def getNodesArtist(self) -> matplotlib.collections.CircleCollection:
        """Returns a matplotlib.collections.CircleCollection object with the nodes drawn as circles.
        The circle sizes are proportional to the nodes radii.
        The circle colors are mapped to the nodes arterial tree.
        The circle edge colors are mapped to the nodes topology class.
        By default, nodes are viewed projected to the XY plane.
        """
        nodes_facecolor_map_dict = {
            ArteryPointTree.RIGHT.value: COLOR_NODE_FACE_RCA,
            ArteryPointTree.LEFT.value: COLOR_NODE_FACE_LCA,
            ArteryPointTree.RL.value: COLOR_NODE_FACE_BOTH
        }
        nodes_edgecolor_map_dict = {
            ArteryPointTopologyClass.OSTIUM.value: COLOR_NODE_EDGE_START,
            ArteryPointTopologyClass.SEGMENT.value: COLOR_NODE_EDGE_DEFAULT,
            ArteryPointTopologyClass.ENDPOINT.value: COLOR_NODE_EDGE_END,
            ArteryPointTopologyClass.INTERSECTION.value: COLOR_NODE_EDGE_CROSS
        }
        nodes_edgewidth_map_dict = {
            ArteryPointTopologyClass.OSTIUM.value: 2.2,
            ArteryPointTopologyClass.SEGMENT.value: 0.0,
            ArteryPointTopologyClass.ENDPOINT.value: 1,
            ArteryPointTopologyClass.INTERSECTION.value: 1.8
        }
        c_in  = []
        c_out = []
        lw    = []
        for n in self.graph.nodes:
            c_in.append(nodes_facecolor_map_dict[self.graph.nodes[n]["arterial_tree"].value])
            c_out.append(nodes_edgecolor_map_dict[self.graph.nodes[n]["topology_class"].value])
            lw.append(nodes_edgewidth_map_dict[self.graph.nodes[n]["topology_class"].value])
        # circle sizes (as points/pixels in the patch area) go from 10 pt to 50 pt
        radii_as_dots = self.nodes_r*MILLIMETERS_TO_INCHES*FIGURE_DPI
        circle_sizes = numpy.pi*radii_as_dots**2
        circle_sizes = 10 + 40*(circle_sizes - numpy.min(circle_sizes))/(numpy.max(circle_sizes) - numpy.min(circle_sizes))
        nodes_collection_mpl = CircleCollection(
            sizes=circle_sizes,
            offsets=self.nodes_positions[:,[0,1]],
            offset_transform=self.ax.transData,
            edgecolors=c_out,
            facecolors=c_in,
            linewidths=lw,
            antialiaseds=True,
            zorder=2.0,
            picker=True,
            pickradius=1
        )
        return nodes_collection_mpl
    
    def getEdgesPositions(self) -> numpy.ndarray:
        """Returns a numpy.ndarray of (n_edges, 2, 3) with the n-th edge's coordinates on each row.
        The length is n_edges, one per each edge segment.
        The edge segment is defined as a numpy array of shape (2,3),
        where the first row is the source node positions [x, y, z],
        and the second row is the target node position."""
        # Memorty allocation
        edges_positions = numpy.zeros((self.graph.number_of_edges(), 2, 3), dtype="float")
        # Fill the array
        for i, (u_,v_) in enumerate(self.graph.edges(data=False)):
            edges_positions[i,0,:] = self.graph.nodes[u_]["x"], self.graph.nodes[u_]["y"], self.graph.nodes[u_]["z"]
            edges_positions[i,1,:] = self.graph.nodes[v_]["x"], self.graph.nodes[v_]["y"], self.graph.nodes[v_]["z"]
        return edges_positions

    def getEdgesArtistMonocrome(self) -> matplotlib.collections.LineCollection:
        """Returns a matplotlib.collections.LineCollection object with the edges drawn as mono-chromatic lines."""
        line_collection = matplotlib.collections.LineCollection(
            self.__current_projection_edges_positions,
            zorder=1.0,
            linewidth=0.7,
            color=COLOR_EDGE_DEFAULT
        )
        return line_collection
    
    def getNodesDistanceFromOstia(self) -> dict[str: float]:
        """Returns a dictionary with nodes id as keys, the distance of the node from the ostium as value.
        If two ostia are present, the distance is the minimum of the two.
        """
        node_dist_map = {}
        ostia = BasicCenterlineGraph.getCoronaryOstiaNodeId(graph=self.graph)
        for ostium in ostia:
            node_dist_map.update({ostium: 0.0})
            d_ = networkx.single_source_dijkstra_path_length(self.graph, source=ostium, weight="euclidean_distance")
            try:
                d_.pop(ostium)
            except KeyError:
                pass
            # If two ostia, get the one with lowest distance
            for k in d_.keys():
                if k in node_dist_map.keys():
                    if d_[k] < node_dist_map[k]:
                        node_dist_map[k] = d_[k]
                else:
                    node_dist_map.update({k: d_[k]})
        return node_dist_map
    
    def getEdgesDistanceArray(self) -> numpy.ndarray:
        """Returns a numpy array with the distance of each edge's first node from respective ostium.
        If two ostia are present for the first node, the distance is the minimum of the two.

        Returns
        -------
        numpy.ndarray
        """
        temp_node_dist_map_ = self.getNodesDistanceFromOstia()
        edge_distance_color_map = []
        for (u_,v_) in self.graph.edges(data=False):
            edge_distance_color_map.append(temp_node_dist_map_[u_])
        edge_distance_color_map = numpy.array(edge_distance_color_map)
        return edge_distance_color_map
    
    def getEdgesArtistColormappedDistance(self) -> matplotlib.collections.LineCollection:
        """Returns a matplotlib.collections.LineCollection object with the edges drawn as lines with color mapped to the distance from the ostia."""
        # Find distance of each edge's first node from respective ostium
        edge_distance_color_map = self.getEdgesDistanceArray()
        # Get matplotlib artist and return it
        line_collection = matplotlib.collections.LineCollection(
            self.__current_projection_edges_positions,
            zorder=1.0,
            linewidth=1.5, 
            colors=EDGE_COLORMAP_DISTANCE(edge_distance_color_map/edge_distance_color_map.max()),
            capstyle="round"
        )
        return line_collection

    def getEdgesDistanceColorbar(self) -> matplotlib.axes.Axes:
        """Returns a matplotlib.axes.Axes object (inset in self.ax) with the colorbar for the edges distance from ostia."""
        distances = self.getEdgesDistanceArray()
        ax_ = self.ax.inset_axes(
            bounds=[0.78, 0.915, 0.20, 0.05],
            xticks=[-0.5, 14.5],
            xticklabels=[],
            yticks=[],
            yticklabels=[],
        )
        ax_.set_title(
            "Distance from ostium",
            fontfamily=AXES_TEXT_FONTFAMILY,
            fontsize=AXES_TEXT_SIZE_SMALL*1.2,
            color=AXES_TEXT_COLOR,
            y=0.7
        )
        ax_.set_xlabel(
            f"0 -> {distances.max():.2f} mm",
            fontfamily=AXES_TEXT_FONTFAMILY,
            fontsize=AXES_TEXT_SIZE_SMALL,
            color=AXES_TEXT_COLOR,
            labelpad=-4
        )
        ax_.imshow(
            [numpy.linspace(0,1,15).tolist(), numpy.linspace(0,1,15).tolist()],
            cmap=EDGE_COLORMAP_DISTANCE,
            interpolation="bicubic"
        )
        return ax_

    def getEdgesArtistColormappedRadius(self) -> matplotlib.collections.LineCollection:
        """Returns a matplotlib.collections.LineCollection object with the edges drawn as lines with color mapped to the radius of the first node."""
        # Find distance of each edge's first node from respective ostium
        edge_node_radii_color_map = numpy.array(
            [self.graph.nodes[u_]["r"] for (u_,_) in self.graph.edges(data=False)]
        )
        # Get matplotlib artist and return it
        line_collection = matplotlib.collections.LineCollection(
            self.__current_projection_edges_positions,
            zorder=1.0,
            linewidths=0.5+2.5*edge_node_radii_color_map/edge_node_radii_color_map.max(), 
            colors=EDGE_COLORMAP_RADIUS(edge_node_radii_color_map/edge_node_radii_color_map.max()),
            capstyle="round"
        )
        return line_collection

    def getEdgesRadiusColorbar(self):
        """Returns a matplotlib.axes.Axes object (inset in self.ax) with the colorbar for the radius of the first node of each edge."""        
        edge_node_radii_color_map = numpy.array(
            [self.graph.nodes[u_]["r"] for (u_,_) in self.graph.edges(data=False)]
        )
        ax_ = self.ax.inset_axes(
            bounds=[0.78, 0.915, 0.20, 0.05],
            xticks=[-0.5, 14.5],
            xticklabels=[],
            yticks=[],
            yticklabels=[],
        )
        ax_.set_title(
            "Lumen radius",
            fontfamily=AXES_TEXT_FONTFAMILY,
            fontsize=AXES_TEXT_SIZE_SMALL*1.2,
            color=AXES_TEXT_COLOR,
            y=0.7
        )
        ax_.set_xlabel(
            f"0 -> {edge_node_radii_color_map.max():.2f} mm",
            fontfamily=AXES_TEXT_FONTFAMILY,
            fontsize=AXES_TEXT_SIZE_SMALL,
            color=AXES_TEXT_COLOR,
            labelpad=-4
        )
        ax_.imshow(
            [numpy.linspace(0,1,15).tolist(), numpy.linspace(0,1,15).tolist()],
            cmap=EDGE_COLORMAP_RADIUS,
            interpolation="bicubic"
        )
        return ax_

    def getLegendArtist(self) -> matplotlib.legend.Legend:
        legend_elements = [
            Line2D([0], [0], marker='o', markerfacecolor=COLOR_NODE_FACE_RCA, color=COLOR_INFO_BOX_FACE,   markersize=10, lw=0),
            Line2D([0], [0], marker='o', markerfacecolor=COLOR_NODE_FACE_LCA, color=COLOR_INFO_BOX_FACE,   markersize=10, lw=0),
            Line2D([0], [0], marker='o', markerfacecolor=COLOR_INFO_BOX_FACE, color=COLOR_NODE_EDGE_START, markersize=10, lw=0),
            Line2D([0], [0], marker='o', markerfacecolor=COLOR_INFO_BOX_FACE, color=COLOR_NODE_EDGE_CROSS, markersize=10, lw=0),
            Line2D([0], [0], marker='o', markerfacecolor=COLOR_INFO_BOX_FACE, color=COLOR_NODE_EDGE_END,   markersize=10, lw=0)
        ]

        legend_artist = matplotlib.legend.Legend(
            parent=self.ax,
            handles=legend_elements,
            labels=["RCA", "LCA", "OSTIA", "INTERSECTIONS", "ENDPOINTS"],
            loc="upper right",
            prop={
                # Legend text font properties
                "size": INFO_TEXT_FONTSIZE,
                "family": INFO_TEXT_FONTFAMILY,
                "weight": INFO_TEXT_FONTWEIGHT
            },
            labelcolor=COLOR_INFO_TEXT,
            title="Node Colors Legend",
            title_fontproperties={
                "size": 7.0,
                "family": INFO_TEXT_FONTFAMILY,
                "weight": "bold"
            },
            framealpha=0.95,
            facecolor=COLOR_INFO_BOX_FACE,
            edgecolor=COLOR_INFO_BOX_EDGE,
            borderpad=0.5
        )
        return legend_artist

    def getNodeHighlightedArtist(self) -> matplotlib.collections.CircleCollection:
        """Returns a matplotlib.collections.CircleCollection object with the highlighting effect artist.
        """
        offset_zero_ = numpy.mean(self.nodes_positions[:,[0,1]], axis=0).reshape((1,2))
        node_highlighted_artist = matplotlib.collections.CircleCollection(
            sizes=[55],
            offsets=offset_zero_,
            offset_transform=self.ax.transData,
            facecolors=COLOR_HIGHLIGHT_NODE,
            alpha=ALPHA_HIGHLIGHT_NODE,
            linewidths=INFO_BBOX_EDGE_WIDTH,
            edgecolor=COLOR_INFO_ARROW,
            zorder=3.0
        )
        return node_highlighted_artist

    def getInfoTextboxArtist(self) -> matplotlib.text.Annotation:
        """Returns a matplotlib.text.Annotation object with the information text.
        The textbox is not visible by default.
        The textbox is populated with the default debug string.
        To change the text of the textbox, use the set_text() method of the artist.
        """
        # To keep a textbox fixed at a location, one might also use
        # matplotlib.offsetbox.AnnotationBbox. However, since the info textbox is set in
        # the lower-left corner, there is no need of using that, and the straightforward Annotation
        # is more intuitive.
        default_arrow_x = numpy.mean(self.nodes_positions[:,0])
        default_arrow_y = numpy.mean(self.nodes_positions[:,1])
        node_hover_annotation = matplotlib.text.Annotation(
            # annotation position (arrow points to this position)
            xy=(default_arrow_x,default_arrow_y), xycoords='data',
            # text
            # https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text
            text="debug text",
            xytext=(7, 7), textcoords='axes points',
            color=COLOR_INFO_TEXT,
            fontfamily=INFO_TEXT_FONTFAMILY, fontsize=INFO_TEXT_FONTSIZE, fontweight=INFO_TEXT_FONTWEIGHT,
            horizontalalignment='left', verticalalignment='bottom',
            # bbox
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyBboxPatch.html#matplotlib.patches.FancyBboxPatch
            bbox=dict(
                boxstyle='round',
                facecolor=COLOR_INFO_BOX_FACE,
                edgecolor=COLOR_INFO_BOX_EDGE,
                linewidth=INFO_BBOX_EDGE_WIDTH,
                alpha=0.95
            ),
            # arrow and end patch
            arrowprops=dict(
                # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrowPatch.html#matplotlib.patches.FancyArrowPatch
                arrowstyle=INFO_ARROW_LINESTYLE,
                capstyle=INFO_ARROW_CAPSTYLE,
                color=COLOR_INFO_ARROW,
                patchB=None,
                shrinkA=0.0, shrinkB=0.0,
                zorder=3.0
            ),
            zorder=3.000001
        )
        return node_hover_annotation

    def getMainMenuDisplayText(self) -> str:
        """Returns a string with the main menu display text."""
        MAIN_MENU_KEY_OPTIONS = ["n", "p"]
        MAIN_MENU_KEY_TEXT = ["toggle views.", "toggle projections."]
        # build the menu
        text = "MAIN MENU"
        for k, ktext in zip(MAIN_MENU_KEY_OPTIONS, MAIN_MENU_KEY_TEXT):
            text += f"\n{k}:  {ktext}"
        return text
    
    def getMainMenuTextboxArtist(self) -> matplotlib.offsetbox.AnchoredText:
        """Returns a matplotlib.offsetbox.AnchoredText artist anchored to the lower right corner of
        the axes with the menu information.
        The textbox is always visible on top of everything (zorder 5.0).
        The textbox is populated with the main menu at first.
        Then, if the selected option has a submenu, the submenu gets printed instead of the main menu
        via the set_text() method of the artist.
        """
        main_menu_text = self.getMainMenuDisplayText()
        menu_textbox = matplotlib.offsetbox.AnchoredText(
            # https://matplotlib.org/stable/api/offsetbox_api.html#matplotlib.offsetbox.AnchoredText
            # https://matplotlib.org/stable/api/offsetbox_api.html#matplotlib.offsetbox.AnchoredOffsetbox
            s=main_menu_text,
            loc="lower right",
            frameon=True,
            bbox_to_anchor=(1.0, 0.0),
            bbox_transform=self.ax.transAxes,
            borderpad=1.0,
            # Properties to be passed to the matplotlib.text.Text artist instance used to draw the text.
            prop={
                "family": INFO_TEXT_FONTFAMILY,
                "size": INFO_TEXT_FONTSIZE*0.9,
                "weight": INFO_TEXT_FONTWEIGHT,
                "color": COLOR_INFO_TEXT
            },
            zorder=5.0
        )
        # This line is necessary to have the menu saved when exporting the figure (apparently, not tested it)
        menu_textbox.set_clip_on(True)
        # matplotlib.offsetbox.AnchoredText.patch is a matplotlib.patches.FancyBboxPatch instance
        menu_textbox.patch.set(
            boxstyle="round",
            facecolor=COLOR_INFO_BOX_FACE,
            edgecolor=COLOR_INFO_BOX_EDGE,
            linewidth=INFO_BBOX_EDGE_WIDTH,
            alpha=0.95
        )
        return menu_textbox

    # Interactive effects

    def on_key_press_event_viewstyle_carousel(self, event: matplotlib.backend_bases.KeyEvent):
        if event.key == "n":
            # Set all elements' and legends artists to invisible
            for artist_list, legend in zip(self.viewstyle_carousel_artists_cycler, self.viewstyle_carousel_legends_cycler):
                for artist in artist_list:
                    artist.set_visible(False)
                legend.set_visible(False)
            # Now set visible all legend and artists of the next style
            self.viewstyle_carousel_current_index = (self.viewstyle_carousel_current_index + 1) % len(self.viewstyle_carousel_artists_cycler)
            for artist in self.viewstyle_carousel_artists_cycler[self.viewstyle_carousel_current_index]:
                artist.set_visible(True)
            self.viewstyle_carousel_legends_cycler[self.viewstyle_carousel_current_index].set_visible(True)
            # Draw changes
            self.fig.canvas.draw_idle()
    
    def on_key_press_event_viewplane_carousel(self, event: matplotlib.backend_bases.KeyEvent):
        if event.key == "p":
            # Data
            self.viewplane_carousel_current_index = (self.viewplane_carousel_current_index + 1) % len(self.viewplane_carousel_planes_list)
            current_slice = self.viewplane_carousel_planes_list_slices[self.viewplane_carousel_current_index]
            for node_artist in self.viewplane_carousel_nodes_artists_list:
                node_artist.set_offsets(self.nodes_positions[:,current_slice])
            self.__current_projection_edges_positions = self.edges_positions[:,:,current_slice]
            for edge_artist in self.viewplane_carousel_edges_artists_list:
                edge_artist.set_segments(self.__current_projection_edges_positions)
            # Axis labels
            self.ax.set_xlabel(
                self.viewplane_carousel_planes_list[self.viewplane_carousel_current_index][0] + " [mm]"
            )
            self.ax.set_ylabel(
                self.viewplane_carousel_planes_list[self.viewplane_carousel_current_index][1] + " [mm]"
            )
            # reset axis limits
            self.ax.set_xlim(
                numpy.min(self.nodes_positions[:,self.viewplane_carousel_planes_list_slices[self.viewplane_carousel_current_index][0]])-10,
                numpy.max(self.nodes_positions[:,self.viewplane_carousel_planes_list_slices[self.viewplane_carousel_current_index][0]])+10
            )
            self.ax.set_ylim(
                numpy.min(self.nodes_positions[:,self.viewplane_carousel_planes_list_slices[self.viewplane_carousel_current_index][1]])-10,
                numpy.max(self.nodes_positions[:,self.viewplane_carousel_planes_list_slices[self.viewplane_carousel_current_index][1]])+10
            )
            # Change highlight position accordingly
            if self.node_highlighted_id is not None:
                node_pos_ = numpy.array(
                    [self.graph.nodes[self.node_highlighted_id]['x'],self.graph.nodes[self.node_highlighted_id]['y'],self.graph.nodes[self.node_highlighted_id]['z']] 
                )[current_slice]
                self.node_highlighted_artist.set_offsets(node_pos_.reshape((1,2)))
                self.textbox_info_artist.xy = node_pos_
            # Draw changes
            self.fig.canvas.draw_idle()

    def __get_node_info_textbox_text(self, node_id):
        node = self.graph.nodes[node_id]
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
        ostia = BasicCenterlineGraph.getCoronaryOstiumNodeIdRelativeToNode(graph=self.graph, node_id=node_id)
        if len(ostia) == 1:
            distance = networkx.shortest_path_length(self.graph, source=ostia[0], target=node_id, weight="euclidean_distance")
            annotation_text += f"Distance from ostium:\n{distance: 8.3f} mm"
        elif len(ostia) == 2:
            # ostia[0] is alwais left, ostia[1] is alwais right
            distance = networkx.shortest_path_length(self.graph, source=ostia[0], target=node_id, weight="euclidean_distance")
            annotation_text += f"Distance from left ostium:\n{distance: 8.3f} mm"
            distance = networkx.shortest_path_length(self.graph, source=ostia[1], target=node_id, weight="euclidean_distance")
            annotation_text += f"Distance from right ostium:\n{distance: 8.3f} mm"
        return annotation_text

    def __update_highlight_and_info_textbox_artists(self, textbox_text):
        """This utility function is to be called after the property self.node_highlighted_id is updated.
        This sfunction changes the position of the highlight artist and the textbox artist accordingly.
        Drawing (draw_idle()) is left for the caller.
        """
        if textbox_text is None:
            textbox_text = self.__get_node_info_textbox_text(self.node_highlighted_id)
        # Get slice indexes of current axes vies (XY, XZ or YZ)
        current_slice = self.viewplane_carousel_planes_list_slices[self.viewplane_carousel_current_index]
        # Get node coords
        node_coords = numpy.zeros((1,3), dtype="float")
        node_coords[0,:] = self.graph.nodes[self.node_highlighted_id]["x"], self.graph.nodes[self.node_highlighted_id]["y"], self.graph.nodes[self.node_highlighted_id]["z"]
        node_coords = node_coords[0,current_slice]
        # Set highlight artist props
        self.node_highlighted_artist.set_offsets(node_coords)
        self.node_highlighted_artist.set_visible(True)
        # Set textbox props
        self.textbox_info_artist.xy = (node_coords[0], node_coords[1])
        self.textbox_info_artist.set_text(textbox_text)
        self.textbox_info_artist.arrow_patch.set(
            patchB=self.node_highlighted_artist
        )
        self.textbox_info_artist.shrinkB=0.0
        self.textbox_info_artist.set_visible(True)

    def on_left_mouse_button_press_event_node_highlighting(self, event: matplotlib.backend_bases.MouseEvent):
        """Node highlighting and textbox display on a double left mouse click.
        """
        # If not double-click, do not consider it
        if not event.dblclick:
            return
        # If it is not a left double-click and the click is not on the axes,
        # just set the artists to not visible.
        if event.button != matplotlib.backend_bases.MouseButton.LEFT or event.inaxes != self.ax:
            self.node_highlighted_id = None
            self.textbox_info_artist.set_visible(False)
            self.node_highlighted_artist.set_visible(False)
            event.canvas.draw_idle()
            return
        # If it is a left click, check if the click is on a node
        # and highlight it and display its info in the textbox.
        # If the click is not on a node, just set the artists to not visible.
        cont, ind = self.nodes_artist.contains(event)
        if not cont:
            self.node_highlighted_id = None
            self.textbox_info_artist.set_visible(False)
            self.node_highlighted_artist.set_visible(False)
            event.canvas.draw_idle()
            return
        # When defining and drawing studff, first set the position and data of the highlighing patch,
        # then the propoerties of the textbox, then the patchB property of the annotation,
        # and finally the visibility of the artists.
        # Get highlighed node position
        self.node_highlighted_id = self.nodes_artist_collectionIndex_to_nodeId_map[ind["ind"][-1]]
        if len(ind["ind"]) != 1:
            # If multiple nodes are selected, ask to zoom in and select just one node.
            text_ = f"{len(ind['ind'])} nodes selected.\nPlease zoom in and select just one node."
        else:
            # If the click is on a node and just one node is selected, display its info in the textbox.
            text_ = self.__get_node_info_textbox_text(self.node_highlighted_id)
        self.__update_highlight_and_info_textbox_artists(text_)
        event.canvas.draw_idle()
        return

    def on_mouse_scroll_event_node_highlighting(self, event: matplotlib.backend_bases.MouseEvent):
        """When the mouse scrolls, if a node is highlighted, change node with the one up or down the tree
        and update the textbox and highlight.
        """
        if self.node_highlighted_id is None:
            return
        # Get the node's neighbors
        neighbors_plus_self = list(self.graph.neighbors(self.node_highlighted_id)) + [self.node_highlighted_id]
        # Get each neighbour's distance from ostium
        neighbors_distances = [self.nodes_distance_from_ostia[n] for n in neighbors_plus_self]
        if event.button == "up":
            # Get the neighbor with the minimum distance from ostium
            new_node_id = neighbors_plus_self[numpy.argmin(neighbors_distances)]
        elif event.button == "down":
            # Get the neighbor with the maximum distance from ostium
            new_node_id = neighbors_plus_self[numpy.argmax(neighbors_distances)]
        else:
            return
        # Update the highlight position
        old_node_id = self.node_highlighted_id
        self.node_highlighted_id = new_node_id
        if self.node_highlighted_id != old_node_id:
            # Update
            self.__update_highlight_and_info_textbox_artists(None)
            # Draw changes
            event.canvas.draw_idle()
        
        
def drawCenterlinesGraph2D(graph: networkx.Graph):
    """Draws the Coronary Artery tree centerlines in an interactive way.
    
    Parameters
    ----------
    graph : networkx.Graph
        The graph to draw. Assumes this kind of dictionaries:
            nodes: HCATNetwork.node.SimpleCenterlineNode
            edges: HCATNetwork.edge.BasicEdge
            graph: HCATNetwork.graph.BasicCenterlineGraph
    """
    # Figure
    fig, ax = plt.subplots(
        num="Centerlines Graph 2D Viewer | HCATNetwork",
        dpi=FIGURE_DPI
    )
    fig.set_facecolor(FIGURE_FACECOLOR)
    # Axes
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
    # Content
    c = BasicCenterlineGraphInteractiveDrawer(fig, ax, graph)
    # Rescale axes view to content
    ax.autoscale_view()
    # Plot
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