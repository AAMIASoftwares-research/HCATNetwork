
""" Unified colors and style-sheet for the heart network drawing
Dependencies
------------
matplotlib
palettable (https://jiffyclub.github.io/palettable/)
"""
import matplotlib


MILLIMETERS_TO_INCHES = 0.0393701

####################
# FIGURE PROPRIETIES
####################
FIGURE_DPI = 120
FIGURE_FACECOLOR = "#eff5f8"

AXES_FACECOLOR = "#f7fcfc"

AXES_ASPECT_TYPE = "equal"
AXES_ASPECT_ADJUSTABLE = "datalim"
AXES_ASPECT_ANCHOR = "C"

AXES_GRID_COLOR = "#ced4da"
AXES_GRID_ALPHA = 0.8
AXES_GRID_LINESTYLE = "--"
AXES_GRID_LINEWIDTH = 1.0

AXES_TEXT_COLOR = "#343a40"
AXES_TEXT_SIZE = 8.0
AXES_TEXT_SIZE_SMALL = 6.0
AXES_TEXT_FONTFAMILY = "monospace"


####################
# NODES AND EDGES
####################

# Node colors - inside
COLOR_NODE_FACE_DEFAULT = "grey"
COLOR_NODE_FACE_RCA     = "firebrick"
COLOR_NODE_FACE_LCA     = "navy"
COLOR_NODE_FACE_BOTH    = "purple"

# Node colors - outside
COLOR_NODE_EDGE_DEFAULT = "grey"
COLOR_NODE_EDGE_START   = "#2ed111"
COLOR_NODE_EDGE_CROSS   = "gold"
COLOR_NODE_EDGE_END     = "red"

# Edge colors
COLOR_EDGE_DEFAULT = "#22403d"
EDGE_LINEWIDTH = 0.7

EDGE_COLORMAP_DISTANCE = matplotlib.colormaps["turbo_r"]
EDGE_COLORMAP_RADIUS = matplotlib.colormaps["copper_r"]

##########################
# INTERACTIVE HIGHLIGHTING
##########################

COLOR_HIGHLIGHT_NODE = "#faed00"
ALPHA_HIGHLIGHT_NODE = 0.6

####################
# ARROWS/POINTERS
####################

COLOR_INFO_ARROW = "#343a40"
INFO_ARROW_LINESTYLE = "-"
INFO_ARROW_CAPSTYLE = "round"

####################
# TEXTBOXES
####################

COLOR_INFO_TEXT = "#343a40"
INFO_TEXT_FONTFAMILY = "monospace"
INFO_TEXT_FONTWEIGHT = "light"
INFO_TEXT_FONTSIZE = 8.0

COLOR_INFO_BOX_FACE = "#ced4da"
COLOR_INFO_BOX_EDGE = "#343a40"
INFO_BBOX_EDGE_WIDTH = 0.75

