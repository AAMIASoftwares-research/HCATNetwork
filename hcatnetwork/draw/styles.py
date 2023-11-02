
""" Unified colors and style-sheet for the heart network drawing
Dependencies
------------
matplotlib
palettable (https://jiffyclub.github.io/palettable/)
"""
import matplotlib
import PyQt6.QtGui

def get_primary_screen_dpi():
    """Get the DPI of the primary screen"""
    app = PyQt6.QtGui.QGuiApplication([])
    screen = app.primaryScreen()
    dpi = screen.physicalDotsPerInch()
    return dpi

MILLIMETERS_TO_INCHES = 0.0393701
INCHES_TO_MILLIMETERS = 1.0 / MILLIMETERS_TO_INCHES

####################
# FIGURE PROPRIETIES
####################

COMPUTER_MONITOR_DPI = int(round(get_primary_screen_dpi()))
FIGURE_DPI = COMPUTER_MONITOR_DPI
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
NODE_FACECOLOR_DEFAULT = "grey"
NODE_FACECOLOR_RCA     = "firebrick"
NODE_FACECOLOR_LCA     = "navy"
NODE_FACECOLOR_LR    = "purple"

# Node colors - outside
NODE_EDGEECOLOR_DEFAULT = "grey"
NODE_EDGEECOLOR_START   = "#2ed111"
NODE_EDGEECOLOR_CROSS   = "gold"
NODE_EDGEECOLOR_END     = "red"

# Edge colors
EDGE_FACECOLOR_DEFAULT = "#22403d"
EDGE_LINEWIDTH = 0.7

EDGE_COLORMAP_DISTANCE = matplotlib.colormaps["turbo_r"]
EDGE_COLORMAP_RADIUS = matplotlib.colormaps["copper_r"]    ###### MAKE CUSTOM COLORMAPS, AND RENAME ALL CONSTANTS

##########################
# INTERACTIVE HIGHLIGHTING
##########################

NODE_HIGHLIGHT_FACECOLOR = "#faed00"
NODE_HIGHLIGHT_ALPHA = 0.7

####################
# ARROWS/POINTERS
####################

INFO_ARROW_FACECOLOR = "#343a40"
INFO_ARROW_LINESTYLE = "-"
INFO_ARROW_CAPSTYLE = "round"

####################
# TEXTBOXES
####################

INFO_TEXT_COLOR = "#343a40"
INFO_TEXT_FONTFAMILY = "monospace"
INFO_TEXT_FONTWEIGHT = "light"
INFO_TEXT_FONTSIZE = 8.0

INFO_BOX_FACECOLOR = "#ced4da"
INFO_BOX_EDGECOLOR = "#343a40"
INFO_BBOX_EDGE_WIDTH = 0.75

