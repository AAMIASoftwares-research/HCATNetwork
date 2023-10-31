"""Prototype:
2D graph-o-shop for HCATNetwork wants to be a tool for visualizing and editing 2D centerline graphs
on top of the heart images.
https://matplotlib.org/stable/users/explain/figure/event_handling.html
https://matplotlib.org/stable/gallery/event_handling/poly_editor.html

To run as a module, activate the venv, go inside the HCATNetwork parent directory,
and use: python -m HCATNetwork.graph-o-shop_2D
"""

# imports - system
import os, sys, time
import numpy, networkx
# imports - plotting
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
# imports - gui
import PyQt6.QtGui
# imports - my modules
from .node import SimpleCenterlineNode, ArteryPointTopologyClass, ArteryPointTree
from .edge import SimpleCenterlineEdge
from .draw.styles import (NODE_FACECOLOR_LCA, NODE_FACECOLOR_RCA, NODE_EDGEECOLOR_DEFAULT, 
                          NODE_EDGEECOLOR_END, NODE_EDGEECOLOR_CROSS, NODE_EDGEECOLOR_START, 
                          EDGE_FACECOLOR_DEFAULT)



if __name__ == "__main__":
    # mm
    pass