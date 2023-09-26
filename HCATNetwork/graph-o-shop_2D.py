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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.backend_bases import Event
from matplotlib.backend_bases import MouseButton
# imports - gui

# imports - my modules
from .node import SimpleCenterlineNode, ArteryPointTopologyClass, ArteryPointTree
from .edge import BasicEdge
from .draw.styles import (COLOR_NODE_IN_LCA, COLOR_NODE_IN_RCA, COLOR_NODE_OUT_DEFAULT, 
                          COLOR_NODE_OUT_END, COLOR_NODE_OUT_CROSS, COLOR_NODE_OUT_START, 
                          COLOR_EDGE_DEFAULT)



if __name__ == "__main__":
    # place points on the canvas when the mouse left-clicks
    x = numpy.random.rand(10)
    y = numpy.random.rand(10)
    fig, ax = plt.subplots()
    def on_mouse_click_event(event: Event):
        if event.button is MouseButton.LEFT:
            if not event.inaxes:
                return
            x, y = event.xdata, event.ydata
            event.inaxes.scatter(x, y, c="red")
            event.canvas.draw_idle()
    fig.canvas.mpl_connect('button_press_event', on_mouse_click_event)
    ax.scatter(x, y)
    plt.show()

    # connect two points on the canvas when clicked sequentially

    x = numpy.random.rand(10)
    y = numpy.random.rand(10)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    plt.show()

