import numpy
import networkx

### Simple 3D + R graph ###
from node import NodeVertex3Radius
class CenterlineWithRadiusGraph(object):
    name: str
    tree: str
    g: networkx.classes.digraph.DiGraph

    def __init__(self):
        self.g = networkx.DiGraph()




if __name__ == "__main__":
    print("Running 'HCATNetwork.graph' module")
    