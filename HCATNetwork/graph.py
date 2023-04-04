import numpy
import networkx
import node, edge

def saveGraph(
        graph: networkx.classes.graph.Graph|
                     networkx.classes.digraph.DiGraph|
                     networkx.classes.multigraph.MultiGraph|
                     networkx.classes.multidigraph.MultiDiGraph,
        file_path: str):
    networkx.write_gml(graph, file_path)

def loadGraph(file_path: str) ->    networkx.classes.graph.Graph|\
                                    networkx.classes.digraph.DiGraph|\
                                    networkx.classes.multigraph.MultiGraph|\
                                    networkx.classes.multidigraph.MultiDiGraph:
    return networkx.read_gml(file_path)

    


if __name__ == "__main__":
    print("Running 'HCATNetwork.graph' module")



    