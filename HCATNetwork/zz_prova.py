import numpy
import networkx as nx
import matplotlib.pyplot as plt

def prova():
    a = numpy.random.randn(300,3)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection="3d"))
    ax.scatter(a[:,0], a[:,1], a[:,2], c=a[:,2], cmap="rainbow")
    plt.show()

# unit testing
if __name__ == "__main__":
    prova()

    # Prove con Grafi
    print("Prova grafi")
    g = nx.Graph()
    g = nx.DiGraph()
    g = nx.MultiGraph()
    g = nx.MultiDiGraph() # <-- I thing this is the one

    g = nx.Graph()
    g.add_edge(0, 2)
    g.add_edge(2, 3, weight=0.9, distance=0.2, feature=[1.2, 3.4])
    g.add_edge(3, 0, weight=1.2, tensor=numpy.eye(3))
    g = nx.relabel_nodes(g, {0: "Primo", 3:"Ostium"})

    nx.draw(g, with_labels=True)
    plt.show()
    print(g.nodes)

    # adjacency matrix
    E = nx.adjacency_matrix(g)
    print(E)

