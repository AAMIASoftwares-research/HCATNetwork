# not tried since 2023.04.05, prob wont work, useless anyway
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
    print(type(g))

    # adjacency matrix
    E = nx.adjacency_matrix(g)
    print(E)

    # prova di grafo full
    t = numpy.linspace(0,1,100)
    x = numpy.cos(2*t)
    y = numpy.sin(0.213*t) + 0.05*t
    z = t**2
    v = numpy.array([x, y, z]).T
    # make graph
    import node
    import edge
    G = nx.MultiDiGraph()
    for i, vertex in enumerate(v):
        node_dict = node.getNodeDictFromKeyList( node.VertexNode_KeysList )
        node.set_vertexNodeVertex(node_dict, vertex)
        node_dict["class"] = "first" if i == 0 else "last" if i == v.shape[0]-1 else ""
        assert node.assertNodeValidity(node_dict)
        G.add_node(i, **node_dict)
        if i != 0:
            edge_dict = edge.getEdgeDictFromKeyList(edge.SimpleCenterlineEdge_KeysList)
            d = numpy.linalg.norm(v[i-1] - v[i])
            # forward
            edge_dict["weight"] = d
            edge_dict["distance"] = d
            assert edge.assertEdgeValidity(edge_dict)
            G.add_edge(i-1, i, **edge_dict)
            # backward
            edge_dict = edge.getEdgeDictFromKeyList(edge.SimpleCenterlineEdge_KeysList)
            edge_dict["weight"] = -d
            edge_dict["distance"] = -d
            assert edge.assertEdgeValidity(edge_dict)
            G.add_edge(i, i-1, **edge_dict)

    pos={i: v[i,:2].tolist() for i in range(v.shape[0])}
    mapping = {n: l for n, l in zip(G.nodes, G.nodes.data('class', default=""))}
    nx.relabel_nodes(G, mapping)
    nx.draw(G, arrowsize=5, with_labels=True, pos=pos, node_size=50, )
    plt.show()
    # save graph in a standard format
    import graph
    path = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\HCATNetwork\\HCATNetwork\\test\\prova_grafo.GML"
    graph.save_graph(G, path)
    print("####\n####\n####", G.nodes.data(), "\n", type(G), "####\n####\n####")
    Gnew = graph.load_graph(path)
    print(Gnew.nodes.data(), "\n", type(Gnew), "####\n####\n####")



