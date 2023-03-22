import numpy
import matplotlib.pyplot as plt

def prova():
    a = numpy.random.randn(300,3)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection="3d"))
    ax.scatter(a[:,0], a[:,1], a[:,2], c=a[:,2], cmap="rainbow")
    plt.show()

# unit testing
if __name__ == "__main__":
    prova()