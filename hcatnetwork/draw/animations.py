"""Animations useful for many stuff.
Prototype file, which maybe will become its own module.
"""

import numpy
import networkx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ..node import ArteryNodeTopology
from ..graph import SimpleCenterlineGraph
from ..graph import load_graph


##########################
# Fluid movement animation
##########################

def get_particle_velocity_from_graph(graph: networkx.graph.Graph, shortest_path_lengths_from_ostia: dict, nodes_position_array: numpy.ndarray, particle_position: numpy.ndarray, particle_last_closest_node_id: str | None = None) -> numpy.ndarray:
    """Get the velocity of a particle given a SimpleCenterlineGraph.
    nodes_position_array is a numpy array of shape (n_nodes, 3) containing the x y and z coordinates of each node.
    """
    if not (particle_last_closest_node_id is None):
        # this offers a massive speedup:
        # Look for the closest node to the particle just in the neighbours of the last closest node
        neighbours_list_ = [particle_last_closest_node_id]
        for i in range(10):
            new_n_ = []
            for n_ in neighbours_list_:
                new_n_.extend(list(graph.neighbors(n_)))
            new_n_ = list(set(new_n_))
            neighbours_list_.extend(new_n_)
            neighbours_list_ = list(set(neighbours_list_))
        nodes_position_array = numpy.array(
            [[graph.nodes[n]["x"], graph.nodes[n]["y"], graph.nodes[n]["z"]] for n in neighbours_list_]
        )
    # Find the closest node
    dists = numpy.linalg.norm(nodes_position_array - particle_position, axis=1)
    closest_node_i = dists.argmin()
    closest_node_id = list(graph.nodes)[closest_node_i] if particle_last_closest_node_id is None else neighbours_list_[closest_node_i]   
    # Deal with endpoints
    if graph.nodes[closest_node_id]["topology"].value ==  ArteryNodeTopology.ENDPOINT.value:
        return (numpy.zeros((3,)), closest_node_id, True) 
    # Find the node pointing to the distal direction
    neighbors = list(graph.neighbors(closest_node_id))
    dists_ = [shortest_path_lengths_from_ostia[n] for n in neighbors]
    neighbors_distal_idxs = [i for i in range(len(neighbors)) if dists_[i] > shortest_path_lengths_from_ostia[closest_node_id]]
    if len(neighbors_distal_idxs) == 0:
        # This situation should already be dealt with by the
        # previous condition on the endpoints
        pass
    elif len(neighbors_distal_idxs) == 1:
        nodes_pointing_to_distal = neighbors[neighbors_distal_idxs[0]]
    elif len(neighbors_distal_idxs) > 1:
        neigh_ = [neighbors[i] for i in neighbors_distal_idxs]
        neigh_pos_ = numpy.array([[graph.nodes[n]["x"], graph.nodes[n]["y"], graph.nodes[n]["z"]] for n in neigh_])
        neigh_closest_to_particle_ = neigh_[numpy.linalg.norm(neigh_pos_ - particle_position, axis=1).argmin()]
        nodes_pointing_to_distal = neigh_closest_to_particle_
    p_closest_node = numpy.array(
        [graph.nodes[closest_node_id]["x"], graph.nodes[closest_node_id]["y"], graph.nodes[closest_node_id]["z"]]
    )
    p_distal_node = numpy.array(
        [graph.nodes[nodes_pointing_to_distal]["x"], graph.nodes[nodes_pointing_to_distal]["y"], graph.nodes[nodes_pointing_to_distal]["z"]]
    )
    # Velocity parallel to the vessel
    v_vector = p_distal_node - p_closest_node
    v_vector /= numpy.linalg.norm(v_vector) # 1 mm/s
    # fluid velocity profile in a vessel + particle re-entry in vessel
    v_vessel_profile = 1 - (1/(graph.nodes[closest_node_id]["r"])**2) * numpy.linalg.norm(particle_position - p_closest_node) ** 2
    v_vector *= v_vessel_profile if v_vessel_profile > 0 else 0.0
    # re enter the vessel
    v_vector += (p_distal_node - p_closest_node) / (2*numpy.linalg.norm(p_distal_node - p_closest_node))
    v_vector /= numpy.linalg.norm(v_vector)
    # scaling factors - conservation of mass kinda
    v_vector *= 2.5 / graph.nodes[closest_node_id]["r"]
    return (v_vector, closest_node_id, False)




FRAME_RATE = 30
FRAME_RATE_DT = 1 / FRAME_RATE
ANIM_DURATION_S = 30
N_PARTICLES = 1000
F_PROVA = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\CAT08\\CenterlineGraphs_FromReference\\dataset00.GML"

if __name__ == "__main__":

    
    from matplotlib.collections import CircleCollection

    
    graph = load_graph(F_PROVA)
    graph = SimpleCenterlineGraph.resample_coronary_artery_tree(graph, 0.5)
    ostia = SimpleCenterlineGraph.get_coronary_ostia_node_id(graph)
    shortest_path_lengths_from_ostia = networkx.shortest_path_length(graph, ostia[0])
    shortest_path_lengths_from_ostia.update(networkx.shortest_path_length(graph, ostia[1]))
    shortest_path_lengths_from_ostia.update({k: 0.0 for k in ostia})
    node_position_array = numpy.array(
        [[graph.nodes[n]["x"], graph.nodes[n]["y"], graph.nodes[n]["z"]] for n in graph.nodes]
    )

    fig, ax = plt.subplots()

    start_positions = []
    last_closest_node_to_each_particle_list = []
    list_of_nodes = list(graph.nodes)
    for i in range(N_PARTICLES):
        n_ = numpy.random.choice(list_of_nodes)
        start_positions.append([graph.nodes[n_]["x"], graph.nodes[n_]["y"], graph.nodes[n_]["z"]])
        last_closest_node_to_each_particle_list.append(n_)
    start_positions = numpy.array(start_positions)
    start_positions += numpy.random.uniform(-1.5, 1.5, size=(N_PARTICLES, 3))

    ax.scatter(
        node_position_array[:,0], 
        node_position_array[:,1], 
        s=[(fig.get_dpi() * (2*graph.nodes[n]["r"]/ 25.4))**2 for n in graph.nodes],
        facecolors="none",
        edgecolors="black",
        zorder=0.0,
        alpha=0.05
    )

    particles_artist = CircleCollection(
        sizes=[3]*N_PARTICLES,
        offsets=start_positions[:,:2],
        offset_transform=ax.transData,
        #facecolors=numpy.array([[0.1]*N_PARTICLES]),
        #cmap="turbo_r",
        facecolors="red",
        linewidths=0,
        zorder=1.0
    )
    ax.add_collection(particles_artist)

    global start_positions_g
    start_positions_g = start_positions

    def update_fluid_animation(frame_n):
        print(f"Frame {frame_n} of {FRAME_RATE*ANIM_DURATION_S}")
        global start_positions_g
        frame_t = frame_n / FRAME_RATE
        #ax.set_title(f"Time: {frame_t:.1f} s of {ANIM_DURATION_S:d} s")
        #current_velocities = numpy.zeros((N_PARTICLES,))
        for i, p_pos in enumerate(start_positions_g):
            v, closest_node_id, is_at_endpoint = get_particle_velocity_from_graph(graph, shortest_path_lengths_from_ostia, node_position_array, p_pos, last_closest_node_to_each_particle_list[i])
            if is_at_endpoint:
                # Move back to random ostium
                new_ostium_ = numpy.random.choice(ostia)
                start_positions_g[i][0] = graph.nodes[new_ostium_]["x"]
                start_positions_g[i][1] = graph.nodes[new_ostium_]["y"]
                start_positions_g[i][2] = graph.nodes[new_ostium_]["z"]
                start_positions_g[i] += numpy.random.uniform(-2.0, 2.0, size=(3,))
                last_closest_node_to_each_particle_list[i] = new_ostium_
            else:
                # Move to next position
                last_closest_node_to_each_particle_list[i] = closest_node_id
                start_positions_g[i] += 20 * v * FRAME_RATE_DT
                #current_velocities[i] = numpy.linalg.norm(v)
        particles_artist.set_offsets(start_positions_g[:,:2])
        #particles_artist.set_array(current_velocities)
        return (particles_artist,)
    
    anim = FuncAnimation(fig, update_fluid_animation, frames=FRAME_RATE*ANIM_DURATION_S, interval=1000/FRAME_RATE, blit=True)
    ax.autoscale_view()
    ax.axis(False)
    ax.grid(False)
    print("Animating...")
    if 0:
        plt.show()
    else:
        anim.save("fluid_animation.mp4", fps=FRAME_RATE)

