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
    

    # Load CAT08 centerlines into graph and visualise it
    import os, matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    
    CAT08_IM00_folder = os.path.normpath("C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\CAT08\\dataset00\\")
    v_list = []
    for i_ in range(4):
        v_file_path = os.path.join(CAT08_IM00_folder, f"vessel{i_}", "reference.txt")
        v_list.append(numpy.loadtxt(v_file_path, delimiter=" ", usecols=range(4)))
        
    if 1:
        ax1 = plt.subplot(121)
        for i_ in range(4):
            ax1.plot(range(v_list[i_].shape[0]), v_list[i_][:,0]+v_list[i_][:,1]+v_list[i_][:,2], label=str(i_))
        ax1.set_xlabel("idx")
        ax1.set_ylabel("x+y+z [mm]")
        ax1.legend()
        ax2 = plt.subplot(122)
        for i_ in range(4):
            ax2.plot(v_list[i_][:,0],v_list[i_][:,1], label=str(i_))
        ax2.set_xlabel("x [mm]")
        ax2.set_ylabel("y [mm]")
        ax2.legend()
        plt.show()
    if 0:
        d = numpy.linalg.norm(v_list[-1][1:,:3]-v_list[-1][:-1,:3], axis = 1)
        plt.scatter(numpy.zeros(len(d)), d)
        plt.scatter(range(d.shape[0]),d)
        plt.show()

    def getCentelrineLength(reference_centerline: numpy.ndarray):
        return numpy.sum(numpy.linalg.norm(reference_centerline[1:,:3]-reference_centerline[:-1,:3], axis = 1))

    # must define a functions that tells me the 3d coordinate + radius along any s belonging to R>=0
    def getContinuousCenterline(reference_centerline: numpy.ndarray, s:float):
        """Linear interpolation"""
        total_length = numpy.sum(
            numpy.linalg.norm(reference_centerline[1:,:3]-reference_centerline[:-1,:3], axis = 1)
        )
        if s == 0:
            return reference_centerline[0]
        if s == total_length:
            return reference_centerline[-1]
        if s > total_length or s < 0:
            return None
            #raise RuntimeWarning(f"Input parameter s={s}mm exceeds the admissed range of [0;{total_length}]mm")
        idx_before = 0
        len_before = 0.0
        last_len_before = 0.0
        while len_before < s:
            last_len_before = numpy.linalg.norm(reference_centerline[idx_before+1,:3]-reference_centerline[idx_before,:3])
            len_before += last_len_before
            idx_before += 1
        # Reached the point for which the s is between before and after
        exceeded_len = s - (len_before-last_len_before)
        covered_percentage = exceeded_len/last_len_before if last_len_before != 0 else 0.0
        out_point =  covered_percentage*reference_centerline[idx_before+1,:] + (1-covered_percentage)*reference_centerline[idx_before,:]
        return out_point
    
    if 0:
        t = numpy.linspace(0,30,1000)
        s = t#(1.5+numpy.cos(t))*t
        plt.plot(t,s)
        plt.show()
        y = numpy.zeros((len(s),4))
        for i_ in range(len(s)):
            y[i_] = getContinuousCenterline(v_list[-1], s[i_])
        plt.plot(y[:,0], y[:,1])
        plt.scatter(y[:,0], y[:,1])
        plt.show()

    ## preprocess the centerlines: necessary because some of them have the ostium very
    # different from one another in the common segments
    # 1 - find the two arterial trees right away: we use just the first points of the centerlines
    ostia = numpy.array([l[0,:3] for l in v_list])
    min_distance_between_two_trees = 15 #mm
    db = DBSCAN(eps=min_distance_between_two_trees, min_samples=1).fit(ostia)
    tree1_list = [v_list[i] for i in numpy.argwhere(db.labels_==0).flatten()]
    tree1_ostia = ostia[numpy.argwhere(db.labels_==0).flatten(),:]
    tree2_list = [v_list[i] for i in numpy.argwhere(db.labels_==1).flatten()]
    tree2_ostia = ostia[numpy.argwhere(db.labels_==1).flatten(),:]
    # 2 - find the mean distance between all the ostia
    mean_ostia = numpy.mean(ostia, axis=0)
    # 3 - for each tree, find the centerline of which the first point is closer to the mean_ostia, then add the first point to every centerline
    # tree 1
    if tree1_ostia.shape[0] > 1:
        dist_ = numpy.linalg.norm(tree1_ostia - mean_ostia, axis=1)
        idx = numpy.argmin(dist_)
        p = tree1_list[idx][0]
        for i in range(tree1_ostia.shape[0]):
            if i != idx:
                tree1_list[i] = numpy.insert(tree2_list[i], 0, p, axis=0)
    # tree 2
    if tree2_ostia.shape[0] > 1:
        dist_ = numpy.linalg.norm(tree2_ostia - mean_ostia, axis=1)
        idx = numpy.argmin(dist_)
        p = tree2_list[idx][0]
        for i in range(tree2_ostia.shape[0]):
            if i != idx:
                tree2_list[i] = numpy.insert(tree2_list[i], 0, p, axis=0)
    # 4 - add everything back to v_list
    v_list = []
    for t in tree1_list:
        v_list.append(t)
    for t in tree2_list:
        v_list.append(t)
    
    if 1:
        ax1 = plt.subplot(121)
        for i_ in range(4):
            ax1.plot(range(v_list[i_].shape[0]), v_list[i_][:,0]+v_list[i_][:,1]+v_list[i_][:,2], label=str(i_))
        ax1.set_xlabel("idx")
        ax1.set_ylabel("x+y+z [mm]")
        ax1.legend()
        ax2 = plt.subplot(122)
        for i_ in range(4):
            ax2.plot(v_list[i_][:,0],v_list[i_][:,1], label=str(i_))
        ax2.set_xlabel("x [mm]")
        ax2.set_ylabel("y [mm]")
        ax2.legend()
        plt.show()

    # Now find the almost-mean-shift
    spacing=0.29
    max_length = numpy.max([getCentelrineLength(c) for c in v_list])
    tuple_list = []
    r_thresh_modifier_count = 0
    len_dbscan_set_previous = 0 
    d_ostium_array = numpy.linspace(0,max_length, int(max_length/spacing))
    i_ = 0
    while i_ < d_ostium_array.shape[0]:
        d_ostium = d_ostium_array[i_]
        p_d = [getContinuousCenterline(c, d_ostium) for c in v_list]
        p_d_pruned = [x for x in p_d if x is not None]
        r = numpy.mean([p[3] for p in p_d_pruned])
        if r_thresh_modifier_count == 0:
            r_thresh = 0.75*numpy.exp(-0.8*d_ostium/10) + 0.25
        else:
            r_thresh = 0.12
            r_thresh_modifier_count -= 1
        db = DBSCAN(
            eps=r_thresh*r,
            min_samples=1
        ).fit(numpy.array(p_d_pruned)[:,:3])
        if i_ > 0 and r_thresh_modifier_count==0:
            if len(set(db.labels_)) > len_dbscan_set_previous:
                # This means that, since the last step, something branched off
                # To make sure that the point in which it branches off is very near to the previous segment
                # (no hard turns), we go back 3 < n < 5 steps and temporarily set the 
                # segments-branching threshold (eps in DBSCAN) to a much lower value, so that
                # the interested segments can branch off before they would do with the previous threshold.
                n_backwards = 4
                i_ = max(0, i_ - n_backwards - 1)
                r_thresh_modifier_count = n_backwards + 1
                for ii in range(
                    min(len(tuple_list),n_backwards+1)
                    ):
                    tuple_list.pop()
                len_dbscan_set_previous = len(set(db.labels_))
                continue
        if r_thresh_modifier_count == 0:
            len_dbscan_set_previous = len(set(db.labels_))
        p_out_list = []
        for index in set(db.labels_):
            idx_positions = numpy.argwhere(db.labels_ == index).flatten()                
            if len(idx_positions) < 1:
                raise RuntimeError("Should always be at least 1")
            elif len(idx_positions) == 1:
                p_out_list.append(p_d_pruned[idx_positions[0]])
            else:
                p_d_pruned_out = numpy.array(
                    [p_d_pruned[i] for i in idx_positions]
                )
                p_new = numpy.mean(
                    p_d_pruned_out,
                    axis=0
                )
                p_out_list.append(p_new)
        tuple_list.append((d_ostium, p_out_list))
        # Update iterator
        i_ += 1
        print(f"{100*i_/len(d_ostium_array):3.1f}", end="\r")
    print("")

    vl, cl = [], []
    for t in tuple_list:
        for v in t[1]:
            vl.append([v[0], v[1], v[2], v[3]])
            cl.append(t[0])
    vl = numpy.array(vl)
    cl = numpy.array(cl).flatten()
    # plots
    if 0:
        for i_ in range(4):
            plt.plot(v_list[i_][:,0],v_list[i_][:,1])
        plt.scatter(vl[:,0], vl[:,1], c=cl)
        plt.show()

    ## now, we do a final dbscan so we can separate the two arterial trees (L and R)
    # Note that this step is possible under the assumption that the arterial trees are disconnected
    # This happens in 95% of humans, but not in all of them
    # This assumption i smore than ok for CAT08 and any other dataset annotations probably,
    # but for an inferred arterial tree this should not be taken for granted.
    min_distance_between_two_trees = 2.5 #mm
    db = DBSCAN(eps=min_distance_between_two_trees, min_samples=10).fit(vl[:,:3])
    tree1 = vl[numpy.argwhere(db.labels_==0).flatten(),:]
    d_ostium1 = cl[numpy.argwhere(db.labels_==0).flatten()]
    tree2 = vl[numpy.argwhere(db.labels_==1).flatten(),:]
    d_ostium2 = cl[numpy.argwhere(db.labels_==1).flatten()]

    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    ax.scatter(tree1[:,0], tree1[:,1], tree1[:,2], s=10*tree1[:,3], c=d_ostium1, cmap="plasma")
    ax.scatter(tree2[:,0], tree2[:,1], tree2[:,2], s=10*tree2[:,3], c=d_ostium2, cmap="rainbow")
    plt.show()

    # now, it only remains to create the final graph




   
        






