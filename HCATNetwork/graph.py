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
    CAT08_IM00_folder = os.path.normpath("C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\CAT08\\dataset00\\")
    v_list = []
    for i_ in range(4):
        v_file_path = os.path.join(CAT08_IM00_folder, f"vessel{i_}", "reference.txt")
        v_list.append(numpy.loadtxt(v_file_path, delimiter=" ", usecols=range(4)))
        plt.plot(range(v_list[-1].shape[0]), v_list[-1][:,0]+v_list[-1][:,1]+v_list[-1][:,2], label=str(i_))
    plt.legend()
    plt.show()

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
    
    # CAT08 ha sempre 1 RCA, e 3 LCA -> sfruttiamo questo fatto
    from sklearn.cluster import DBSCAN
    spacing=0.25
    max_length = numpy.max([getCentelrineLength(c) for c in v_list])
    tuple_list = []
    for d_ostium in numpy.linspace(0,max_length, int(max_length/spacing)):
        p_d = [getContinuousCenterline(c, d_ostium) for c in v_list]
        p_d_pruned = [x for x in p_d if x is not None]
        if len(p_d) > len(p_d_pruned):
            pass#break
        r = numpy.mean([p[3] for p in p_d_pruned])
        db = DBSCAN(eps=0.75*r, min_samples=1).fit(
            numpy.array(p_d_pruned)[:,:3]
        )
        if 0:
            # Fino a qui compreso funziona! Ora devi solo creare il grafo
            tuple_list.append((d_ostium, p_d, p_d_pruned, db.labels_))
        # Number of clusters in labels, ignoring noise if present.
        """
        n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        n_noise_ = list(db.labels_).count(-1)
        print(p_d, p_d_pruned,"\n", db.labels_, "\n", n_clusters_, n_noise_)
        quit()
        """
        p_out_list = []
        for index in set(db.labels_):
            idx_positions = numpy.argwhere(db.labels_ == index).flatten()                
            if len(idx_positions) < 1:
                raise RuntimeError("Should always be at least 1")
            elif len(idx_positions) == 1:
                p_out_list.append(p_d_pruned[idx_positions[0]])
            else:
                p_d_pruned_to_median = numpy.array(
                    [p_d_pruned[i] for i in idx_positions]
                )
                p_median = numpy.median(
                    p_d_pruned_to_median,
                    axis=0
                )
                p_out_list.append(p_median)
        tuple_list.append((d_ostium, p_out_list))
    
    for i_ in range(4):
        plt.plot(v_list[i_][:,0],v_list[i_][:,1])
    
    vl = []
    cl = []
    for t in tuple_list:
        for v in t[1]:
            vl.append([v[0], v[1], v[2]])
            cl.append(t[0])
    vl = numpy.array(vl)
    plt.scatter(vl[:,0], vl[:,1], vl[:,2], c=cl)
    plt.show()



   
        






