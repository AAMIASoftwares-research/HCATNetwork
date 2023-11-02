"""Here are gathered some utilities to interface numpy and HCATNetwork objects with 3D Slicer.
https://www.slicer.org/

"""

import os, json
import numpy, networkx

#################
# Basic utilities
#################

def numpy_array_to_open_curve_json(arr: numpy.ndarray, labels: list[str] | None = None, descriptions: list[str] | None = None) -> str:
    """This function takes a numpy array and returns a string containing a json-formatted
    string that can be used to open an Open Curve object in 3D Slicer.

    This function just takes the numpy array and creates a list of control points,
    it does not transform the points into the image coordinates system.
    That has to be done before calling this function.
    
    If the intention is to save it to a file, the file should have the following extension:
    .SlicerOpenCurve.mkr.json

    Parameters
    ----------
    arr : numpy.ndarray
        The numpy array containing the points to be converted into an Open Curve.
        Must be a 2D array with shape (N, 3), where N is the number of points, and columns are x, y, z.
        Can also be a 2D array with shape (N, 4), where the columns are x, y, z, r.
    labels : list[str] | None, optional
        The list of labels to be assigned to each control point. If None, the labels will be
        the index of the point in the array, by default None.
    descriptions : list[str] | None, optional
        The list of descriptions to be assigned to each control point. If None, the descriptions will be
        empty strings, by default None.

    Returns
    -------
    str
        The json-formatted string that can be used to open an Open Curve object in 3D Slicer.
    """
    # Check the input array
    if len(arr.shape) != 2:
        raise RuntimeError(f"Unsupported input array shape: {arr.shape}. Must be 2D.")
    if arr.shape[1] not in [3, 4]:
        raise RuntimeError(f"Unsupported input array shape: {arr.shape}. Must have 3 or 4 columns x, y, z (,r).")
    # Create the control points list
    control_points = []
    for i_, p in enumerate(arr):
        point = [p[0], p[1], p[2]]
        cp_ = {
            "id": str(i_),
            "label": str(labels[i_]) if not labels is None else str(i_),
            "description": str(descriptions[i_]) if not descriptions is None else "",
            "associatedNodeID": "",
            "position": point,
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], # Identity 3x3 matrix. Note: this does not change anything in the final visualisation
            "selected": False,
            "locked": True,
            "visibility": True,
            "positionStatus": "defined"
        }
        control_points.append(cp_)
    measurements = [
        {
            "name": "length",
            "enabled": False,
            "units": "mm",
            "printFormat": "%-#4.4g%s"
        },
        {
            "name": "curvature mean",
            "enabled": False,
            "printFormat": "%5.3f %s"
        },
        {
            "name": "curvature max",
            "enabled": False,
            "printFormat": "%5.3f %s"
        }
    ]
    markups = [
        {
            "type": "Curve",
            "coordinateSystem": "RAS", # Right, Anterior, Superior
            "coordinateUnits": "mm",
            "locked": True,
            "fixedNumberOfControlPoints": True,
            "labelFormat": "%N-%d",
            "lastUsedControlPointNumber": i_,
            "controlPoints": control_points, 
            "measurements": measurements
        }
    ]
    slicer_dict = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": markups
    }
    # Make the json string
    json_string = json.dumps(slicer_dict, indent=4)
    return json_string

def numpy_array_to_fiducials_json(arr: numpy.ndarray, labels: list[str] | None = None, descriptions: list[str] | None = None) -> str:
    """This function takes a numpy array and returns a string containing a json-formatted
    string that can be used to open a Points List, or List of Fiducials (markers) object in 3D Slicer.

    This function just takes the numpy array and creates a list of control points,
    it does not transform the points into the image coordinates system.
    That has to be done before calling this function.

    If the intention is to save it to a file, the file should have the following extension:
    .SlicerFiducial.mkr.json

    Parameters
    ----------
    arr : numpy.ndarray
        The numpy array containing the points to be converted into an Open Curve.
        Must be a 2D array with shape (N, 3), where N is the number of points, and columns are x, y, z.
        Can also be a 2D array with shape (N, 4), where the columns are x, y, z, r.
    labels : list[str] | None, optional
        The list of labels to be assigned to each control point. If None, the labels will be
        the index of the point in the array, by default None.
    descriptions : list[str] | None, optional
        The list of descriptions to be assigned to each control point. If None, the descriptions will be
        empty strings, by default None.

    Returns
    -------
    str
        The json-formatted string that can be used to open an Open Curve object in 3D Slicer.
    """
    # Check the input array
    if len(arr.shape) != 2:
        raise RuntimeError(f"Unsupported input array shape: {arr.shape}. Must be 2D.")
    if arr.shape[1] not in [3, 4]:
        raise RuntimeError(f"Unsupported input array shape: {arr.shape}. Must have 3 or 4 columns x, y, z (,r).")
    # Create the control points list
    control_points = []
    for i_, p in enumerate(arr):
        point = [p[0], p[1], p[2]]
        cp_ = {
            "id": str(i_),
            "label": str(labels[i_]) if not labels is None else str(i_),
            "description": str(descriptions[i_]) if not descriptions is None else "",
            "associatedNodeID": "",
            "position": point,
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], # Identity 3x3 matrix. Note: this does not change anything in the final visualisation
            "selected": False,
            "locked": True,
            "visibility": True,
            "positionStatus": "defined"
        }
        control_points.append(cp_)
    measurements = []
    markups = [
        {
            "type": "Fiducial",
            "coordinateSystem": "RAS", # Right, Anterior, Superior
            "coordinateUnits": "mm",
            "locked": True,
            "fixedNumberOfControlPoints": True,
            "labelFormat": "%N-%d",
            "lastUsedControlPointNumber": i_,
            "controlPoints": control_points, 
            "measurements": measurements
        }
    ]
    slicer_dict = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": markups
    }
    # Make the json string
    json_string = json.dumps(slicer_dict, indent=4)
    return json_string

#######################
# SimpleCenterlineGraph
#######################

from ...node.node import ArteryNodeTopology, ArteryNodeSide
from ...graph.graph import SimpleCenterlineGraph

def convert_graph_to_3dslicer_opencurve(graph: SimpleCenterlineGraph, save_directory: str, affine_transformation_matrix: numpy.ndarray | None = None) -> None:
    """This function converts each segment, from ostium to endpoint, into an open curve
    that can be loaded directly in 3D Slicer.
    
    The 3D Slicer curve control points coordinate system is the RAS (Right, Anterior, Superior) coordinate system.
    
    Parameters
    ----------
    save_directory : str
        The directory where the curves will be saved.
        If the directory is just a name, it will be created inside the current working directory.
    affine_transformation_matrix : numpy.ndarray, optional
        The affine transformation matrix to apply to the points, by default None.
        If None, the identity transformation is applied.
    
    Raises
    ------
    FileNotFoundError
        If the save_directory does not exist or cannot be created.
    ValueError
        If the affine_transformation_matrix is not a 4x4 matrix.
    """
    # Directory handling
    if not os.path.exists(save_directory):
        try:
            os.mkdir(save_directory)
        except:
            raise FileNotFoundError(f"Directory {save_directory} does not exist and cannot be created.")
    # Affine transformation matrix handling
    if affine_transformation_matrix is None:
        affine_transformation_matrix = numpy.identity(4)
    else:
        if affine_transformation_matrix.shape != (4, 4):
            raise ValueError(f"Affine transformation matrix must be a 4x4 matrix, not {affine_transformation_matrix.shape}.")
    # Cycle through all endpoints
    for n in graph.nodes:
        if graph.nodes[n]['topology_class'] == ArteryNodeTopology.ENDPOINT:
            endpoint_node_id = n
            # get coronary ostium node id that is connected to this endpoint
            ostia_node_id = graph.get_relative_coronary_ostia_node_id(endpoint_node_id)
            for ostium_node_id in ostia_node_id:
                # continue if the returned ostium is None
                if ostia_node_id is None:
                    continue
                # get the path from ostium to endpoint
                path = networkx.algorithms.shortest_path(graph, ostium_node_id, endpoint_node_id)
                # create the 3D Slicer open curve file content in json format
                arr_ = numpy.array(
                    [[graph.nodes[n]['x'], graph.nodes[n]['y'], graph.nodes[n]['z']] for n in path]
                )
                # - transform the points according to the transformation matrix
                arr_ = numpy.concatenate((arr_, numpy.ones((arr_.shape[0], 1))), axis=1).T # 4 x N
                arr_ = numpy.matmul(affine_transformation_matrix, arr_).T
                arr_ = arr_[:, :3]
                # - get other data
                labels_ = [n for n in path]
                descriptions_ = [f"{graph.nodes[n]['arterial_tree'].name} {graph.nodes[n]['topology_class'].name}" for n in path]
                # - make the json string through this utility function
                file_content_str = numpy_array_to_open_curve_json(arr_, labels_, descriptions_)
                # create the file
                if graph.nodes[ostium_node_id]['arterial_tree'] == ArteryNodeSide.LEFT:
                    tree = "left"
                if graph.nodes[ostium_node_id]['arterial_tree'] == ArteryNodeSide.RIGHT:
                    tree = "right"
                f_name = f"{tree}_arterial_segment_{ostium_node_id}_to_{endpoint_node_id}.SlicerOpenCurve.mkr.json"
                f_path = os.path.join(save_directory, f_name)
                f = open(f_path, "w")
                # write the file
                f.write(file_content_str)
                f.close()

def convert_graph_to_3dslicer_fiducials(graph: SimpleCenterlineGraph, save_filename: str, affine_transformation_matrix: numpy.ndarray | None = None) -> None:
    """This function converts the whole graph into a fiducial object (a list of markers)
    that can be loaded directly in 3D Slicer.

    The 3D Slicer fiducials coordinate system is the RAS (Right, Anterior, Superior) coordinate system.
    
    Parameters
    ----------
    save_filename : str
        The file where the fiducials will be saved.
        It must end with ".SlicerFiducial.mkr.json", else everything after the first "." will be replaced by
        the correct extension.
    affine_transformation_matrix : numpy.ndarray, optional
        The affine transformation matrix to apply to the points, by default None.
        If None, the identity transformation is applied.
    
    Raises
    ------
    FileNotFoundError
        If the save_filename does not exist or cannot be created.
    ValueError
        If the affine_transformation_matrix is not a 4x4 matrix.
    """
    # Directory handling
    dir_, f_ = os.path.split(save_filename)
    if not os.path.exists(dir_):
        try:
            os.mkdir(dir_)
        except:
            raise FileNotFoundError(f"Directory {dir_} does not exist and cannot be created.")
    # Affine transformation matrix handling
    if affine_transformation_matrix is None:
        affine_transformation_matrix = numpy.identity(4)
    else:
        if affine_transformation_matrix.shape != (4, 4):
            raise ValueError(f"Affine transformation matrix must be a 4x4 matrix, not {affine_transformation_matrix.shape}.")
    # Handle file name
    if not f_.endswith(".SlicerFiducial.mkr.json"):
        f_ = f_.split(".")[0]
        f_ += ".SlicerFiducial.mkr.json"
        save_filename = os.path.join(dir_, f_)
    # Create the 3D Slicer fiducials file content in json format
    arr_ = numpy.array(
        [[graph.nodes[n]['x'], graph.nodes[n]['y'], graph.nodes[n]['z']] for n in graph.nodes]
    )
    # - transform the points according to the transformation matrix
    arr_ = numpy.concatenate((arr_, numpy.ones((arr_.shape[0], 1))), axis=1).T # 4 x N
    arr_ = numpy.matmul(affine_transformation_matrix, arr_).T
    arr_ = arr_[:, :3]
    # - get other data
    labels_ = [n for n in graph.nodes]
    descriptions_ = [f"{graph.nodes[n]['arterial_tree'].name} {graph.nodes[n]['topology_class'].name}" for n in graph.nodes]
    file_content_str = numpy_array_to_fiducials_json(arr_, labels_, descriptions_)
    # create and write the file
    f = open(save_filename, "w")
    f.write(file_content_str)
    f.close()


if __name__ == "__main__":
    print("hcatnetwork.utils.slicer.slicer")
