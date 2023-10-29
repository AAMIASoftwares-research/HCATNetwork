"""Here are gathered some utilities to interface numpy and HCATNetwork objects with 3D Slicer.
https://www.slicer.org/

"""

import os
import json
import numpy

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
