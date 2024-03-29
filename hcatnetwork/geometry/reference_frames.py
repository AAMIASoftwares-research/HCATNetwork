"""Collection of functions to retrieve an orthogonal coordinates system (orthogonal axes)
from a given point.
"""
import numpy

def get_orthogonal_axes(point, normal, x_axis_suggestion=None, suggestion_type="vector"):
    """Get an orthogonal axes from a given point and a normal vector.

    Args:
        point (np.ndarray): A 3D point.
        normal (np.ndarray): A 3D normal vector.
        x_axis_suggestion (np.ndarray, optional): A 3D vector to suggest the x axis. Defaults to None.
            If left empty, the x axis will be created from scratch.
        suggestion_type (str, optional): The type of suggestion. Can be "vector" or "point". Defaults to "vector".
            If suggestion_type is "vector", x_axis_suggestion is a vector that will be used as the x axis after being
            projected on the plane having normal as its normal.
            If suggestion_type is "point", x_axis_suggestion is a point towards which the x axis will be directed. 

    Returns:
        tuple: A tuple containing the orthogonal axes (x, y, z) as 3D vectors.
        z is actually the normal vector.

    Usage:
    In any dimension, to obtain the point p in the reference system defined by the
    unit vectors v0, v1, v2, centered in O, we can do:
    ```
    p_proj_new_reference_system = numpy.dot(p - O, v0)*v0 + numpy.dot(p - O, v1)*v1 + numpy.dot(p - O, v2)*v2
    ```

    This can be used to project any point in the 3D space on this new coordinate system.
    For example, to project a point p on the plane defined by the point and the normal, we can do:
    ```
    p = numpy.array([1, 2, 3])
    normal = numpy.array([0, 0, 1])
    a, b, normal = get_orthogonal_axes(point, normal)
    p_proj_2d_new_reference_system = numpy.array([numpy.dot(p - point, a), numpy.dot(p - point, b)])
    # or
    p_proj_2d_new_reference_system = ( numpy.dot(p - point, a)*a + numpy.dot(p - point, b)*b )[:2]
    ```
    """
    # clean suggestion
    if x_axis_suggestion is not None:
        if suggestion_type not in ["vector", "point"]:
            raise ValueError("suggestion_type must be 'vector' or 'point'")
    # clean normal
    normal = normal / numpy.linalg.norm(normal)
    # setting the reference for the new x axis (a) on the plane of the circle
    if x_axis_suggestion is None:
        # This code works fine if we have no reference and we want to create the new x axis (a) from scratch
        #
        # - solving dot(a,v) = 0
        #         a0*v0 + a1*v1 + a2*v2 = 0
        #         a0*v0 = -a1*v1 - a2*v2
        #         a0 = (-a1*v1 - a2*v2) / v0 , v0 != 0, with a1, a2 freely chosen
        # or, i could just decide the a axis myself
        if normal[0] == 0:
            a0, a1, a2 = 1.0, 0.0, 0.0
        else:
            a1, a2 = 1.0, 0.0
            a0 = (-a1*normal[1] - a2*normal[2]) / normal[0]
        a = numpy.array([a0, a1, a2])
        a = a / numpy.linalg.norm(a)
    else:
        if suggestion_type == "point":
            if (x_axis_suggestion == point).all():
                x_axis_suggestion[0] += 1
            x_axis_suggestion -= point # vector from point to circle_x_axis
        x_axis_suggestion = x_axis_suggestion - numpy.dot(x_axis_suggestion, normal)*normal # project onto plane with normal "normal"
        a = x_axis_suggestion / numpy.linalg.norm(x_axis_suggestion) # normalize
    # - solving b = a x v to find third axis (b <-> y)
    #   the - sign so that it follows the right hand rule
    b = numpy.cross(a, normal)
    b = - b / numpy.linalg.norm(b)
    return (a, b, normal)

def transform_to_reference_frame(origin, x_versor, y_versor, z_versor, point):
    """Get the coordinates of a point in a reference frame defined by the given versors.

    Args:
        origin (np.ndarray): The origin of the reference frame.
        x_versor (np.ndarray): The versor of the x axis.
        y_versor (np.ndarray): The versor of the y axis.
        z_versor (np.ndarray): The versor of the z axis.
        point (np.ndarray): The coordinates of the point in the global reference frame.

    Returns:
        np.ndarray: The coordinates of the point in the reference frame.

    Usage:
    In any dimension, to obtain the point p in the reference system defined by the
    unit vectors v0, v1, v2, centered in O, we can do:
    ```
    p_proj_new_reference_system = numpy.dot(p - O, v0)*v0 + numpy.dot(p - O, v1)*v1 + numpy.dot(p - O, v2)*v2
    ```

    To obtain the coordinates of a point on the plane, say, (x,y) of the reference frame passed as arguments, we can do:
    ```
    output_2d = output[:2] # it is that easy!
    """
    return numpy.dot(point - origin, x_versor)*numpy.array([1.0, 0.0, 0.0]) + numpy.dot(point - origin, y_versor)*numpy.array([0.0, 1.0, 0.0]) + numpy.dot(point - origin, z_versor)*numpy.array([0.0, 0.0, 1.0])

def transform_from_reference_frame(origin, x_versor, y_versor, z_versor, point):
    """Get the coordinates of a point in the global reference frame from the given reference frame.

    Args:
        origin (np.ndarray): The origin of the reference frame expressed in global reference frame coordinates.
        x_versor (np.ndarray): The versor of the x axis in the global reference frame.
        y_versor (np.ndarray): The versor of the y axis in the global reference frame.
        z_versor (np.ndarray): The versor of the z axis in the global reference frame.
        point (np.ndarray): The coordinates of the point in the reference frame.

    Returns:
        np.ndarray: The coordinates of the point in the global reference frame.
    """
    return origin + point[0]*x_versor + point[1]*y_versor + point[2]*z_versor

if __name__ == "__main__":
    # Example: creating a new reference frame
    point = numpy.array([1, 2, 3])
    normal = numpy.random.rand(3)
    a, b, normal = get_orthogonal_axes(point, normal)
    print(a, b, normal)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis("equal")
    ax.set(xlim=(point[0]-1, point[0]+1), ylim=(point[1]-1, point[1]+1), zlim=(point[2]-1, point[2]+1))
    ax.quiver(*point, *a, color="r")
    ax.quiver(*point, *b, color="g")
    ax.quiver(*point, *normal, color="b")
    ax.legend(["a -> x", "b -> y", "normal -> z"])
    plt.show()
    # Casting a point from the space to the new reference frame's XY plane
    p_to_project = numpy.array([3, -1, 2])
    p_projected = transform_to_reference_frame(point, a, b, normal, p_to_project)
    p_projected_2d = p_projected.copy()
    p_projected_2d[2] = 0
    p_projected_in_global = transform_from_reference_frame(point, a, b, normal, p_projected_2d)
    fig = plt.figure()
    # - 3d
    ax = fig.add_subplot(121, projection='3d')
    ax.quiver(*point, *a, color="r", label="a -> x")
    ax.quiver(*point, *b, color="g", label="b -> y")
    ax.quiver(*point, *normal, color="b", label="normal -> z")
    ax.scatter(*point, color="black", label="Origin of new axes")
    ax.scatter(*p_to_project, color="gold", label="Original point")
    ax.scatter(*p_projected_in_global, color="red", label="Projected point")
    ax.plot([p_to_project[0], p_projected_in_global[0]], [p_to_project[1], p_projected_in_global[1]], [p_to_project[2], p_projected_in_global[2]], color="gray", linestyle="--", linewidth=0.4)
    ax.autoscale(enable=True, axis="both", tight=True)
    ax.axis("equal")
    ax.legend()
    # - 2d
    ax2 = fig.add_subplot(122)
    ax2.axis("equal")
    ax2.quiver(0, 0, 1, 0, color="r", label="a -> x")
    ax2.quiver(0, 0, 0, 1, color="g", label="b -> y")
    ax2.scatter(0, 0, color="black", label="Origin of new axes")
    ax2.scatter(*p_projected[:2], color="red", label="Projected point")
    ax2.legend()
    plt.show()



    