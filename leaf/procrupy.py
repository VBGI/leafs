"""
Statistical methods for shapes analysis.

Procrustes Analysis: calculates a linear transformation sufficient to align
two shapes.
Generalized Procrustes Analysis: calculates an approximate mean of a set
of shapes.

"""
import numpy as np


__version__ = '0.1'


def _centralize(shape):
    """Create a shape which is just like the input, but aligned to origin.

    The origin is (0; 0) in 2D case.
    The input shape stays unchanged. It's expected to be a numpy.array.

    """
    centroid = shape.mean(0)
    return shape - centroid


def _normalize(shape):
    """Create a shape which is just like the input but scaled by 1/norm(shape).

    The shape is expected to be a numpy.array and have a non-zero determinant.

    """
    if np.linalg.norm(shape):
        return shape.astype(float) / np.linalg.norm(shape)


def _align_first_to_second(first_shape, second_shape):
    """Create a shape which is first_shape aligned to the second_shape.

    The rotation is obtained with help of singular value decomposition.
    Both input shapes are expected to be numpy.array.

    """
    u, _, vh = np.linalg.svd(np.dot(second_shape.conj().transpose(),
                                    first_shape))
    v = vh.conj().transpose()
    uh = u.conj().transpose()
    return np.dot(np.dot(first_shape, v), uh)


def procrustes_analysis(shape, example_shape, is_scaling_allowed=True):
    """
    Ordinary Procrustes Analysis of a pair of shapes.

    Calculate a linear transformation which makes the points of one shape
    conform to the points in another shape in the best way.
    Return (rotation, scale, translation) tuple.
    If scaling is disallowed, return scale = 1.

    >>> example_shape = numpy.array([[1, 1], [5, 1], [5, 5]])
    >>> shape = numpy.array([[-2, -2], [-10, -2], [-10, -10]])
    >>> rotation, scale, translation = procrustes_analysis(shape,
                                                           example_shape)
    >>> rotation
    array([[ -1.00000000e+00, 0.00000000e+00],
    [ 1.57019580e-16, -1.00000000e+00]])
    >>> scale
    0.49999999999999983
    >>> translation
    array([ 2.22044605e-15, 1.77635684e-15])

    """
    # calculate the centroids to align both shapes to (0,0)
    example_centered, shape_centered = map(_centralize, (example_shape, shape))

    # calculate the norms to scale both shapes to common size
    example_scaled, shape_scaled = map(_normalize, (example_centered,
                                                    shape_centered))

    # calculate rotation
    u, s, vh = np.linalg.svd(
        np.dot(example_scaled.conj().transpose(), shape_scaled))
    s_trace = s.sum()
    v = vh.conj().transpose()
    uh = u.conj().transpose()
    rotation = np.dot(v, uh)

    # calculate scale
    if is_scaling_allowed:
        example_norm = np.linalg.norm(example_centered)
        norm = np.linalg.norm(shape_centered)
        scale = s_trace * example_norm / norm
    else:
        scale = 1

    # calculate translation
    centroid = shape.mean(0)
    example_centroid = example_shape.mean(0)
    translation = example_centroid - (scale * np.dot(centroid, rotation))

    return (rotation, scale, translation)


def generalized_procrustes_analysis(shapes, threshold=0.00001):
    """
    Generalized Procrustes Analysis of an array of shapes.

    Calculate (with precision set by the threshold parameter)
    a coordinate reference with regard to position, scale, and rotation,
    to which all shapes are aligned.

    >>> shapes = numpy.array([[[1, 1], [5, 5], [10, 3]],
                              [[2, 2], [7, 7], [15, 5]],
                              [[-5, -5], [-1, -1], [-10, -3]]])
    >>> mean_shape = generalized_procrustes_analysis(shapes, threshold=0.001)
    >>> mean_shape
    array([[-2, -1], [2, 3], [4, 1]])

    """
    # don't modify the original shapes array + avoid errors with integers
    _shapes = shapes.astype(float)

    # select the first shape to be the first approximate mean
    mean = _shapes[0]

    finished = False
    while not finished:
        # move all shapes so that their centers are aligned to origin,
        # (0; 0) in 2-dimensional case
        _shapes = map(_centralize, _shapes)

        # make all the shapes roughly the same size
        _shapes = map(_normalize, _shapes)

        # rotate each shape to align with the current approximate mean shape
        _shapes = [_align_first_to_second(shape, mean) for shape in _shapes]

        # calculate the new approximate mean from all shapes
        new_mean = sum((shape for shape in _shapes)) / len(_shapes)

        # evaluate the difference between the previous and new mean
        finished = all([new_mean_el - mean_el < threshold
                        for (new_mean_el, mean_el)
                        in zip(new_mean.flatten(), mean.flatten())])

        mean = new_mean

    return mean


