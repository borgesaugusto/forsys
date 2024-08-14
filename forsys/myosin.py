from PIL import Image
import numpy as np
import itertools
import math
from scipy import interpolate

def read_myosin(frame,
                tiff_path,
                integrate=False,
                normalize="average",
                layers: int = 1,
                **kwargs):
    image = Image.open(tiff_path)

    if kwargs.get("use_all", False):
        big_edges_to_use = frame.big_edges_list
    else:
        big_edges_to_use = frame.internal_big_edges

    return get_intensities(big_edges_to_use,
                           image,
                           integrate,
                           normalize,
                           layers,
                           **kwargs)


def get_intensities(big_edges,
                    image,
                    integrate=False,
                    normalize="average",
                    layers: int = 1,
                    **kwargs):
    intensities = {}

    image_to_use = image

    for be_id, big_edge in enumerate(big_edges):
        intensities_per_edge = []
        if integrate:
            vertices, length = get_interpolation(big_edge, layers, **kwargs)
            intensity_to_use = sum(map(image_to_use.getpixel,
                                       vertices)) / length
        else:
            for vertex in big_edge.vertices:
                intensity = get_intensity(image_to_use, vertex, layers, **kwargs)
                intensities_per_edge.append(intensity)

            intensity_to_use = np.mean(list(map(np.median,
                                                intensities_per_edge)))
        try:
            key_to_use = big_edges.index(big_edge)
        except ValueError:
            key_to_use = "ext_"+str(be_id)

        intensities[key_to_use] = intensity_to_use
        # big_edge.gt = intensities[key_to_use]

    intensities_only_internal = {k: v for k, v in intensities.items() if not isinstance(k, str)}

    if normalize == "maximum":
        raise NotImplementedError
    elif normalize == "average":
        mean_value = np.mean(list(intensities_only_internal.values()))
        intensities_only_internal = {k: v/mean_value for k, v in intensities_only_internal.items()}

    
    for be_id, big_edge in enumerate(big_edges):
        big_edge.gt = intensities_only_internal[be_id]

    return intensities_only_internal


def get_intensity(image, vertex, layers=1, **kwargs):
    offset = kwargs.get("offset", [0, 0])
    rescale = kwargs.get("rescale", [1, 1])
    x_y_position = [(vertex.x * rescale[0]) + offset[0], (vertex.y * rescale[1]) + offset[1]]
    pixel_positions = get_layer_elements(x_y_position, layers)
    return list(map(image.getpixel, pixel_positions))


def get_layer_elements(position, layers=1):
    pixel_positions = []
    layer_range = np.arange(-layers, layers + 1)
    for (ii, kk) in itertools.product(layer_range, layer_range):
        xy_pixel = (position[0] + ii, position[1] + kk)
        pixel_positions.append(xy_pixel)
    return pixel_positions


def get_interpolation(big_edge, layers, **kwargs):
    offset = kwargs.get("offset", [0, 0])
    rescale = kwargs.get("rescale", [1, 1])
    length = 0
    all_vertices = set()
    xs_values = [(value * rescale[0]) + offset[0] for value in big_edge.xs]
    ys_values = [(value * rescale[1]) + offset[1] for value in big_edge.ys]
    # xy_pairs = list(zip(big_edge.xs, big_edge.ys))
    xy_pairs = list(zip(xs_values, ys_values))

    for ii in range(1, len(xy_pairs)):
        current_vertex = list(map(math.ceil, xy_pairs[ii - 1]))
        next_vertex = list(map(math.ceil, xy_pairs[ii]))
        all_vertices.update(walk_two_vertices(current_vertex, next_vertex, layers))
        length += np.linalg.norm(np.array(xy_pairs[ii - 1]) - np.array(xy_pairs[ii]))

    return all_vertices, length


def walk_two_vertices(v0, v1, layers):
    vertices_to_return = set()
    diff_x = abs(v0[0] - v1[0])
    diff_y = abs(v0[1] - v1[1])
    axis = 0 if diff_x > diff_y else 1
    delta = 1 if v0[axis] < v1[axis] else -1
    interpolation = interpolate.interp1d([v0[axis], v1[axis]],
                                                [v0[axis - 1], v1[axis - 1]],
                                                kind="linear")
    for value in range(v0[axis], v1[axis], delta):
        if axis == 0:
            position = (value, interpolation(value))
        else:
            position = (interpolation(value), value)

        vertices_to_return.update(get_layer_elements(position, layers))
    return vertices_to_return