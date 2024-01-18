from PIL import Image
import numpy as np
import itertools
import math
from scipy import interpolate

from forsys.vertex import Vertex 
from forsys.edge import SmallEdge, BigEdge 


def read_myosin(frame,
                tiff_path,
                integrate=False,
                normalize="average",
                corrected_by_tilt=False,
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
                           corrected_by_tilt,
                           layers,
                           **kwargs)


def get_intensities(big_edges,
                    image,
                    integrate=False,
                    normalize="average",
                    corrected_by_tilt=False,
                    layers: int = 1,
                    **kwargs):
    intensities = {}

    if corrected_by_tilt:
        image_to_use = Image.fromarray(modified_intensities(image))
    else:
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
        key_to_use = big_edges.index(big_edge)
        big_edge.gt = intensities_only_internal[key_to_use]

    return intensities_only_internal


# def rotate_big_edge(big_edge_prev, angle):
#     Rmatrix = np.array([[np.cos(angle), - np.sin(angle)],
#                        [np.sin(angle), - np.cos(angle)]])
    
#     vertices = {}
#     for vid in range(len(big_edge_prev.xs)):
#         new_x, new_y = np.array(np.matmul(Rmatrix, [big_edge_prev.xs[vid], big_edge_prev.ys[vid]]))
#         vertices[10000 + vid] = Vertex(10000 + vid, new_x, new_y)
#         if vid > 0:
#             edges[10000 + vid] = SmallEdge(vid - 1, vertices[vid - 1], vertices[vid])

#     big_edge = BigEdge(0, list(vertices.values()))

#     return big_edge


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


def arc_length(x, y):
    gradient = np.gradient(y, x)
    return np.trapz(np.sqrt(1 + gradient**2))


def modified_intensities(image):
    import scipy.optimize as scop
    # import matplotlib.pyplot as plt
    # import os

    # image = Image.open(image_path)
    # max_intensity = np.amax(image)
    # def linear_function(x, slope, angle, max_intensity):
    #     return max_intensity - slope * np.tan(angle) * x
    def linear_function(x, slope, A):
        return A + slope * x

    yrange = range(0, image.size[1])
    intensity_vs_y = [np.median(np.array(image)[jj, :]) for jj in yrange]

    # fit only the first and last point
    # parameters = scop.curve_fit(linear_function, yrange, intensity_vs_y)
    parameters = scop.curve_fit(linear_function,
                                [yrange[0], yrange[-1]],
                                [intensity_vs_y[0], intensity_vs_y[-1]])

    ####################################
    # modify the myosin intensities now
    ####################################
    corrected_pixels = np.array(image).copy().astype("int32")
    # for ii in range(0, image.size[0]):
    for jj in range(0, image.size[1]):
        corrected_pixels[jj, :] = corrected_pixels[jj, :] - linear_function(jj, parameters[0][0], parameters[0][1])

    # intensity_vs_y_post = [np.median(corrected_pixels[jj, :]) for jj in yrange]
    # _, axs = plt.subplots(2, 1, sharex=True)
    # effective_slope = round(parameters[0][0], 4)
    # a_value = round(parameters[0][1], 4)
    # axs[0].plot(yrange, intensity_vs_y)
    # axs[0].plot(yrange, linear_function(yrange, *parameters[0]), ls="--", color="red")
    # axs[0].text(0.75, 0.85, f"y = {a_value} + {effective_slope} * x",transform=axs[0].transAxes)
    # axs[0].set_ylabel("Intensity")

    # axs[1].plot(yrange, intensity_vs_y_post)
    # axs[1].set_ylabel("Intensity")

    # plt.xlabel("Y-Axis [px]")

    # plt.savefig(os.path.join("results_fit_test", image_path.split("/")[-1].split(".")[0]), dpi=350)
    # plt.close()

    return corrected_pixels
