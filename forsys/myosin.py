from PIL import Image
import numpy as np
import itertools


def get_intensities(frame,
                    tiff_path,
                    integrate=False,
                    normalize="average",
                    corrected_by_tilt=False,
                    layers: int = 1,
                    **kwargs):

    im = Image.open(tiff_path)

    intensities = {}
    if kwargs.get("use_all", False):
        big_edges_to_use = frame.big_edges_list
    else:
        big_edges_to_use = frame.internal_big_edges

    if corrected_by_tilt:
        image_to_use = Image.fromarray(modified_intensities(im))
    else:
        image_to_use = im

    for be_id, big_edge in enumerate(big_edges_to_use):
        intensities_per_edge = []

        if integrate:
            vertices, length = get_interpolation(big_edge)
            intensity_to_use = sum(map(image_to_use.getpixel,
                                       vertices)) / length
        else:
            for vertex in big_edge.vertices:
                intensity = get_intensity(image_to_use, vertex, layers=layers)
                # number_of_connections = len(vertex.ownEdges)
                # Normalize by the number of edges connected to it
                # intensity = intensity / number_of_connections
                intensities_per_edge.append(intensity)

            intensity_to_use = np.mean(list(map(np.median,
                                                intensities_per_edge)))
        try:
            key_to_use = big_edges_to_use.index(big_edge)
        except ValueError:
            key_to_use = "ext_"+str(be_id)

        intensities[key_to_use] = intensity_to_use
        big_edge.gt = intensities[key_to_use]

    # divide by maximum
    # max_intensity = np.amax(im)
    # intensities = {k: v/max_intensity for k, v in intensities.items()}
    intensities_only_internal = {k: v for k, v in intensities.items() if not isinstance(k, str)}

    if normalize == "maximum":
        raise NotImplementedError
    elif normalize == "average":
        mean_value = np.mean(list(intensities_only_internal.values()))
        intensities_only_internal = {k: v/mean_value for k, v in intensities_only_internal.items()}
    else:
        raise NotImplementedError

    return intensities_only_internal


def get_intensity(image, vertex, layers=1):
    pixel_positions = get_layer_elements([vertex.x, vertex.y], layers)
    return list(map(image.getpixel, pixel_positions))


def get_layer_elements(position, layers=1):
    pixel_positions = []
    layer_range = np.arange(-layers, layers + 1)
    for (ii, kk) in itertools.product(layer_range, layer_range):
        xy_pixel = (position[0] + ii, position[1] + kk)
        pixel_positions.append(xy_pixel)
    return pixel_positions


def get_interpolation(big_edge, layers=0):
    # len_vertices = len(big_edge.vertices)
    length = 0
    all_vertices = set()
    y_iteration = False
    for ii in range(0, len(big_edge.vertices) - 1):
        if big_edge.vertices[ii].x < big_edge.vertices[ii + 1].x:
            v0 = big_edge.vertices[ii]
            v1 = big_edge.vertices[ii + 1]
        else:
            v0 = big_edge.vertices[ii + 1]
            v1 = big_edge.vertices[ii]

        difference = (v1.x - v0.x, v1.y - v0.y)

        if difference[0] == 0:
            y_iteration = True
            if v0.y < v1.y:
                iter_interval = range(v0.y, v1.y)
            else:
                iter_interval = range(v1.y, v0.y)
        else:
            iter_interval = range(int(v0.x), int(v1.x + 1))
            slope = difference[1] / difference[0]

        length += np.linalg.norm(difference)
        for position in iter_interval:
            if y_iteration:
                x_position = v0.x
                y_position = position
            else:
                x_position = position
                y_position = slope * (x_position - v0.x) + v0.y

            layer_elements = get_layer_elements((x_position, y_position),
                                                layers)
            all_vertices.update(layer_elements)

    return all_vertices, length


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
