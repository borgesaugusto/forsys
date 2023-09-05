from PIL import Image
import numpy as np

def get_intensities(frame, 
                    tiff_path, 
                    normalized=True, 
                    method="average", 
                    corrected_by_tilt=False, 
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
        for vertex in big_edge.vertices:
            intensity = np.median(get_intensity(image_to_use, vertex, layer=1))
            # number_of_connections = len(vertex.ownEdges)
            # Normalize by the number of edges connected to it
            # intensity = intensity / number_of_connections
            intensities_per_edge.append(intensity)

        try:
            key_to_use = big_edges_to_use.index(big_edge)
        except ValueError:
            key_to_use = "ext_"+str(be_id)
        intensities[key_to_use] = np.mean(intensities_per_edge)

    # divide by maximum
    # max_intensity = np.amax(im)
    # intensities = {k: v/max_intensity for k, v in intensities.items()}
    intensities_only_internal = {k: v for k, v in intensities.items() if not isinstance(k, str)}

    if normalized:
        if method == "maximum":
            raise(NotImplementedError)
        elif method == "average":
            mean_value = np.mean(list(intensities_only_internal.values()))
            intensities_only_internal = {k: v/mean_value for k, v in intensities_only_internal.items()}
        else:
            raise(NotImplemented)

    # for k, v in intensities_only_internal.items():
    #     print(f"This edge {frame.big_edges_list[k]} has intensity: {round(v, 3)} and force {round(frame.forces[k], 3)}")
        
    return intensities_only_internal


def get_intensity(image, vertex, layer=1):
    intensity = []
    if layer == 0:
        intensity.append(image.getpixel((vertex.x, vertex.y)))
    elif layer == 1:
        intensity.append(image.getpixel((vertex.x, vertex.y)))
        intensity.append(image.getpixel((vertex.x + 1, vertex.y)))
        intensity.append(image.getpixel((vertex.x + 1, vertex.y + 1)))
        intensity.append(image.getpixel((vertex.x, vertex.y + 1)))
        intensity.append(image.getpixel((vertex.x - 1, vertex.y + 1)))
        intensity.append(image.getpixel((vertex.x - 1, vertex.y)))
        intensity.append(image.getpixel((vertex.x - 1, vertex.y - 1)))
        intensity.append(image.getpixel((vertex.x, vertex.y - 1)))
        intensity.append(image.getpixel((vertex.x + 1, vertex.y - 1)))
    else:
        raise(NotImplemented)
    
    return intensity

def modified_intensities(image):
    import scipy.optimize as scop
    import matplotlib.pyplot as plt
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



    