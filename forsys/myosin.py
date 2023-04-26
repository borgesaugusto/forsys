from PIL import Image
import numpy as np

def get_intensities(frame, tiff_path, normalized=True, method="average", **kwargs):
    im = Image.open(tiff_path)

    intensities = {}
    if kwargs.get("use_all", False):
        big_edges_to_use = frame.big_edges_list
    else:
        big_edges_to_use = frame.internal_big_edges
        

    for be_id, big_edge in enumerate(big_edges_to_use):
        intensity = []
        for vertex in big_edge.vertices:
            intensity = np.median(get_intensity(im, vertex, layer=1))
            number_of_connections = len(vertex.ownEdges)
            # Normalize by the number of edges connected to it
            intensity = intensity / number_of_connections
            if isinstance(kwargs.get("correction", False), float):
                intensity = intensity + intensity_correction(vertex, kwargs['correction'])
        try:
            key_to_use = big_edges_to_use.index(big_edge)
        except ValueError:
            key_to_use = "ext_"+str(be_id)
        intensities[key_to_use] = np.median(intensity)
    
    # divide by maximum
    # max_intensity = np.amax(im)
    # intensities = {k: v/max_intensity for k, v in intensities.items()}
    intensities_only_internal = {k: v for k, v in intensities.items() if not isinstance(k, str)}

    if normalized:
        if method == "maximum":
            raise(NotImplemented)
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

def modified_intensities(image_path):
    import scipy.optimize as scop

    image = Image.open(image_path)
    # max_intensity = np.amax(image)
    def linear_function(x, slope, angle, max_intensity):
        return max_intensity - slope * np.tan(angle) * x

    intensity_vs_y = [np.median(np.array(image)[jj, :]) for jj in range(0, image.size[1])]
    
    parameters = scop.curve_fit(linear_function, range(0, 952), intensity_vs_y)
    effective_slope = parameters[0][0] * np.tan(parameters[0][1])

    # pixels = np.array(image)

    # for ii in range(0, image.size[0]):
    #     for jj in range(0, image.size[1]):
    #         pixels[ii, jj] = pixels[ii, jj] + effective_slope * jj

    return effective_slope

def intensity_correction(vertex, effective_slope):
    return effective_slope * vertex.y




    