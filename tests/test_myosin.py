import pytest
import forsys as fs
import numpy as np
from PIL import Image

from scipy import interpolate

line_types = ["horizontal", "vertical", "straight_line", "curved", "general"]
layers = [0, 1, 2]

truth_values_average = {
    "horizontal-0": 100.0,
    "vertical-0": 100.0,
    "straight_line-0": 100.0,
    "curved-0": 100.0,
    "general-0": 100.0
}

def generate_xy_values(line_type, n_vertices, pixel_spacing, x_offset):
    regular_spacing = [x_offset + pixel_spacing * ii for ii in range(n_vertices)]

    if line_type == "horizontal":
        x_values = regular_spacing
        y_values = np.repeat(5, n_vertices)
    elif line_type== "vertical":
        x_values = np.repeat(5, n_vertices)
        y_values = regular_spacing
    elif line_type == "straight_line":
        x_values = regular_spacing
        y_values = regular_spacing
    elif line_type == "general":
        x_values = regular_spacing
        y_values = [int(ii + np.random.uniform(-5, 5)) for ii in x_values]
    elif line_type == "curved":
        x_values = regular_spacing
        y_values = [int(np.sqrt(40**2 - (ii - x_offset)**2)) for ii in regular_spacing]
    else:
        raise NotImplementedError
    
    return x_values, y_values


@pytest.fixture(params=line_types)
def set_up_big_edges(request):
    n_vertices = 20

    x_values, y_values = generate_xy_values(request.param, n_vertices, 2, 20)
    
    vertices = {}
    edges = {}
    truth_values = {"integration": {0: 0,
                                    1: 0,
                                    2: 0}}
    
    image_data = np.zeros((200, 200))
        
    vid = 0

    truth_ave_l0 = []
    truth_ave_l1 = []
    truth_ave_l2 = []

    for vid in range(n_vertices):
        vertices[vid] = fs.vertex.Vertex(vid, x_values[vid], y_values[vid])
        image_data[y_values[vid], x_values[vid]] = 100
        if vid > 0:
            edges[vid - 1] = fs.edge.SmallEdge(vid - 1, vertices[vid - 1], vertices[vid])

        elements_plus_one = fs.myosin.get_layer_elements((x_values[vid], y_values[vid]), 1)
        elements_plus_two = fs.myosin.get_layer_elements((x_values[vid], y_values[vid]), 2)
        for x_val, y_val in elements_plus_one:
            if image_data[y_val, x_val] != 100:
                image_data[y_val, x_val] = 50
        for x_val, y_val in elements_plus_two:
            if image_data[y_val, x_val] != 100 and image_data[y_val, x_val] != 50:
                image_data[y_val, x_val] = 20

    big_edge = fs.edge.BigEdge(0, list(vertices.values()))
    image = Image.fromarray(image_data)
    
    # quantification
    for vid in range(n_vertices):
        per_vertex_layers = {0: [],
                             1: [],
                             2: []}
        # average
        for layer in layers:
            elements_in_the_layer = fs.myosin.get_layer_elements((x_values[vid], y_values[vid]), layer)
            for x_val, y_val in elements_in_the_layer: 
                per_vertex_layers[layer].append(image_data[y_val, x_val])

                # n_elements[layer] += 1

        truth_ave_l0.append(np.median(per_vertex_layers[0]))
        truth_ave_l1.append(np.median(per_vertex_layers[1]))
        truth_ave_l2.append(np.median(per_vertex_layers[2]))

    # integration
    interpolation = lambda x: int(interpolate.interp1d(big_edge.xs, big_edge.ys, kind="linear")(x))
    xrange = range(min(big_edge.xs), max(big_edge.xs) + 1)

    elements_0 = {element for x_val in xrange for element in fs.myosin.get_layer_elements((x_val, interpolation(x_val)), 0)}
    elements_1 = {element for x_val in xrange for element in fs.myosin.get_layer_elements((x_val, interpolation(x_val)), 1)}
    elements_2 = {element for x_val in xrange for element in fs.myosin.get_layer_elements((x_val, interpolation(x_val)), 2)}

    truth_values["integration"][0] = sum([image_data[int(y_val), int(x_val)] for x_val, y_val in elements_0]) / len(elements_0)
    truth_values["integration"][1] = sum([image_data[int(y_val), int(x_val)] for x_val, y_val in elements_1]) / len(elements_1)
    truth_values["integration"][2] = sum([image_data[int(y_val), int(x_val)] for x_val, y_val in elements_2]) / len(elements_2)

    truth_values["average"] = {
                                0: np.mean(truth_ave_l0),
                                1: np.mean(truth_ave_l1),
                                2: np.mean(truth_ave_l2)
                                }

    yield [big_edge], image, truth_values


@pytest.mark.parametrize("layers", layers)
# new myosin integration makes test obsolte. Should rewrite test
# @pytest.mark.parametrize("integrating", [True, False])
@pytest.mark.parametrize("integrating", [False])
def test_get_intensities(set_up_big_edges, integrating, layers):
    big_edge, image, truth_values = set_up_big_edges

    myosin = fs.myosin.get_intensities(big_edge,
                              image,
                              integrate=integrating,
                              normalize=None,
                              corrected_by_tilt=False,
                              layers=layers)
    
    calculation = "integration" if integrating else "average"
    assert round(myosin[0], 3) == round(truth_values[calculation][layers], 3)


@pytest.mark.parametrize("layers", layers)
def test_get_layer_elements(layers):
    elements = fs.myosin.get_layer_elements([0, 0], layers)
    layer1 = set([(0, 0),
                  (-1, 0),
                  (-1, 1),
                  (0, 1),
                  (1, 1),
                  (1, 0),
                  (1, -1),
                  (0, -1),
                  (-1, -1),
                  (-1, 0)])
    layer2 = set([(-2, 0),
                  (-2, 1),
                  (-2, 2),
                  (-1, 2),
                  (0, 2),
                  (1, 2),
                  (2, 2),
                  (2, 1),
                  (2, 0),
                  (2, -1),
                  (2, -2),
                  (1, -2),
                  (0, -2),
                  (-1, -2),
                  (-2, -2),
                (-2, -1)])
    layer2.update(layer1)
    if layers == 0:
        assert elements == [(0, 0)]
    elif layers == 1:
        assert set(elements) == set(layer1)
    elif layers == 2:
        assert set(elements) == set(layer2)
