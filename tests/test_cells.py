import pytest

import os
import numpy as np
import forsys as fs

@pytest.fixture
def set_up_cell():
    def get_hexagon_vertices(apothem=1):
        hex_vertices = [[apothem * np.cos(ang), apothem * np.sin(ang)] 
                        for ang in np.arange(0, 2*np.pi, 60 * np.pi / 180)]
        return hex_vertices

    vertices = {}
    edges = {}

    for vid, vertex in enumerate(get_hexagon_vertices()):
        vertices[vid] = fs.vertex.Vertex(vid, vertex[0], vertex[1])
        if vid > 0:
            edges[vid - 1] = fs.edge.SmallEdge(vid - 1, vertices[vid - 1], vertices[vid])
    edges[len(edges)] = fs.edge.SmallEdge(len(edges), vertices[0], vertices[vid])


    cell = fs.cell.Cell(0, list(vertices.values()), {})


    yield cell

@pytest.fixture()
def set_up_4_cells():
    filename = os.path.join("tests", "data", "skeleton", "triangle_holes.npy")
    skeleton = fs.skeleton.Skeleton(filename,
                                    minimum_distance=7)
    vertices, edges, cells = skeleton.create_lattice()
    frame = fs.frames.Frame(0,
                            vertices,
                            edges,
                            cells)
    yield frame

def test_get_area(set_up_cell):
    assert abs(np.round(set_up_cell.get_area(), 6)) == np.round(1.5 * np.sqrt(3), 6)


def test_get_area_sign(set_up_cell):
    assert set_up_cell.get_area_sign() == -1
    set_up_cell.vertices = set_up_cell.vertices[::-1]
    assert set_up_cell.get_area_sign() == 1


def test_get_perimeter(set_up_cell):
    assert int(set_up_cell.get_perimeter()) == 6


def test_get_edges(set_up_4_cells):
    # TODO: Create a cell from scratch to make it independent of reading npy
    edges_cell_1 = [542, 545, 552, 561, 570, 578, 586, 594, 602, 611, 620, 628,
                    1678, 1670, 1661, 1654, 1645, 1637, 1629, 1620, 1612, 1603,
                    1595, 1586, 1576, 1568, 1804, 1797, 1789, 1781, 1773, 2337,
                    2330, 2322, 2314, 2306, 2297, 2290, 2282, 2273, 2264, 896,
                    904, 912, 921, 929, 937, 946, 955, 963, 971, 980, 988, 996,
                    1004, 1012, 1021, 1030, 1038, 1047, 1055, 1064, 1072, 1080,
                    1089, 1097, 1106, 1114, 1123, 1132, 1141, 1150, 1158, 1167]
    assert set_up_4_cells.cells[1].get_edges() == edges_cell_1
