import pytest

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


def test_get_area(set_up_cell):
    assert abs(np.round(set_up_cell.get_area(), 6)) == np.round(1.5 * np.sqrt(3), 6)


def test_get_area_sign(set_up_cell):
    assert set_up_cell.get_area_sign() == -1
    set_up_cell.vertices = set_up_cell.vertices[::-1]
    assert set_up_cell.get_area_sign() == 1


def test_get_perimeter(set_up_cell):
    assert int(set_up_cell.get_perimeter()) == 6