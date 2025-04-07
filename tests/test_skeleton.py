import pytest
import os

import forsys as fs

@pytest.fixture
def cells_3_orient1():
    filename = os.path.join("tests", "data", "skeleton", "3_no_cells.npy")
    skeleton = fs.skeleton.Skeleton(filename,
                                    minimum_distance=10)
    vertices, edges, cells = skeleton.create_lattice()
    frame = fs.frames.Frame(0,
                            vertices,
                            edges,
                            cells)
    yield frame

def test_3_orient1(cells_3_orient1):
    all_edges = {tuple(sorted(e.get_vertices_id())): e.id for e in cells_3_orient1.edges.values()}
    assert len(cells_3_orient1.vertices[368].ownCells) == 2
    assert len(cells_3_orient1.vertices[368].ownEdges) == 3

    assert tuple(sorted([350, 1764])) not in all_edges
    assert tuple(sorted([1909, 2131])) not in all_edges
    assert tuple(sorted([2071, 1209])) not in all_edges
