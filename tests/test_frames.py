import pytest
import numpy as np
from forsys.frames import Frame
from forsys.vertex import Vertex
from forsys.edge import BigEdge, SmallEdge

@pytest.fixture
def set_up_frame():
    vertices = {}
    edges = {}
    cells = {}
    frame = Frame(0, vertices, edges, cells)


    for vid in range(10):
        vertices[vid] = Vertex(vid, vid + 1 , vid + 1)
        if vid > 0:
            edges[vid - 1] = SmallEdge(vid - 1, vertices[vid - 1], vertices[vid])

    frame.big_edges = {
        0: BigEdge(0,
                   vertices=[vertices[0], vertices[1], vertices[2], vertices[3], vertices[4]]),
        1: BigEdge(1, 
                   vertices=[vertices[5],
                             vertices[6],
                             vertices[7],
                             vertices[8],
                             vertices[9]])
    }
    return frame

def test_filter_edges_sg(set_up_frame):
    set_up_frame.filter_edges(method="SG")
    # Check if Savitzky-Golay filter is applied correctly
    assert np.allclose(set_up_frame.big_edges[0].xs, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(set_up_frame.big_edges[0].ys, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(set_up_frame.big_edges[1].xs, [6.0, 7.0, 8.0, 9.0, 10.0])
    assert np.allclose(set_up_frame.big_edges[1].ys, [6.0, 7.0, 8.0, 9.0, 10.0])

def test_filter_edges_none(set_up_frame):
    set_up_frame.filter_edges(method="none")
    # Check if no filtering is applied
    assert np.allclose(set_up_frame.big_edges[0].xs, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(set_up_frame.big_edges[0].ys, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(set_up_frame.big_edges[1].xs, [6.0, 7.0, 8.0, 9.0, 10.0])
    assert np.allclose(set_up_frame.big_edges[1].ys, [6.0, 7.0, 8.0, 9.0, 10.0])

def test_filter_edges_invalid_method(set_up_frame):
    with pytest.raises(ValueError):
        set_up_frame.filter_edges(method="invalid")