import pytest
import numpy as np
import pandas as pd
import copy
import os
import forsys as fs

@pytest.fixture
def set_up_frame():
    vertices = {}
    edges = {}
    cells = {}
    frame = fs.frames.Frame(0, vertices, edges, cells)


    for vid in range(10):
        vertices[vid] = fs.vertex.Vertex(vid, vid + 1 , vid + 1)
        if vid > 0:
            edges[vid - 1] = fs.edge.SmallEdge(vid - 1, vertices[vid - 1], vertices[vid])

    frame.big_edges = {
        0: fs.edge.BigEdge(0,
                   vertices=[vertices[0], vertices[1], vertices[2], vertices[3], vertices[4]]),
        1: fs.edge.BigEdge(1, 
                   vertices=[vertices[5],
                             vertices[6],
                             vertices[7],
                             vertices[8],
                             vertices[9]])
    }
    return frame


@pytest.fixture
def furrow():
    frames = {}
    for ii in range(0, 2):
        surfaceEvolver = fs.surface_evolver.SurfaceEvolver(os.path.join(
                                                            "tests", 
                                                            "data", 
                                                            "furrow_gauss_velocity", 
                                                            f"stage{ii}.dmp"))
        frames[ii] = fs.frames.Frame(ii,
                                    surfaceEvolver.vertices,
                                    surfaceEvolver.edges, 
                                    surfaceEvolver.cells, 
                                    time=ii, 
                                    gt=True)
    forsys = fs.ForSys(frames, cm=False)
    yield forsys

@pytest.fixture
def set_up_frame_with_small_edge():
    vertices = {}
    edges = {}
    cells = {}
    frame = fs.frames.Frame(0, vertices, edges, cells)

    vertices[0] = fs.vertex.Vertex(0, 0, 1)
    vertices[1] = fs.vertex.Vertex(1, 1, 2)
    vertices[2] = fs.vertex.Vertex(2, 2, 3)

    edges[0] = fs.edge.SmallEdge(0, vertices[0], vertices[1])
    edges[1] = fs.edge.SmallEdge(1, vertices[1], vertices[2])


    frame.big_edges = {
        0: fs.edge.BigEdge(0,
                    vertices=[vertices[0], vertices[1]])
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


def test_filter_edges_invalid_frame(set_up_frame_with_small_edge):
    initial_big_edges = copy.deepcopy(set_up_frame_with_small_edge.big_edges)
    set_up_frame_with_small_edge.filter_edges(method="SG")
    assert set_up_frame_with_small_edge.big_edges == initial_big_edges


def test_get_cell_properties_df(furrow):
    furrow.build_force_matrix(when=0)
    furrow.solve_stress(when=0)

    furrow.build_pressure_matrix(when=0)
    furrow.solve_pressure(when=0, method="lagrange_pressure")

    df = furrow.frames[0].get_cell_properties_df(center_method="centroid")
    assert isinstance(df, pd.DataFrame)
    assert "cid" in df.columns and df["cid"].dtype == int
    assert "centerx" in df.columns and df["centerx"].dtype == float
    assert "centery" in df.columns  and df["centery"].dtype == float
    assert "perimeter" in df.columns and df["perimeter"].dtype == float
    assert "area" in df.columns and df["area"].dtype == float
    assert "pressure" in df.columns and df["pressure"].dtype == float
    assert len(df) == len(furrow.frames[0].cells)

def test_get_edges_props_df(furrow):
    furrow.build_force_matrix(when=0)
    furrow.solve_stress(when=0)

    df = furrow.frames[0].get_edges_props_df()
    assert isinstance(df, pd.DataFrame)
    assert "eid" in df.columns and df["eid"].dtype == int
    assert "posx" in df.columns and df["posx"].dtype == float
    assert "posy" in df.columns  and df["posy"].dtype == float
    assert "tension" in df.columns and df["tension"].dtype == float
    assert "gt_tension" in df.columns and df["gt_tension"].dtype == float
    assert len(df) == len(furrow.frames[0].big_edges)