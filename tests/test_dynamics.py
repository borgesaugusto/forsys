import pytest

import numpy as np
import forsys as fs


@pytest.fixture
def set_up_mesh():
    frames = {}
    vertices = {}
    edges = {}
    cells = {}

    vertices[0] = fs.vertex.Vertex(0, 0, 1)
    vertices[1] = fs.vertex.Vertex(1, 0, 1)
    edges[0] = fs.edge.SmallEdge(0, vertices[0], vertices[1])
    cells[0] = fs.cell.Cell(0, vertices.values())
    frames[0] = fs.frames.Frame(0, vertices, edges, cells, time=0)

    vertices[0] = fs.vertex.Vertex(0, 4, 4)
    vertices[1] = fs.vertex.Vertex(1, 0, 1)
    edges[0] = fs.edge.SmallEdge(0, vertices[0], vertices[1])
    cells[0] = fs.cell.Cell(0, vertices.values())
    frames[1] = fs.frames.Frame(1, vertices, edges, cells, time=1)

    vertices[0] = fs.vertex.Vertex(0, 12, 11)
    vertices[1] = fs.vertex.Vertex(1, 0, 1)
    edges[0] = fs.edge.SmallEdge(0, vertices[0], vertices[1])
    cells[0] =  fs.cell.Cell(0, vertices.values())
    frames[2] = fs.frames.Frame(2, vertices, edges, cells, time=2)
    
    vertices[0] = fs.vertex.Vertex(0, 18, 22)
    vertices[1] = fs.vertex.Vertex(1, 0, 1)
    edges[0] = fs.edge.SmallEdge(0, vertices[0], vertices[1])
    cells[0] = fs.cell.Cell(0, vertices.values())
    frames[3] = fs.frames.Frame(3, vertices, edges, cells, time=3)

    forsys = fs.ForSys(frames, cm=False)
    forsys.mesh.mapping = {0: {0: 0, 1:1}, 2: {0: 0, 1:1}, 1: {0: 0, 1:1}}

    yield forsys.mesh
    
    del forsys
    del vertices
    del edges
    del cells

@pytest.mark.skip(reason="Cell neighbor calculation doesn't work with this test")
def test_acceleration(set_up_mesh):
    assert np.all(set_up_mesh.calculate_acceleration(0, 0) == [4., 4.])
    assert np.all(set_up_mesh.calculate_acceleration(0, 1) == [4., 4.])
    assert np.all(set_up_mesh.calculate_acceleration(0, 2) == [-2., 4.])
    assert np.all(set_up_mesh.calculate_acceleration(0, 3) == [-2., 4.])

@pytest.mark.skip(reason="Cell neighbor calculation doesn't work with this test")
def test_velocity(set_up_mesh):
    assert np.all(set_up_mesh.calculate_velocity(0, 0) == [4., 3.])
    assert np.all(set_up_mesh.calculate_velocity(0, 1) == [8., 7.])
    assert np.all(set_up_mesh.calculate_velocity(0, 2) == [6., 11.])
    assert np.all(set_up_mesh.calculate_velocity(0, 3) == [6., 11.])