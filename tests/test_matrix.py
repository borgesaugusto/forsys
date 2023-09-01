import pytest
import os 

import numpy as np
import forsys as fs

@pytest.fixture
def extra_cell():
    frames = {}
    skeleton = fs.skeleton.Skeleton(os.path.join("tests", 
                                                 "data", 
                                                 "test_nonzero.tif"))
    vertices, edges, cells = skeleton.create_lattice()
    vertices, edges, cells, _ = fs.virtual_edges.generate_mesh(vertices, 
                                                                edges, 
                                                                cells, 
                                                                ne=6)
    frames[0] = fs.frames.Frame(0,
                                vertices,
                                edges, 
                                cells, 
                                time=0)
    forsys = fs.ForSys(frames, cm=False)

    yield forsys

def test_fmatrix_three_non_zero(extra_cell):
    extra_cell.build_force_matrix()
    extra_cell.solve_stress()

    assert not np.any(extra_cell.frames[0].forces == 0)