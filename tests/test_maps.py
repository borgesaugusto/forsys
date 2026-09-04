import pytest
import numpy as np
import pandas as pd
import os
import forsys as fs
import matplotlib.pyplot as plt


@pytest.fixture
def furrow():
    frames = {}
    for ii in range(2):
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
    for ii in range(2):
        forsys.build_force_matrix(when=ii)
        forsys.solve_stress(when=ii)

        forsys.build_pressure_matrix(when=ii)
        forsys.solve_pressure(when=ii, method="lagrange_pressure")
    yield forsys

def test_mapped_cells(furrow):

    cells, edges = furrow.get_maps(key=1)
    assert cells[1] == 1
    assert cells[5] == 5
    assert cells[10] == 10
    assert cells[15] == 15
    assert cells[20] == 20
    assert cells[25] == 25
    assert cells[30] == 30
    assert cells[35] == 35

def test_mapped_edges(furrow):

    cells, edges = furrow.get_maps(key=1)
    assert edges[1] == 1
    assert edges[20] == 20
    assert edges[40] == 40
    assert edges[60] == 60
    assert edges[80] == 80
    assert edges[100] == 100
    assert edges[120] == 120
    assert edges[140] == 140