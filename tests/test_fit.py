import pytest
import os 

import numpy as np
import forsys as fs


@pytest.fixture
def intial_furrow():
    frames = {}
    surfaceEvolver = fs.surface_evolver.SurfaceEvolver(os.path.join("tests", "data", 
                                                        "initial_furrow.dmp"))
    vertices, edges, cells = surfaceEvolver.create_lattice()
    frames[0] = fs.frames.Frame(vertices,
                                edges, 
                                cells, 
                                time=0, 
                                gt=surfaceEvolver.get_edges(), 
                                surface_evolver=False)
    forsys = fs.ForSys(frames, cm=False)
    forsys.solve(0, 'ext', metadata={'area_target': 'ave'})
    yield forsys

@pytest.fixture
def last_furrow():
    frames = {}
    surfaceEvolver = fs.surface_evolver.SurfaceEvolver(os.path.join("tests", "data", 
                                                        "last_furrow.dmp"))
    vertices, edges, cells = surfaceEvolver.create_lattice()
    frames[0] = fs.frames.Frame(vertices,
                                edges, 
                                cells, 
                                time=0, 
                                gt=surfaceEvolver.get_edges(), 
                                surface_evolver=False)
    forsys = fs.ForSys(frames, cm=False)
    forsys.solve(0, 'ext', metadata={'area_target': 'ave'})
    yield forsys

def test_fit_initial(intial_furrow):
    def distance_to_yeqx(gt, values):
        return 1 - sum([abs(gt[ii] - values[ii]) 
                    for ii in range(0, len(gt))]) / len(values)
    tensions_df = intial_furrow.frames[0].get_tensions()

    r_value = distance_to_yeqx(tensions_df['gt'].values, tensions_df['tension'].values)
    print("Test R-value: ", r_value)
    assert 1 > r_value > 0.94

def test_fit_furrow(last_furrow):
    def distance_to_yeqx(gt, values):
        return 1 - sum([abs(gt[ii] - values[ii]) 
                    for ii in range(0, len(gt))]) / len(values)
    tensions_df = last_furrow.frames[0].get_tensions()

    r_value = distance_to_yeqx(tensions_df['gt'].values / tensions_df['gt'].mean(), 
                                tensions_df['tension'].values)
    print("Test R-value: ", r_value)
    assert 1 > r_value > 0.98
