import pytest
import os 

import numpy as np
import forsys as fs

def distance_to_yeqx(gt, values):
        return 1 - sum([abs(gt[ii] - values[ii]) 
                    for ii in range(0, len(gt))]) / len(values)

@pytest.fixture
def furrow():
    frames = {}
    for ii in range(0, 8):
        surfaceEvolver = fs.surface_evolver.SurfaceEvolver(os.path.join(
                                                            "tests", 
                                                            "data", 
                                                            "furrow_gauss_velocity", 
                                                            f"stage{ii}.dmp"))
        vertices, edges, cells = surfaceEvolver.create_lattice()
        frames[ii] = fs.frames.Frame(ii,
                                    vertices,
                                    edges, 
                                    cells, 
                                    time=ii, 
                                    gt=True)
    forsys = fs.ForSys(frames, cm=False)
    yield forsys

def test_fit_initial_furrow(furrow):
    furrow.build_force_matrix(when=0)
    furrow.solve_stress(when=0)

    tensions_df = furrow.frames[0].get_tensions()

    r_value = distance_to_yeqx(tensions_df['gt'].values / tensions_df['gt'].mean(), 
                                tensions_df['stress'].values)
    assert 1 > r_value > 0.94

def test_fit_last_furrow(furrow):
    last_frame = len(furrow.frames) - 1
    furrow.build_force_matrix(when=last_frame)
    furrow.solve_stress(when=last_frame)
    
    tensions_df = furrow.frames[last_frame].get_tensions()

    r_value = distance_to_yeqx(tensions_df['gt'].values / tensions_df['gt'].mean(), 
                                tensions_df['stress'].values)
    assert 1 > r_value > 0.93

def test_fit_furrow_movement_velocity(furrow):
    all_r_values = []
    for ii in furrow.frames.keys():
        furrow.build_force_matrix(when=ii)
        furrow.solve_stress(when=ii, b_matrix="velocity")
        tensions_df = furrow.frames[ii].get_tensions()
        r_value = distance_to_yeqx(tensions_df['gt'].values / tensions_df['gt'].mean(), 
                                    tensions_df['stress'].values)
        all_r_values.append(r_value)
    assert np.all([1 > value > 0.94 for value in all_r_values])
