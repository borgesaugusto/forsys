import pytest
import os 

import numpy as np
import forsys as fs

import scipy.stats as sps


@pytest.fixture
def furrow():
    frames = {}
    for ii in range(0, 8):
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


def test_pressure_initial_furrow(furrow):
    furrow.build_force_matrix(when=0)
    furrow.solve_stress(when=0)

    furrow.build_pressure_matrix(when=0)
    furrow.solve_pressure(when=0, method="lagrange_pressure")

    pressures_df = furrow.frames[0].get_pressures()
    pressures_df["gt_pressure"] = (pressures_df["gt_pressure"] - pressures_df["gt_pressure"].mean()) / pressures_df["gt_pressure"].std()
    pressures_df["pressure"] = (pressures_df["pressure"] - pressures_df["pressure"].mean()) / pressures_df["pressure"].std()
    
    correlation = sps.pearsonr(pressures_df['gt_pressure'], pressures_df['pressure'])[0]

    assert correlation > 0.99


def test_pressure_last_furrow(furrow):
    last_frame = len(furrow.frames) - 1
    furrow.build_force_matrix(when=last_frame, circle_fit_method="dlite")
    furrow.solve_stress(when=last_frame)

    furrow.build_pressure_matrix(when=last_frame)
    furrow.solve_pressure(when=last_frame, method="lagrange_pressure")

    pressures_df = furrow.frames[last_frame].get_pressures()
    pressures_df["gt_pressure"] = (pressures_df["gt_pressure"] - pressures_df["gt_pressure"].mean()) / pressures_df["gt_pressure"].std()
    pressures_df["pressure"] = (pressures_df["pressure"] - pressures_df["pressure"].mean()) / pressures_df["pressure"].std()
    
    correlation = sps.pearsonr(pressures_df['gt_pressure'], pressures_df['pressure'])[0]

    assert correlation > 0.99


def test_pressure_all_frames_furrow(furrow):
    for ii in range(0, 8):
        furrow.build_force_matrix(when=ii, circle_fit_method="dlite")
        furrow.solve_stress(when=ii)

        furrow.build_pressure_matrix(when=ii)
        furrow.solve_pressure(when=ii, method="lagrange_pressure")

        pressures_df = furrow.frames[ii].get_pressures()
        pressures_df["gt_pressure"] = (pressures_df["gt_pressure"] - pressures_df["gt_pressure"].mean()) / pressures_df["gt_pressure"].std()
        pressures_df["pressure"] = (pressures_df["pressure"] - pressures_df["pressure"].mean()) / pressures_df["pressure"].std()
        
        correlation = sps.pearsonr(pressures_df['gt_pressure'], pressures_df['pressure'])[0]
        
        assert correlation > 0.99