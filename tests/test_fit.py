import pytest
import os 

import numpy as np
import forsys as fs
import warnings
from sklearn.metrics import mean_absolute_percentage_error, r2_score

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

def test_fit_initial_furrow(furrow):
    furrow.build_force_matrix(when=0)
    furrow.solve_stress(when=0)

    tensions_df = furrow.frames[0].get_tensions()

    r_value = r2_score(tensions_df['gt'].values / tensions_df['gt'].mean(), 
                                tensions_df['stress'].values)
    print("initial furrow", r_value)
    assert 1 > r_value > 0.94


def test_fit_last_furrow(furrow):
    last_frame = len(furrow.frames) - 1
    furrow.build_force_matrix(when=last_frame)
    furrow.solve_stress(when=last_frame)
    
    tensions_df = furrow.frames[last_frame].get_tensions()

    r_value = r2_score(tensions_df['gt'].values / tensions_df['gt'].mean(), 
                                tensions_df['stress'].values)
    print("Final furrow", r_value)
    assert 1 > r_value > 0.93


def test_fit_last_furrow_taubin(furrow):
    last_frame = len(furrow.frames) - 1
    furrow.build_force_matrix(when=last_frame, circle_fit_method="taubinSVD")
    furrow.solve_stress(when=last_frame)
    
    tensions_df = furrow.frames[last_frame].get_tensions()

    r_value = r2_score(tensions_df['gt'].values / tensions_df['gt'].mean(), 
                                tensions_df['stress'].values)
    print("Final furrow", r_value)
    assert 1 > r_value > 0.93


def test_fit_furrow_movement_velocity(furrow):
    all_r_values = []
    for ii in furrow.frames.keys():
        furrow.build_force_matrix(when=ii)
        furrow.solve_stress(when=ii, b_matrix="velocity")
        tensions_df = furrow.frames[ii].get_tensions()

        arguments = [tensions_df['gt'].values / tensions_df['gt'].mean(), 
                                    tensions_df['stress'].values / tensions_df['stress'].mean()]
        r_value = r2_score(*arguments)
        print(f"t ={ii} with MAPE {mean_absolute_percentage_error(*arguments)}")
        all_r_values.append(r_value)
    print("Dynamic furrow", all_r_values)
    assert np.all([1 > value > 0.94 for value in all_r_values])

def test_fit_furrow_movement_velocity_taubin(furrow):
    all_r_values = []
    for ii in furrow.frames.keys():
        furrow.build_force_matrix(when=ii, circle_fit_method="taubinSVD")
        furrow.solve_stress(when=ii, b_matrix="velocity")
        tensions_df = furrow.frames[ii].get_tensions()

        arguments = [tensions_df['gt'].values / tensions_df['gt'].mean(), 
                                    tensions_df['stress'].values / tensions_df['stress'].mean()]
        r_value = r2_score(*arguments)
        print(f"t ={ii} with MAPE {mean_absolute_percentage_error(*arguments)}")
        all_r_values.append(r_value)
    print("Dynamic furrow", all_r_values)
    assert np.all([1 > value > 0.94 for value in all_r_values])

def test_fit_furrow_velocity_lsq_initial_last_val(furrow, recwarn):
    all_r_values = []
    for ii in furrow.frames.keys():
        if ii == 0:
            initial_condition = np.ones(len(furrow.frames[0].internal_big_edges))
        else:
            initial_condition = furrow.frames[ii-1].get_tensions()["stress"]

        furrow.build_force_matrix(when=ii)
        furrow.solve_stress(when=ii,
                            b_matrix="velocity",
                            initial_condition=initial_condition,
                            method="lsq")            
            
        tensions_df = furrow.frames[ii].get_tensions()
        r_value = r2_score(tensions_df['gt'].values / tensions_df['gt'].mean(), 
                                    tensions_df['stress'].values / tensions_df['stress'].mean())
        all_r_values.append(r_value)
    print("Dynamic furrow using last", all_r_values)
    for r in recwarn:
        assert "Numerically" not in str(r.message), "Warnings raised while solving, probably LSQ went to numerical"
    assert np.all([1 > value > 0.92 for value in all_r_values])


def test_fit_furrow_velocity_lsq_initial_gt(furrow, recwarn):
    all_r_values = []
    for ii in furrow.frames.keys():
        initial_condition = list(furrow.frames[ii].get_gt_tensions()["gt"].values / furrow.frames[ii].get_gt_tensions()["gt"].mean())

        furrow.build_force_matrix(when=ii)
        furrow.solve_stress(when=ii,
                            b_matrix="velocity",
                            initial_condition=initial_condition,
                            method="lsq")
        tensions_df = furrow.frames[ii].get_tensions()
        r_value = r2_score(tensions_df['gt'].values / tensions_df['gt'].mean(), 
                                    tensions_df['stress'].values / tensions_df['stress'].mean())
        all_r_values.append(r_value)
    print("Dynamic furrow using GT", all_r_values)
    print(tensions_df)
    for r in recwarn:
        assert "Numerically" not in str(r.message), "Warnings raised while solving, probably LSQ went to numerical"
    assert np.all([1 > value > 0.94 for value in all_r_values])

def test_fit_furrow_velocity_lsq_initial_one(furrow, recwarn):
    all_r_values = []
    for ii in furrow.frames.keys():
        initial_condition = np.ones(len(furrow.frames[ii].get_gt_tensions()["gt"]))
        # initial_condition = np.abs(np.random.normal(1, 0.1, len(furrow.frames[ii].get_gt_tensions()["gt"])))
        

        furrow.build_force_matrix(when=ii)
        furrow.solve_stress(when=ii,
                            b_matrix="velocity",
                            initial_condition=initial_condition,
                            method="lsq")
        tensions_df = furrow.frames[ii].get_tensions()
        # r_value = distance_to_yeqx(tensions_df['gt'].values / tensions_df['gt'].mean(), 
        #                             tensions_df['stress'].values / tensions_df['stress'].mean())
        arguments = [tensions_df['gt'].values / tensions_df['gt'].mean(), 
                                    tensions_df['stress'].values / tensions_df['stress'].mean()]
        r_value = r2_score(*arguments)
        print(f"t ={ii} with MAPE {mean_absolute_percentage_error(*arguments)}")
        all_r_values.append(r_value)
    print("Dynamic furrow using Ones", all_r_values)
    print(tensions_df)
    for r in recwarn:
        assert "Numerically" not in str(r.message), "Warnings raised while solving, probably LSQ went to numerical"
    assert np.all([1 > value > 0.95 for value in all_r_values])

@pytest.mark.skip(reason="Eliminating edges doesn't work with this normalization ")
def test_fit_furrow_velocity_lsq_initial_gt_deleting10(furrow, recwarn):
    all_r_values = []
    angle_limit = 180 * (1 - 0.1) * np.pi / 180
    for ii in furrow.frames.keys():
        initial_condition = list(furrow.frames[ii].get_gt_tensions()["gt"].values / furrow.frames[ii].get_gt_tensions()["gt"].mean())

        furrow.build_force_matrix(when=ii,
                                  angle_limit=angle_limit)
        furrow.solve_stress(when=ii,
                            b_matrix="velocity",
                            initial_condition=initial_condition,
                            method="lsq")
        tensions_df = furrow.frames[ii].get_tensions()
        
        arguments = [tensions_df['gt'].values / tensions_df['gt'].mean(), 
                                    tensions_df['stress'].values / tensions_df['stress'].mean()]
        r_value = r2_score(*arguments)
        all_r_values.append(r_value)
        print(f"t ={ii} with MAPE {mean_absolute_percentage_error(*arguments)}")
    print("Dynamic furrow using GT with 10% discard", all_r_values)
    for r in recwarn:
        assert "Numerically" not in str(r.message), "Warnings raised while solving, probably LSQ went to numerical"
    assert np.all([1 > value > 0.94 for value in all_r_values])


def test_fit_furrow_velocity_lsq_initial_one_std(furrow, recwarn):
    all_r_values = []
    for ii in furrow.frames.keys():
        initial_condition = np.ones(len(furrow.frames[ii].get_gt_tensions()["gt"]))
        # initial_condition = np.abs(np.random.normal(1, 0.1, len(furrow.frames[ii].get_gt_tensions()["gt"])))
        

        furrow.build_force_matrix(when=ii)
        furrow.solve_stress(when=ii,
                            b_matrix="velocity",
                            initial_condition=initial_condition,
                            method="lsq",
                            use_std=True)
        tensions_df = furrow.frames[ii].get_tensions()
        # r_value = distance_to_yeqx(tensions_df['gt'].values / tensions_df['gt'].mean(), 
        #                             tensions_df['stress'].values / tensions_df['stress'].mean())
        arguments = [tensions_df['gt'].values / tensions_df['gt'].mean(), 
                                    tensions_df['stress'].values / tensions_df['stress'].mean()]
        r_value = r2_score(*arguments)
        print(f"t ={ii} with MAPE {mean_absolute_percentage_error(*arguments)}")
        all_r_values.append(r_value)
    print("Dynamic furrow using Ones", all_r_values)
    print(tensions_df)
    for r in recwarn:
        assert "Numerically" not in str(r.message), "Warnings raised while solving, probably LSQ went to numerical"
    assert np.all([1 > value > 0.95 for value in all_r_values])
