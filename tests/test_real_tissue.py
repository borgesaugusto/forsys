import pytest
import os 
import pandas as pd
import numpy as np
import forsys as fs

from sklearn.metrics import mean_absolute_percentage_error, r2_score

@pytest.fixture
def experiment_1():
    frames = {}
    skeleton = fs.skeleton.Skeleton(os.path.join("tests", 
                                                 "data", 
                                                 "experimental", 
                                                 f"exp_1.tif"), mirror_y=False)
    vertices, edges, cells = skeleton.create_lattice()
    vertices, edges, cells, _ = fs.virtual_edges.generate_mesh(vertices, edges, cells, ne=6)
    frames[0] = fs.frames.Frame(0, vertices, edges, cells, time=0)
    frames[0].filter_edges(method="SG")
    forsys = fs.ForSys(frames, cm=False, initial_guess={})

    df = pd.read_csv(os.path.join("tests", "data", "experimental", "exp_1.csv"))
    
    yield (forsys, df)

def test_myosin_correlation(experiment_1):
    experiment = experiment_1[0]
    df = experiment_1[1]

    experiment.build_force_matrix(when=0,
                            metadata={"ignore_four": False},
                            angle_limit=np.inf,
                            circle_fit_method="dlite")
    experiment.solve_stress(when=0, 
                            b_matrix="static",
                            velocity_normalization=0,
                            adimensional_velocity=False,
                            method="lsq",
                            use_std=False,
                            allow_negatives=False)

    experiment.build_force_matrix(when=0)
    experiment.solve_stress(when=0, allow_negatives=False)

    tensions_df = experiment.frames[0].get_tensions()
    rounded_tensions = tensions_df["stress"].round(2)
    rounded_df = df["stress"].round(2)
    assert np.all(rounded_tensions.values == rounded_df.values)