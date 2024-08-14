import forsys as fs
import pytest
import os

DMP_FOLDER_TO_TEST = os.path.join("tests",
                                  "data",
                                  "12_12")

def test_bench_reading_evolver(benchmark):
    path = os.path.join(DMP_FOLDER_TO_TEST,
                        f"step_{24}.dmp")
    benchmark(fs.surface_evolver.SurfaceEvolver, path)
            

def test_bench_creating_frames(benchmark):
    path = os.path.join(DMP_FOLDER_TO_TEST, 
                        f"step_{24}.dmp")
    surfaceEvolver = fs.surface_evolver.SurfaceEvolver(path)
    params = [0,
              surfaceEvolver.vertices,
              surfaceEvolver.edges,
              surfaceEvolver.cells,              
              0,
              True]
    benchmark(fs.frames.Frame, *params)


def test_bench_creating_forsys(benchmark):
    frames = {}
    path = os.path.join(DMP_FOLDER_TO_TEST, 
                        f"step_{24}.dmp")
    surfaceEvolver = fs.surface_evolver.SurfaceEvolver(path)
    params = [7,
              surfaceEvolver.vertices,
              surfaceEvolver.edges,
              surfaceEvolver.cells,              
              7,
              True]
    frames[0] = fs.frames.Frame(*params)
    benchmark(fs.ForSys, frames)


def test_bench_building_force_matrix_dlitefit(benchmark):
    frames = {}
    path = os.path.join(DMP_FOLDER_TO_TEST, 
                        f"step_{24}.dmp")
    surfaceEvolver = fs.surface_evolver.SurfaceEvolver(path)
    params = [0,
              surfaceEvolver.vertices,
              surfaceEvolver.edges,
              surfaceEvolver.cells,              
              0,
              True]
    frames[0] = fs.frames.Frame(*params)
    forsys = fs.ForSys(frames)
    benchmark(forsys.build_force_matrix, 0, circle_fit_method="dlite")
    

def test_bench_building_force_matrix_taubinfit(benchmark):
    frames = {}
    path = os.path.join(DMP_FOLDER_TO_TEST, 
                        f"step_{24}.dmp")
    surfaceEvolver = fs.surface_evolver.SurfaceEvolver(path)
    params = [0,
              surfaceEvolver.vertices,
              surfaceEvolver.edges,
              surfaceEvolver.cells,              
              0,
              True]
    frames[0] = fs.frames.Frame(*params)
    forsys = fs.ForSys(frames)
    benchmark(forsys.build_force_matrix, 0, circle_fit_method="taubinSVD")


def test_bench_solving_force_matrix_nnls(benchmark):
    frames = {}
    path = os.path.join(DMP_FOLDER_TO_TEST, 
                        f"step_{24}.dmp")
    surfaceEvolver = fs.surface_evolver.SurfaceEvolver(path)
    params = [0,
              surfaceEvolver.vertices,
              surfaceEvolver.edges,
              surfaceEvolver.cells,              
              0,
              True]
    frames[0] = fs.frames.Frame(*params)
    forsys = fs.ForSys(frames)
    forsys.build_force_matrix(0)
    benchmark(forsys.solve_stress, 0, method="nnls")

@pytest.mark.skip(reason="LSQ testing is too random to be reliably benchmarked")
def test_bench_solving_force_matrix_lsq(benchmark):
    frames = {}
    path = os.path.join(DMP_FOLDER_TO_TEST, 
                        f"step_{24}.dmp")
    surfaceEvolver = fs.surface_evolver.SurfaceEvolver(path)
    params = [0,
              surfaceEvolver.vertices,
              surfaceEvolver.edges,
              surfaceEvolver.cells,              
              0,
              True]
    frames[0] = fs.frames.Frame(*params)
    forsys = fs.ForSys(frames)
    forsys.build_force_matrix(0)
    benchmark(forsys.solve_stress, 0, method="lsq")


@pytest.mark.skip(reason="NNLS testing is too random to be reliably benchmarked")
def test_bench_solving_force_matrix_nnls_velocity(benchmark):
    frames = {}
    for ii in range(0, 5):
        path = os.path.join(DMP_FOLDER_TO_TEST, 
                            f"step_{ii + 20}.dmp")
        surfaceEvolver = fs.surface_evolver.SurfaceEvolver(path)
        params = [ii,
                surfaceEvolver.vertices,
                surfaceEvolver.edges,
                surfaceEvolver.cells,              
                ii,
                True]
        frames[ii] = fs.frames.Frame(*params)
    forsys = fs.ForSys(frames)
    forsys.build_force_matrix(4, b_matrix="velocity")
    benchmark(forsys.solve_stress, 4, method="nnls")


def test_bench_read_skeleton(benchmark):
    path = os.path.join("tests", 
                        "data", 
                        "experimental", 
                        f"exp_1.tif")
    benchmark(fs.skeleton.Skeleton, path, mirror_y=False)


def test_bench_create_lattice(benchmark):
    path = os.path.join("tests", 
                        "data", 
                        "experimental", 
                        f"exp_1.tif")
    skeleton = fs.skeleton.Skeleton(path, mirror_y=False)
    benchmark(skeleton.create_lattice)


def test_bench_generate_mesh(benchmark):
    path = os.path.join("tests", 
                        "data", 
                        "experimental", 
                        f"exp_1.tif")
    skeleton = fs.skeleton.Skeleton(path, mirror_y=False)
    vertices, edges, cells = skeleton.create_lattice()
    benchmark(fs.virtual_edges.generate_mesh, vertices, edges, cells, ne=6)


def test_bench_building_pressure_matrix(benchmark):
    frames = {}
    path = os.path.join(DMP_FOLDER_TO_TEST, 
                        f"step_{24}.dmp")
    surfaceEvolver = fs.surface_evolver.SurfaceEvolver(path)
    params = [0,
              surfaceEvolver.vertices,
              surfaceEvolver.edges,
              surfaceEvolver.cells,              
              0,
              True]
    frames[0] = fs.frames.Frame(*params)
    forsys = fs.ForSys(frames)
    forsys.build_force_matrix(0)
    forsys.solve_stress(0, method="nnls")
    benchmark(forsys.build_pressure_matrix, 0)

@pytest.mark.skip(reason="NNLS testing is too random to be reliably benchmarked")
def test_bench_solving_pressure_matrix(benchmark):
    frames = {}
    path = os.path.join(DMP_FOLDER_TO_TEST, 
                        f"step_{24}.dmp")
    surfaceEvolver = fs.surface_evolver.SurfaceEvolver(path)
    params = [0,
              surfaceEvolver.vertices,
              surfaceEvolver.edges,
              surfaceEvolver.cells,              
              0,
              False]
    frames[0] = fs.frames.Frame(*params)
    forsys = fs.ForSys(frames)
    forsys.build_force_matrix(0)
    forsys.solve_stress(0, method="nnls", allow_negatives=False)
    forsys.build_pressure_matrix(0)
    benchmark(forsys.solve_pressure, 0, method="lagrange_pressure")


def test_bench_plot_mesh(benchmark):
    frames = {}
    path = os.path.join(DMP_FOLDER_TO_TEST, 
                        f"step_{24}.dmp")
    surfaceEvolver = fs.surface_evolver.SurfaceEvolver(path)
    params = [0,
              surfaceEvolver.vertices,
              surfaceEvolver.edges,
              surfaceEvolver.cells,              
              0,
              True]
    frames[0] = fs.frames.Frame(*params)
    forsys = fs.ForSys(frames)
    params = [forsys.frames[0].vertices,
              forsys.frames[0].edges,
              forsys.frames[0].cells]
    benchmark(fs.plot.plot_mesh, *params)

def test_bench_plot_inference(benchmark):
    frames = {}
    path = os.path.join(DMP_FOLDER_TO_TEST, 
                        f"step_{24}.dmp")
    surfaceEvolver = fs.surface_evolver.SurfaceEvolver(path)
    params = [0,
              surfaceEvolver.vertices,
              surfaceEvolver.edges,
              surfaceEvolver.cells,              
              0,
              True]
    frames[0] = fs.frames.Frame(*params)
    forsys = fs.ForSys(frames)
    forsys.build_force_matrix(0)
    forsys.solve_stress(0, method="nnls", allow_negatives=False)
    forsys.build_pressure_matrix(0)
    forsys.solve_pressure(0, method="lagrange_pressure")
    benchmark(fs.plot.plot_inference, forsys.frames[0], pressure=True)