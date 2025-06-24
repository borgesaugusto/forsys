import os
import json

import forsys as fs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def create_folders(folder_to_create):
    if not os.path.exists(folder_to_create):
        os.makedirs(os.path.join(folder_to_create, "connections"))
        os.makedirs(os.path.join(folder_to_create, "myosin"))
        for folder_name in ["static", "dynamic"]:
            os.makedirs(os.path.join(folder_to_create,
                                     folder_name,
                                     "csvs"))
            os.makedirs(os.path.join(folder_to_create,
                                     folder_name,
                                     "fit_per_time"))
            os.makedirs(os.path.join(folder_to_create,
                                     folder_name,
                                     "tissues"))
            os.makedirs(os.path.join(folder_to_create,
                                     folder_name,
                                     "forces"))
            os.makedirs(os.path.join(folder_to_create,
                                     folder_name,
                                     "pressures"))
            os.makedirs(os.path.join(folder_to_create,
                                     folder_name,
                                     "stress_tensor"))
            
def create_folders_sweep(folder_to_create):
    if not os.path.exists(folder_to_create):
        os.makedirs(os.path.join(folder_to_create, "connections"))
        os.makedirs(os.path.join(folder_to_create, "myosin"))
        os.makedirs(os.path.join(folder_to_create, "fit_per_time"))
        os.makedirs(os.path.join(folder_to_create, "csvs"))

def create_directory(name, upperFolder):
    directory = os.path.join(upperFolder, name)
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_initial_guess(guess_file, min_time, max_time):
    try:
        with open(guess_file) as jfile:
            initial_guess = json.load(jfile)
            initial_guess = {int(k): {int(kin): vin for kin, vin in v.items()} for k, v in initial_guess.items()}
    except (FileNotFoundError, TypeError):
        initial_guess = {}
        print("No guess file, using zero guess")
    number_of_frames = max_time - min_time

    initial_guess = {k: {} for k in range(number_of_frames)
                     if k not in initial_guess.keys()} | initial_guess
    return initial_guess


def create_plots(frame_number, forsys, res_folder, myo=False, pressure=True, compress=1):
    vertices = forsys.frames[frame_number].vertices
    edges = forsys.frames[frame_number].edges
    cells = forsys.frames[frame_number].cells
    # mesh
    fs.plot.plot_mesh(vertices,
                        edges,
                        cells,
                        f"mesh_{frame_number}.png",
                        f"{res_folder}/tissues/",
                        mirror_y=True)
    # stresses
    fs.plot.plot_inference(forsys.frames[frame_number],
                            step=frame_number,
                            folder=os.path.join(res_folder, "forces"),
                            normalized="absolute",
                            mirror_y=False,
                            colorbar=False,
                            compress_scale=compress)
    print("Saving to ", os.path.join(res_folder, "forces", f"{frame_number}.png"))
    plt.savefig(os.path.join(res_folder, "forces", f"{frame_number}.png"), dpi=350)
    plt.close()
    if myo:
        # myosin
        print("Plotting myosin")
        fs.plot.plot_inference(forsys.frames[frame_number],
                               ground_truth=True,
                                step=frame_number,
                                folder=os.path.join(res_folder, "forces"),
                                normalized="absolute",
                                mirror_y=False,
                                colorbar=False)
        plt.savefig(os.path.join(res_folder, "myosin", f"{frame_number}.png"), dpi=350)
        plt.close()
    
    if pressure:
        fs.plot.plot_inference(forsys.frames[frame_number],
                                step=frame_number,
                                pressure=True,
                                folder=os.path.join(res_folder, "pressures"),
                                normalized="max",
                                mirror_y=False,
                                colorbar=False)
        plt.savefig(os.path.join(res_folder, "pressures", f"{frame_number}.png"), dpi=350)
        plt.close()
    
        fs.plot.plot_stress_tensor(forsys.frames[frame_number], 
                        os.path.join(res_folder, "stress_tensor"), 
                        frame_number, 
                        grid=12, 
                        radius=5,
                        tensor_scale=1.5)


def create_csvs(frame) -> tuple:
    tensions = []
    lengths = []
    positions_x = []
    positions_y = []
    be_ids = []
    curvatures = []

    cell_ids = []
    areas = []
    perimeters = []
    cell_posx = []
    cell_posy = []
    pressures = []

    for cellid, cell in frame.cells.items():
        cell_ids.append(cellid)
        areas.append(abs(cell.get_area()))
        perimeters.append(cell.get_perimeter())
        cell_posx.append(cell.get_cm()[0])
        cell_posy.append(cell.get_cm()[1])
        pressures.append(cell.pressure)

    cell_df = pd.DataFrame({
           "id": cell_ids,
           "area": areas,
           "perimeter": perimeters,
           "position_x": cell_posx,
           "position_y": cell_posy,
           "pressure": pressures,
    })

    for beid, big_edge in frame.big_edges.items():
        tensions.append(big_edge.tension)
        lengths.append(big_edge.get_length())
        positions_x.append(np.mean(big_edge.xs))
        positions_y.append(np.mean(big_edge.ys))
        be_ids.append(big_edge.big_edge_id)
        curvatures.append(big_edge.calculate_total_curvature())

    force_df = pd.DataFrame({
        "id": be_ids,
        "tension": tensions,
        "length": lengths,
        "position_x": positions_x,
        "position_y": positions_y,
        "curvature": curvatures,
    })

    return cell_df, force_df
