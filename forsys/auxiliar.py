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

def create_csvs(forsys: fs.ForSys, time=None, with_mapping:bool = False) -> tuple:
    """
    Create simple DFs for vertices, cells and big edges for a given frame or \
    long DFs for all frames if no time was given.

    :forsys: Forsys object
    :type forsys: fs.Forsys
    :time: Unique time of the frame
    :type time: int
    :with_mapping: If True, add mapping into csv
    :type with_mapping: bool, optional
    """
    if time is None:
        cell_df, force_df, v_df = create_csvs_long(forsys, is_mapping=with_mapping)
        return cell_df, force_df, v_df
    else:
        cell_df_long, force_df_long, v_df_long =create_csvs_simple(forsys, time, is_mapping=with_mapping)
        return cell_df_long, force_df_long, v_df_long

def create_csvs_simple(forsys: fs.ForSys, time:int, is_mapping:bool = False):
    frame = forsys.frames[time]
    
    be_mapped_ids = []
    be_ids = []
    tensions = []
    lengths = []
    positions_x = []
    positions_y = []
    curvatures = []
    own_cells = []
    vertices = []

    cell_mapped_ids = []
    cell_ids = []
    areas = []
    perimeters = []
    cell_posx = []
    cell_posy = []
    pressures = []
    neighbors = []

    v_id = []
    x_arr = []
    y_arr = []
    v_cells=[]

    if is_mapping and time > 0:
        cells_map, edge_map = forsys.get_maps(time)

    for cellid, cell in frame.cells.items():
            if (is_mapping and time>0):
                if(cells_map[cellid]!=None):
                    cell_mapped_ids.append(int(cells_map[cellid]))
                else:
                    cell_mapped_ids.append(pd.NA)
            cell_ids.append(cellid)
            areas.append(abs(cell.get_area()))
            perimeters.append(cell.get_perimeter())
            cell_posx.append(cell.get_cm()[0])
            cell_posy.append(cell.get_cm()[1])
            neighbors.append(cell.neighbors)
            pressures.append(cell.pressure)

    if (is_mapping and time>0):
        cell_df = pd.DataFrame({
            "id_mapped": cell_mapped_ids,
            "id": cell_ids,
            "area": areas,
            "perimeter": perimeters,
            "position_x": cell_posx,
            "position_y": cell_posy,
            "neighbors": neighbors,
            "pressure": pressures,
        })
    else:
        cell_df = pd.DataFrame({
            "id": cell_ids,
            "area": areas,
            "perimeter": perimeters,
            "position_x": cell_posx,
            "position_y": cell_posy,
            "neighbors": neighbors,
            "pressure": pressures,
        })
    
    for _, big_edge in frame.big_edges.items():
            if (is_mapping and time>0):
                if(edge_map[big_edge.big_edge_id]!=None):
                    be_mapped_ids.append(int(edge_map[big_edge.big_edge_id]))
                else:
                    be_mapped_ids.append(pd.NA)
            be_ids.append(big_edge.big_edge_id)
            tensions.append(big_edge.tension)
            lengths.append(big_edge.get_length())
            positions_x.append(np.mean(big_edge.xs))
            positions_y.append(np.mean(big_edge.ys))
            own_cells.append(big_edge.own_cells)
            vertices.append([big_edge.vertices[0].id, big_edge.vertices[-1].id])
            curvatures.append(big_edge.calculate_total_curvature())

    if (is_mapping and time>0):
        force_df = pd.DataFrame({
            "id_mapped":be_mapped_ids,
            "id": be_ids,
            "tension": tensions,
            "length": lengths,
            "position_x": positions_x,
            "position_y": positions_y,
            "own_cells": own_cells,
            "vertices": vertices,
            "curvature": curvatures,
        })

    else:
        force_df = pd.DataFrame({
            "id": be_ids,
            "tension": tensions,
            "length": lengths,
            "position_x": positions_x,
            "position_y": positions_y,
            "own_cells": own_cells,
            "vertices": vertices,
            "curvature": curvatures,
        })

    for _, vertex in frame.vertices.items():
        v_id.append(vertex.id)
        x_arr.append(vertex.x)
        y_arr.append(vertex.y)
        v_cells.append(vertex.ownCells)

    v_df = pd.DataFrame({
        "id": v_id,
        "position_x": x_arr,
        "position_y": y_arr,
        "cells":v_cells,
    })

    return cell_df, force_df, v_df

def create_csvs_long(forsys: fs.ForSys, is_mapping:bool = False) -> tuple:
    """
    Create DFs in long format for vertices, cells and big edges for all frames.

    :forsys: ForSys object
    :type forsys: fs.ForSys
    """
    times = list(forsys.frames.keys())

    cell_df_long = None
    force_df_long = None
    v_df_long = None

    for t in times:
        cell_df, force_df, v_df = create_csvs_simple(
            forsys, t, is_mapping=is_mapping
        )

        if is_mapping:
            # id_prev
            if t == times[0]:
                cell_df.insert(0, "id_prev", pd.Series(pd.NA, index=cell_df.index, dtype="Int64"))
                force_df.insert(0,"id_prev", pd.Series(pd.NA, index=force_df.index, dtype="Int64"))

            else:
                cell_df.rename(columns={"id_mapped": "id_prev"}, inplace=True)
                force_df.rename(columns={"id_mapped": "id_prev"}, inplace=True)

            # id_next
            if t != times[-1]:
                cells_map, edge_map = forsys.get_maps(t + 1)

                cells_map = {v: k for k, v in cells_map.items()}
                edge_map = {v: int(k) for k, v in edge_map.items()}

                id_cell_next = cell_df["id"].map(cells_map).astype("Int64")
                id_force_next = force_df["id"].map(edge_map).astype("Int64")
            else:
                id_cell_next = pd.Series(pd.NA, index=cell_df.index, dtype="Int64")
                id_force_next = pd.Series(pd.NA, index=force_df.index, dtype="Int64")

            cell_df.insert(2, "id_next", id_cell_next)
            force_df.insert(2, "id_next", id_force_next)

        cell_df.insert(0, "time", t)
        force_df.insert(0, "time", t)
        v_df.insert(0, "time", t)

        # ---------- Append ----------
        if cell_df_long is None:
            cell_df_long = cell_df.copy()
            force_df_long = force_df.copy()
            v_df_long = v_df.copy()
        else:
            cell_df_long = pd.concat([cell_df_long, cell_df], ignore_index=True)
            force_df_long = pd.concat([force_df_long, force_df], ignore_index=True)
            v_df_long = pd.concat([v_df_long, v_df], ignore_index=True)

    return cell_df_long, force_df_long, v_df_long
