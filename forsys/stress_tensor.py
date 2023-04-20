import numpy as np
import pandas as pd

# from sympy import Matrix

# TODO Use map() to improve dfs creation

def area_term():
    # for cid, cells in cells.keys():
    pass

def get_cells_df(frame):
    cells = pd.DataFrame()
    cell_centers_x = [cell.get_cm()[0] for _, cell in frame.cells.items()]
    cell_centers_y = [cell.get_cm()[1] for _, cell in frame.cells.items()]
    cell_ids = [cid for cid, _ in frame.cells.items()]
    cell_areas = [abs(cell.get_area()) for _, cell in frame.cells.items()]
    cell_pressures = [cell.pressure for _, cell in frame.cells.items()]

    cells["ids"] = cell_ids
    cells["xcm"] = cell_centers_x
    cells["ycm"] = cell_centers_y
    cells["area"] = cell_areas
    cells["pressure"] = cell_pressures

    return cells

def get_big_edges_df(frame):
    bedges = pd.DataFrame()
    
    bedges_celli = []
    bedges_cellj = []

    bedges_ids = [big_edge.big_edge_id for _, big_edge in frame.big_edges.items()]
    bedges_stress = [big_edge.tension for _, big_edge in frame.big_edges.items()]
    bedges_vector= [big_edge.get_vector_from_vertex(big_edge.vertices[0].id) for _, big_edge in frame.big_edges.items()]
    for _, big_edge in frame.big_edges.items():
        bedges_celli.append(big_edge.own_cells[0])
        try:
            bedges_cellj.append(big_edge.own_cells[1])
        except IndexError:
            bedges_cellj.append(-1)


    bedges["ids"] = bedges_ids
    bedges["stress"] = bedges_stress
    bedges["vector"] = bedges_vector
    bedges["cell1"] = bedges_celli
    bedges["cell2"] = bedges_cellj

    return bedges



def membrane_term():
    pass

def stress_tensor(frame, grid=5, radius=1):
    sigmas = {}
    cells = get_cells_df(frame)
    big_edges = get_big_edges_df(frame)

    _, x_bins = np.histogram(cells["xcm"], grid)
    _, y_bins = np.histogram(cells["ycm"], grid)

    x_bin_centers = [(x_bins[ii] + x_bins[ii+1]) / 2 for ii in range(len(x_bins) -1)]
    y_bin_centers = [(y_bins[ii] + y_bins[ii+1]) / 2 for ii in range(len(y_bins) -1)]
    bins_centers = [x_bin_centers, y_bin_centers]

    # min_distance = radius * np.mean([x_bins[1] - x_bins[0], y_bins[1] - y_bins[0]])
    min_distance = radius * np.sqrt(cells["area"].mean() / np.pi)

    for row in range(grid):
        for column in range(grid):
            center = ((x_bins[row + 1] + x_bins[row]) / 2, (y_bins[column + 1] + y_bins[column]) / 2) 
            current_cell_mesh = cells.loc[(center[0] - cells["xcm"])**2 + (center[1] - cells["ycm"])**2 <= min_distance**2]
            total_area = current_cell_mesh["area"].sum()

            if total_area == 0:
                sigmas[f"{row}{column}"] = np.array([[0, 0], [0, 0]], dtype=float)
                continue
            # calculate the tensor itself
            pressure_area_term = - np.sum([cell["pressure"] * cell["area"] for _, cell in current_cell_mesh.iterrows()])

            current_edges_mesh = big_edges.loc[(big_edges["cell1"]).isin(current_cell_mesh["ids"]) |
                                               (big_edges["cell2"].isin(current_cell_mesh["ids"]))]
            
            tension_xx = 0
            tension_yy = 0
            tension_xy = 0
            for _, bedge in current_edges_mesh.iterrows():
                vector_norm = np.linalg.norm(bedge["vector"])
                tension_xx += bedge["stress"] * (bedge["vector"][0] * bedge["vector"][0]) / vector_norm 
                tension_yy += bedge["stress"] * (bedge["vector"][1] * bedge["vector"][1]) / vector_norm 
                tension_xy += bedge["stress"] * (bedge["vector"][0] * bedge["vector"][1]) / vector_norm 

            sigma_xx = (pressure_area_term + tension_xx) / total_area
            sigma_yy = (pressure_area_term + tension_yy) / total_area
            sigma_xy = tension_xy / total_area

            sigmas[f"{row}{column}"] = np.array([[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]], dtype=float)

    ####################
    return sigmas, bins_centers, (x_bins, y_bins)