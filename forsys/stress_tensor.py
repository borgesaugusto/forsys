import numpy as np
import pandas as pd
from typing import Tuple

# TODO Use map() to improve dfs creation

def get_cells_df(frame: object) -> pd.DataFrame:
    """Create a DataFrame with all the information about the cells required
    to calculate the stress tensor.

    :param frame: Frame at which to calculate the stress tensor
    :type frame: object
    :return: Dataframe with ids, centroid position, area and perimeter of cells in the system 
    :rtype: pd.DataFrame
    """
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

def get_big_edges_df(frame: object) -> pd.DataFrame:
    """Create a DataFrame with all the information about the edges required
    to calculate the stress tensor.

    :param frame: Frame at which to calculate the stress tensor
    :type frame: object
    :return: Dataframe with ids, edge's stress, and which cells are around it.
    :rtype: pd.DataFrame
    """
    bedges = pd.DataFrame()
    
    bedges_celli = []
    bedges_cellj = []

    bedges_vector = []

    bedges_ids = [big_edge.big_edge_id for _, big_edge in frame.big_edges.items()]
    bedges_stress = [big_edge.tension for _, big_edge in frame.big_edges.items()]
    # bedges_vector= [big_edge.get_vector_from_vertex(big_edge.vertices[0].id) for _, big_edge in frame.big_edges.items()]
    for _, big_edge in frame.big_edges.items():
        bedges_celli.append(big_edge.own_cells[0])
        try:
            bedges_cellj.append(big_edge.own_cells[1])
        except IndexError:
            bedges_cellj.append(-1)

        orientation = np.dot(big_edge.xs, np.roll(big_edge.ys,1)) - np.dot(big_edge.ys, np.roll(big_edge.xs, 1))
        if orientation > 0:
            bedges_vector.append(big_edge.get_vector_from_vertex(big_edge.vertices[0].id))
        else:
            bedges_vector.append(big_edge.get_vector_from_vertex(big_edge.vertices[-1].id))

    bedges["ids"] = bedges_ids
    bedges["stress"] = bedges_stress
    bedges["vector"] = bedges_vector
    bedges["cell1"] = bedges_celli
    bedges["cell2"] = bedges_cellj

    return bedges

def stress_tensor(frame: object, grid: int = 5, radius: float = 1) -> Tuple:
    """Calculate stress tensor in the system using Batchelor's formula.
    Batchelor, G. K. The stress system in a suspension of force-free particles. J. Fluid Mech. 41, 545-570 (1970).

    :param frame: Frame object when to calculate the stress tensor.
    :type frame: object
    :param grid: Amount of bins to divide the space in, defaults to 5
    :type grid: int, optional
    :param radius: Number of average cell radii to perform the summation, defaults to 1
    :type radius: float, optional
    :return: Tuple with stress tensor's values, bin positions and grid centers.
    :rtype: Tuple
    """
    sigmas = {}
    cells = get_cells_df(frame)
    big_edges = get_big_edges_df(frame)

    _, x_bins = np.histogram(cells["xcm"], grid)
    _, y_bins = np.histogram(cells["ycm"], grid)

    x_bin_centers = [(x_bins[ii] + x_bins[ii+1]) / 2 for ii in range(len(x_bins) -1)]
    y_bin_centers = [(y_bins[ii] + y_bins[ii+1]) / 2 for ii in range(len(y_bins) -1)]
    bins_centers = [x_bin_centers, y_bin_centers]

    min_distance = radius * np.sqrt(cells["area"].mean() / np.pi)

    for row in range(grid):
        for column in range(grid):
            center = ((x_bins[row + 1] + x_bins[row]) / 2, (y_bins[column + 1] + y_bins[column]) / 2) 
            current_cell_mesh = cells.loc[(center[0] - cells["xcm"])**2 + (center[1] - cells["ycm"])**2 <= min_distance**2]
            total_area = current_cell_mesh["area"].sum()

            if total_area == 0:
                sigmas[f"{row}{column}"] = np.array([[0, 0], [0, 0]], dtype=float)
                continue

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

    return sigmas, bins_centers, (x_bins, y_bins)