import numpy as np
import pandas as pd
import scipy.signal as spsignal
import os

from dataclasses import dataclass, field
from warnings import warn

import forsys as fs
import forsys.edge as fedge

@dataclass
class Frame():
    """
    Class representation of a microscopy frame.

    :param frame_id: Unique identifier of the frame
    :type frame_id: int
    :param vertices: Vertices dictionary for the whole frame
    :type vertices: dict
    :param edges: SmallEdges dictionary for the whole frame
    :type edges: dict
    :param cells: Cells dictionary for the whole frame
    :type cells: dict
    :param time: Real time corresponding to the frame. This is used to calcualte dynamical information.
    :type time: float
    :param gt: If True, the BigEdges are assigned their ground truths. Defaults to False
    :type gt: bool, optional
    """
    frame_id: int
    vertices: dict
    edges: dict
    cells: dict

    time: float=0.0

    gt: bool=False

    def __post_init__(self):
        """Constructor method
        """
        self.big_edge_gt_tension = {}
        self.big_edge_tension = {}
        self.big_edges = {}

        self.big_edges_list = fs.virtual_edges.create_edges_new(self.vertices, self.cells)
        # create big edge objects
        for big_edge_id, big_edge in enumerate(self.big_edges_list):
            vertices_objects = [self.vertices[vid] for vid in big_edge]
            self.big_edges[big_edge_id] = fedge.BigEdge(big_edge_id, vertices_objects)

        self.external_edges_id = [self.big_edges_list.index(e) 
                                    for e in fs.virtual_edges.get_border_edge(self.big_edges_list, 
                                                                self.vertices)]

        self.internal_big_edges_vertices = [edge for eid, edge in enumerate(self.big_edges_list) 
                                        if eid not in self.external_edges_id and 
                                        (len(self.vertices[edge[0]].ownCells) > 2 or  
                                        len(self.vertices[edge[-1]].ownCells) > 2)]
        self.internal_big_edges = [self.big_edges[eid] for eid, edge in enumerate(self.big_edges_list) 
                                    if eid not in self.external_edges_id and 
                                    (len(self.vertices[edge[0]].ownCells) > 2 or  
                                    len(self.vertices[edge[-1]].ownCells) > 2)]

        for _, cell in self.cells.items():
            cell.calculate_neighbors()

        self.border_vertices = self.get_external_edges()
        if self.gt:
            for big_edge_id, big_edge in self.big_edges.items():
                objects = [self.edges[eid].gt for eid in big_edge.edges]
                big_edge.gt = np.mean(objects)

    def assign_tensions(self, xresult: list) -> None:
        """Assign tensions to the small edges following a list of values. 
        Deprecated function.

        :param xresult: Inference result from the matrix solver
        :type xresult: list
        """
        warn("Legacy function, soon to be deprecated", DeprecationWarning, 2)
        for i in range(0, len(self.big_edges_list)):
            e = self.big_edges_list[i]
            edges_to_use = self.get_big_edge_edgesid(e)
            for e in edges_to_use:
                self.edges[e].tension = xresult[i]

    def get_big_edges(self, use_all: bool = False) -> list:
        """Get list of big edges in the frame, depending on whether
        to use externals or not

        :param use_all: If True external edges are also used, defaults to False
        :type use_all: bool, optional
        :return: List of big edges to use
        :rtype: list
        """
        if use_all:
            big_edges_to_use = self.big_edges       
        else:
            big_edges_to_use = [big_edge for big_edge in self.big_edges.values() 
                                if not big_edge.external]
        return big_edges_to_use
        
    def get_external_edges(self) -> list:
        """Get list of external edges in the system

        :return: List of external edges
        :rtype: list
        """
        return [big_edge for big_edge in self.big_edges.values() 
                                if big_edge.external]
    
    def get_external_edges_ids(self) -> list:
        """Get list of the external's edges IDs.

        :return: List of IDs for all external big edges in the system.
        :rtype: list
        """
        return [big_edge.big_edge_id for big_edge in self.big_edges.values() 
                                if big_edge.external]

    def assign_gt_tensions_to_big_edges(self, gt_tensions: dict, use_all: bool = False) -> None:
        """Assign the ground truth stress to the big edges from a dictionary.

        :param gt_tensions: Dictionary with the ground truth stress values 
        :type gt_tensions: dict
        :param use_all: If True external big edges are also included, defaults to False
        :type use_all: bool, optional
        """
        if use_all:
            big_edges_to_use = self.get_big_edges(use_all=True)
        else:
            big_edges_to_use = self.get_big_edges(use_all=False)
        
        for big_edge_id, big_edge in enumerate(big_edges_to_use):
            big_edge.gt = gt_tensions[big_edge_id]
            for edge_id in big_edge.edges:
                self.edges[edge_id].gt = big_edge.gt

    def assign_tensions_to_big_edges(self) -> None:
        """Assign the inferred tension to the big edge. 
        """
        for big_edge_id, big_edge in self.big_edges.items():
            objects = [self.edges[eid].tension for eid in big_edge.edges]
            self.big_edges[big_edge_id].tension = np.mean(objects)


    def assign_gt_small_edges(self, big_edges_to_use:str = "internal") -> None:
        """Assign the ground truth stress values to the small edges.

        :param big_edges_to_use: "ext" includes external edges, defaults to "internal"
        :type big_edges_to_use: str, optional
        """
        if big_edges_to_use == "ext":
            big_edges_to_use = self.big_edges_list
        else:
            big_edges_to_use = self.internal_big_edges

        for big_edge in big_edges_to_use:
            for small_edge_id in big_edge.edges:
                self.edges[small_edge_id].gt = big_edge.gt

    def assign_pressures(self, pressures: dict, mapping: dict) -> None:
        """Assign the ground truth pressures to the cells

        :param pressures: Inferred pressures dictionary
        :type pressures: dict
        :param mapping: Mapping between cells' ids and IDs in the pressure dictionary
        :type mapping: dict
        """
        for cid, cell in self.cells.items():
            key_to_use = mapping[cid]
            cell.pressure = pressures[key_to_use]

    def get_big_edge_edgesid(self, big_edge_vertices: list) -> list:
        """Get ids of the vertices that form big edges

        :param big_edge_vertices: List of vertices in the big edge
        :type big_edge_vertices: list
        :return: List of ids
        :rtype: list
        """
        # TODO legacy function to-remove
        warn("Legacy function, soon to be deprecated", DeprecationWarning, 2)
        return [list(set(self.vertices[big_edge_vertices[vid]].ownEdges) & 
                set(self.vertices[big_edge_vertices[vid+1]].ownEdges))[0]
                for vid in range(0, len(big_edge_vertices)-1)]

    def get_tensions(self, with_border: bool = False) -> pd.DataFrame:
        """Create tension DataFrame for the ground truth and inferred values.

        :param with_border: True includes the values for the external edges, defaults to False
        :type with_border: bool, optional
        :return: DataFrame with the ID of the edge, its groun truth stress and the inferred one
        :rtype: pd.DataFrame
        """
        df = pd.DataFrame.from_dict({beid: big_edge.gt for beid, big_edge in self.big_edges.items()}.items()).rename(columns={0: 'id', 1: 'gt'})
        df['stress'] = [big_edge.tension for big_edge in self.big_edges.values()]
        if not with_border:
            # Only return results that don't belong the edges in the border
            df = df.loc[~df.id.isin(self.get_external_edges_ids())]

        return df
    

    def get_gt_tensions(self, with_border: bool = False) -> pd.DataFrame:
        df = pd.DataFrame.from_dict({beid: big_edge.gt for beid, big_edge in self.big_edges.items()}.items()).rename(columns={0: 'id', 
                                                                                                                                1: 'gt'})
        if not with_border:
            df = df.loc[~df.id.isin(self.get_external_edges_ids())]
        return df


    def get_pressures(self) -> pd.DataFrame:
        """Create the pressures DataFrame.

        :return: DataFrame with the ID of the cells, their ground truth pressure and the inferred value
        :rtype: pd.DataFrame
        """
        pressure_df = pd.DataFrame.from_dict({cid: cell.gt_pressure for cid, cell in self.cells.items()}.items()).rename(columns={0: "id", 1: "gt_pressure"})
        inferred_pressures = [cell.pressure for cell in self.cells.values()]
        pressure_df["pressure"] = np.array(inferred_pressures).astype(float)

        return pressure_df

    def export_tensions(self, fname: str, folder: str = "", is_gt: bool = True, with_border: bool = False) -> str:
        """Create DataFrame from dictionary of either the ground truth value
        or the inferred ones.

        :param fname: File name to save csv file
        :type fname: str
        :param folder: Folder to save output to, defaults to ""
        :type folder: str, optional
        :param is_gt: True saves the ground truth values, False saves the inferred ones, defaults to True
        :type is_gt: bool, optional
        :param with_border: True saves values including external edges, defaults to False
        :type with_border: bool, optional
        :return: Result of pd.DataFrame().to_csv() function call
        :rtype: str
        """
        if is_gt:
            df = pd.DataFrame.from_dict(self.big_edge_gt_tension.items()).rename(columns={0: 'id', 1: 'tension'})
        else:
            df = pd.DataFrame.from_dict(self.big_edge_tension.items()).rename(columns={0: 'id', 1: 'tension'})
        


        if not with_border:
            # Only return results that don't belong the edges in the border
            df = df.loc[~df.id.isin(self.get_external_edges_ids())]
            
        return df.to_csv(os.path.join(folder, fname), index=False)

    def get_big_edge_by_cells(self, c1_id: int, c2_id: int) -> fedge.BigEdge:
        """Get big edge object from the cells that compose it

        :param c1_id: ID of one the cells
        :type c1_id: int
        :param c2_id: ID of the other cell
        :type c2_id: int
        :return: BigEdge object that is the interface between both cells
        :rtype: fedge.BigEdge
        """
        cell_1 = self.cells[c1_id]
        cell_2 = self.cells[c2_id]
    
        c1_vertices = [v.id for v in cell_1.vertices if len(v.ownEdges) < 3]
        c2_vertices = [v.id for v in cell_2.vertices if len(v.ownEdges) < 3]
        vertices_in_common = np.intersect1d(c1_vertices, 
                                            c2_vertices)
        shared_edge = list(set([edge for vid in vertices_in_common for edge in self.vertices[vid].own_big_edges]))
        return self.big_edges[shared_edge[0]]

    def calculate_stress_tensor(self, coarsing: int = 5, radius: float = 1) -> None:
        """Calculate the principal components of the stress tensor.

        :param coarsing: Number of bins to divide the space, defaults to 5
        :type coarsing: int, optional
        :param radius: Number of average cell raii to integrate the tensor, defaults to 1
        :type radius: float, optional
        """
        self.stress_tensor = fs.stress_tensor.stress_tensor(self, coarsing, radius)

        self.principal_stress = {}

        for row in range(len(self.stress_tensor[1][0])):
            for column in range(len(self.stress_tensor[1][1])):
                principal_component = np.linalg.eig(self.stress_tensor[0][f"{row}{column}"])
                self.principal_stress[(self.stress_tensor[1][0][row], 
                                            self.stress_tensor[1][1][column])] = principal_component
        
    def filter_edges(self, method: str = "SG") -> None:
        """
        Filter the edges of the frame using the specified method.

        :param method: The filtering method to use. Options are "SG" for Savitzky-Golay filter or "none" for no filtering.
        :type method: str, optional
        :raises ValueError: If the specified method is not available.
        :return: None
        """
        if method == "SG":
            filtering_function = spsignal.savgol_filter
            arguments = (5, 3)
        elif method == "none":
            filtering_function = lambda x, *args: x
            arguments = ()
        else:
            raise ValueError("Method not available")

        for index, big_edge in self.big_edges.items():
            try:
                new_xs = filtering_function(big_edge.xs, *arguments)
                new_ys = filtering_function(big_edge.ys, *arguments)
            except ValueError:
                new_xs = big_edge.xs
                new_ys = big_edge.ys
            for index, vertex in enumerate(big_edge.vertices):
                vertex.x = new_xs[index]
                vertex.y = new_ys[index]

    def get_cell_properties_df(self, center_method) -> pd.DataFrame:
        all_ids = []
        all_centerx = []
        all_centery = []
        all_perimeters = []
        all_areas = []
        all_pressures = []

        for cell in self.cells.values():
            center = fs.virtual_edges.calculate_circle_center(cell.vertices, center_method)
            all_ids.append(cell.id)
            all_centerx.append(center[0])
            all_centery.append(center[1])
            all_perimeters.append(cell.get_perimeter())
            all_areas.append(cell.get_area())
            all_pressures.append(cell.pressure)

        return pd.DataFrame({"cid": all_ids,
                             "centerx": all_centerx,
                             "centery": all_centery,
                             "perimeter": all_perimeters,
                             "area": all_areas,
                             "pressure": all_pressures})


    def get_edges_props_df(self) -> pd.DataFrame:
        all_ids = []
        all_tensions = []
        all_gt_tensions = []
        all_posx = []
        all_posy = []

        for edge in self.big_edges.values():
            all_ids.append(edge.big_edge_id)
            all_tensions.append(edge.tension)
            all_gt_tensions.append(edge.gt)
            all_posx.append(np.mean(edge.xs))
            all_posy.append(np.mean(edge.ys))

        return pd.DataFrame({"eid": all_ids,
                             "posx": all_posx,
                             "posy": all_posy,
                             "tension": all_tensions,
                             "gt_tension": all_gt_tensions})
