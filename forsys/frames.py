import numpy as np
import pandas as pd
import os

from dataclasses import dataclass, field

import forsys as fs
import forsys.edge as fedge

@dataclass
class Frame():
    frame_id: int
    vertices: dict
    edges: dict
    cells: dict

    time: float=0.0

    # gt: dict = field(default_factory=dict)
    gt: bool=False

    surface_evolver: bool=False

    stress_tensor: dict = field(default_factory=dict)

    def __post_init__(self):
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
        # add gt values
        if self.gt:
            for big_edge_id, big_edge in self.big_edges.items():
                # edges_to_use = self.get_big_edge_edgesid(big_edge)
                objects = [self.edges[eid].gt for eid in big_edge.edges]
                # self.big_edge_gt_tension[big_edge_id] = np.mean(objects)
                big_edge.gt = np.mean(objects)
            # self.assign_gt_tensions_to_big_edges(self.gt)

    def assign_tensions(self, xresult):
        for i in range(0, len(self.big_edges_list)):
            e = self.big_edges_list[i]
            edges_to_use = self.get_big_edge_edgesid(e)
            for e in edges_to_use:
                self.edges[e].tension = xresult[i]

    def get_big_edges(self, use_all=False):
        if use_all:
            big_edges_to_use = self.big_edges       
        else:
            big_edges_to_use = [big_edge for big_edge in self.big_edges.values() 
                                if not big_edge.external]
        return big_edges_to_use
        
    def get_external_edges(self):
        return [big_edge for big_edge in self.big_edges.values() 
                                if big_edge.external]
    
    def get_external_edges_ids(self):
        return [big_edge.big_edge_id for big_edge in self.big_edges.values() 
                                if big_edge.external]

    def assign_gt_tensions_to_big_edges(self, gt_tensions, use_all=False):
        if use_all:
            big_edges_to_use = self.get_big_edges(use_all=True)
        else:
            big_edges_to_use = self.get_big_edges(use_all=False)
        
        for big_edge_id, big_edge in enumerate(big_edges_to_use):
            big_edge.gt = gt_tensions[big_edge_id]
            for edge_id in big_edge.edges:
                self.edges[edge_id].gt = big_edge.gt

    def assign_tensions_to_big_edges(self):
        # construct the big edges and average the tensions that were obtained
        # for big_edge_id, big_edge in enumerate(self.big_edges_list):
        for big_edge_id, big_edge in self.big_edges.items():
            objects = [self.edges[eid].tension for eid in big_edge.edges]
            self.big_edges[big_edge_id].tension = np.mean(objects)


    def assign_gt_small_edges(self, ground_truth ,big_edges_to_use=False):
        if big_edges_to_use == "ext":
            big_edges_to_use = self.big_edges_list
        else:
            big_edges_to_use = self.internal_big_edges

        for edge_id, vertices_list in enumerate(big_edges_to_use):
            edges_to_use = self.get_big_edge_edgesid(vertices_list)
            for small_edge_id in edges_to_use:
                self.edges[small_edge_id].gt = ground_truth[edge_id]

    def assign_pressures(self, pressures, mapping):
        for cid, cell in self.cells.items():
            # key_to_use = list(self.cells.keys()).index(cid)
            key_to_use = mapping[cid]
            cell.pressure = pressures[key_to_use]

    def get_big_edge_edgesid(self, big_edge_vertices):
        # TODO legacy function to-remove
        # e = self.big_edges_list[beid]
        return [list(set(self.vertices[big_edge_vertices[vid]].ownEdges) & 
                set(self.vertices[big_edge_vertices[vid+1]].ownEdges))[0]
                for vid in range(0, len(big_edge_vertices)-1)]

    def get_tensions(self, with_border=False):
        # df = pd.DataFrame.from_dict(self.big_edge_gt_tension.items()).rename(columns={0: 'id', 1: 'gt'})
        df = pd.DataFrame.from_dict({beid: big_edge.gt for beid, big_edge in self.big_edges.items()}.items()).rename(columns={0: 'id', 1: 'gt'})
        # df['tension'] = self.big_edge_tension.values()
        df['tension'] = [big_edge.tension for big_edge in self.big_edges.values()]
        if not with_border:
            # Only return results that don't belong the edges in the border
            df = df.loc[~df.id.isin(self.get_external_edges_ids())]

        return df

    def get_pressures(self):
        pressure_df = pd.DataFrame.from_dict({cid: cell.gt_pressure for cid, cell in self.cells.items()}.items()).rename(columns={0: id, 1: "gt_pressure"})
        inferred_pressures = [cell.pressure for cell in self.cells.values()]
        pressure_df["pressures"] = inferred_pressures

        return pressure_df

    def export_tensions(self, fname, folder="", is_gt=True, with_border=False):
        if is_gt:
            df = pd.DataFrame.from_dict(self.big_edge_gt_tension.items()).rename(columns={0: 'id', 1: 'tension'})
        else:
            df = pd.DataFrame.from_dict(self.big_edge_tension.items()).rename(columns={0: 'id', 1: 'tension'})
        


        if not with_border:
            # Only return results that don't belong the edges in the border
            df = df.loc[~df.id.isin(self.get_external_edges_ids())]
            
        return df.to_csv(os.path.join(folder, fname), index=False)

    def get_big_edge_by_cells(self, c1_id: int, c2_id: int) -> fedge.BigEdge:
        cell_1 = self.cells[c1_id]
        cell_2 = self.cells[c2_id]
    
        c1_vertices = [v.id for v in cell_1.vertices if len(v.ownEdges) < 3]
        c2_vertices = [v.id for v in cell_2.vertices if len(v.ownEdges) < 3]
        vertices_in_common = np.intersect1d(c1_vertices, 
                                            c2_vertices)
        shared_edge = list(set([edge for vid in vertices_in_common for edge in self.vertices[vid].own_big_edges]))
        return self.big_edges[shared_edge[0]]

    def calculate_stress_tensor(self, coarsing=5, radius=1):
        self.stress_tensor = fs.stress_tensor.stress_tensor(self, coarsing, radius)

        self.principal_stress = {}

        for row in range(len(self.stress_tensor[1][0])):
            for column in range(len(self.stress_tensor[1][1])):
                principal_component = np.linalg.eig(self.stress_tensor[0][f"{row}{column}"])
                self.principal_stress[(self.stress_tensor[1][0][row], 
                                            self.stress_tensor[1][1][column])] = principal_component


