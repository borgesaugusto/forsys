import numpy as np
import pandas as pd
import os

from dataclasses import dataclass, field

from forsys import surface_evolver, virtual_edges as ve

@dataclass
class Frame():
    vertices: dict
    edges: dict
    cells: dict

    time: int=0

    gt: dict = field(default_factory=dict)

    surface_evolver: bool=False

    def __post_init__(self):
        self.big_edge_gt_tension = {}
        self.big_edge_tension = {}
        # self.earr = ve.create_edges(self.cells)
        self.earr = ve.create_edges_new(self.vertices, self.cells)
        if self.surface_evolver:
            self.big_edges_list, self.border_vertices = ve.get_big_edges_se(self)
            self.earr = self.big_edges_list
        else:
            self.big_edges_list = self.earr
            self.border_vertices = ve.get_border_from_angles_new(self.earr, self.vertices)

        self.external_edges_id = [self.earr.index(e) for e in ve.get_border_edge(self.earr, self.vertices)]
        # print(self.external)
        # self.external = self.earr
        # add gt values
        if len(self.gt) != 0:
            self.assign_gt_tensions_to_big_edges()

    def assign_tensions(self, xresult):
        for i in range(0, len(self.big_edges_list)):
            e = self.big_edges_list[i]
            edges_to_use = self.get_big_edge_edgesid(e)
            for e in edges_to_use:
                self.edges[e].tension = xresult[i]

    def assign_gt_tensions_to_big_edges(self):
        for i in range(0, len(self.big_edges_list)):
            edges_to_use = self.get_big_edge_edgesid(self.big_edges_list[i])
            objects = [self.edges[eid].gt for eid in edges_to_use]
            self.big_edge_gt_tension[i] = np.mean(objects)
            # exit()

    def assign_tensions_to_big_edges(self):
        # construct the big edges and average the tensions that were obtained
        for i in range(0, len(self.big_edges_list)):
            edges_to_use = self.get_big_edge_edgesid(self.big_edges_list[i])
            objects = [self.edges[eid].tension for eid in edges_to_use]
            self.big_edge_tension[i] = np.mean(objects)


    def get_big_edge_edgesid(self, big_edge_vertices):
        # e = self.big_edges_list[beid]
        return [list(set(self.vertices[big_edge_vertices[vid]].ownEdges) & 
                set(self.vertices[big_edge_vertices[vid+1]].ownEdges))[0]
                for vid in range(0, len(big_edge_vertices)-1)]

    def get_tensions(self, with_border=False):
        df = pd.DataFrame.from_dict(self.big_edge_gt_tension.items()).rename(columns={0: 'id', 1: 'gt'})
        df['tension'] = self.big_edge_tension.values()
        if not with_border:
            # Only return results that don't belong the edges in the border
            df = df.loc[~df.id.isin(self.external_edges_id)]

        return df

    def export_tensions(self, fname, folder="", is_gt=True, with_border=False):
        if is_gt:
            df = pd.DataFrame.from_dict(self.big_edge_gt_tension.items()).rename(columns={0: 'id', 1: 'tension'})
        else:
            df = pd.DataFrame.from_dict(self.big_edge_tension.items()).rename(columns={0: 'id', 1: 'tension'})
        


        if not with_border:
            # Only return results that don't belong the edges in the border
            df = df.loc[~df.id.isin(self.external_edges_id)]


        
        return df.to_csv(os.path.join(folder, fname), index=False)