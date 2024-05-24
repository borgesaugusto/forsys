from dataclasses import dataclass
import pandas as pd
import os
import re
from typing import Tuple

import forsys.vertex as vertex
import forsys.edge as edge
import forsys.cell as cell

@dataclass
class SurfaceEvolver:
    """Class interface with SurfaceEvolver-generated files
    
    :param fname: Path to the SurfaceEvolver file to read
    :type fname: str
    """
    fname: str


    def __post_init__(self):
        self.vertices, self.edges, self.cells = self.create_lattice()

    def create_lattice(self) -> Tuple:
        """
        Create vertices, edges and cells from a Surface Evolver file. 
        All necessary steps are taken by this call.

        :return: Three dictionaries with the vertices, edges and cells respectively   
        :rtype: Tuple
        """
        edges_temp = self.get_edges()
        
        vertices = {}
        for _, r in self.get_vertices().iterrows():
            vertices[int(r.id)] = vertex.Vertex(int(r.id), r.x, r.y)
            
        edges  = {}
        for _, r in edges_temp.iterrows():
            edges[int(r.id)] = edge.SmallEdge(int(r.id), vertices[int(r.id1)], vertices[int(r.id2)])
            edges[int(r.id)].gt = round(edges_temp.loc[edges_temp['id'] == int(r.id)]['force'].iloc[0], 4)

        cells = {}
        for _, r in self.get_cells().iterrows():
            vlist = [edges[abs(e)].v1 if e > 0 else edges[abs(e)].v2 for e in r.edges]
            gt_pressure = round(r["pressures"], 4)
            cells[int(r.id)] = cell.Cell(int(r.id), vlist, gt_pressure=gt_pressure)

        # delete all vertices and edges with no cell
        vertex_to_delete = []
        for vid, v in vertices.items():
            if len(v.ownCells) == 0:
                vertex_to_delete.append(int(vid))

        for i in vertex_to_delete:
            for e in vertices[i].ownEdges.copy():
                try:
                    del edges[e] 
                except KeyError:
                    pass
        
            del vertices[i]

        return vertices, edges, cells

    def calculate_first_last(self) -> Tuple:
        """
        Calculate starting position of the vertices, edges, cells and pressures
        definitions in the Surface Evolver file.

        :return: Four tuples with the initial and final line index for the \
            vertices, edges, cells and pressures respectively.
        :rtype: Tuple
        """
        with open(os.path.join(self.fname),"r") as f:
            ini_v = next(index for index, line in enumerate(f) if line.startswith("vertices  "))
            fin_v = next(index for index, line in enumerate(f) if line.startswith("edges  ")) + ini_v
            f.seek(0)
            ini_e = next(index for index, line in enumerate(f) if line.startswith("edges  "))
            fin_e = next(index for index, line in enumerate(f) if line.startswith("faces  ")) + ini_e
            f.seek(0)
            ini_f = next(index for index, line in enumerate(f) if line.startswith("faces  "))
            fin_f = next(index for index, line in enumerate(f) if line.startswith("bodies  ")) + ini_f
            f.seek(0)
            ini_p = next(index for index, line in enumerate(f) if line.startswith("bodies  "))
            fin_p = next(index for index, line in enumerate(f) if line.startswith("read")) + ini_p

        self.index_v = (ini_v, fin_v)
        self.index_e = (ini_e, fin_e)
        self.index_f = (ini_f, fin_f)
        self.index_pressures = (ini_p, fin_p)

        return self.index_v, self.index_e, self.index_f, self.index_pressures


    def get_first_last(self) -> Tuple:
        """
        Get list of line indices for all conditions. If the lines are already known,
        returns known values. If not, sets them and the returns their values.

        :return: Four tuples with the initial and final line index for the \
            vertices, edges, cells and pressures respectively.
        :rtype: Tuple
        """
        try:
            return self.index_v, self.index_e, self.index_f, self.index_pressures
        except AttributeError:
            return self.calculate_first_last()
    
    def get_vertices(self) -> pd.DataFrame:
        """
        Generate DataFrame with the system's vertices IDs and positions

        :return: Dataframe with three columns: ID, x and y for each vertex in \
            the system.
        :rtype: pd.DataFrame
        """
        vertices = pd.DataFrame()
        ids = []
        xs = []
        ys = []

        index_v, _, _, _ = self.get_first_last()

        with open(os.path.join(self.fname), "r") as f:
            lines = f.readlines()
            for i in range(index_v[0] + 1, index_v[1]):
                ids.append(int(re.search(r"\d+", lines[i]).group()))
                xs.append(round(float(lines[i].split()[1]), 3))
                ys.append(round(float(lines[i].split()[2]), 3))
                # vals.append(float(lines[i][lines[i].find("density"):].split(" ")[1]))
        vertices['id'] = ids
        vertices['x'] = xs
        vertices['y'] = ys

        return vertices
    
    def get_edges(self) -> pd.DataFrame:
        """
        Generate DataFrame with the system's edges IDs, the two vertices that 
        form them and the ground truth density.

        :return: Dataframe with four columns: ID, vertex 1, vertex 2 \
            and ground truth density for each edge.
        :rtype: pd.DataFrame
        """
        edges = pd.DataFrame()
        ids = []
        id1 = []
        id2 = []
        forces = []

        _, index_e, _, _ = self.get_first_last()

        with open(os.path.join(self.fname), "r") as f:
            lines = f.readlines()
            for i in range(index_e[0] + 1, index_e[1]):
                ids.append(int(re.search(r"\d+", lines[i]).group()))
                id1.append(int(lines[i].split()[1]))
                id2.append(int(lines[i].split()[2]))
                forces.append(float(lines[i].split()[4]) if lines[i].split()[3] == "density" else 1)
                # vals.append(float(lines[i][lines[i].find("density"):].split(" ")[1]))
        edges['id'] = ids
        edges['id1'] = id1
        edges['id2'] = id2
        edges['force'] = forces
        return edges

    def get_cells(self) -> pd.DataFrame:
        """
        Generate DataFrame with the system's cells IDs, its edges, 
        and their ground truth pressures.

        :return: Dataframe with three columns: ID, array of edges from the cell \
            and the ground truth pressure.
        :rtype: pd.DataFrame
        """
        cells = pd.DataFrame()
        ids = []
        edges = []

        pressure_dict = self.get_pressures()

        _, _, index_f, _ = self.get_first_last()

        with open(os.path.join(self.fname), "r") as f:
            lines = f.readlines()
            current_edge = []
            first = True

            for i in range(index_f[0] + 1, index_f[1]):
                splitted = lines[i].split()
                if first and "*/" not in splitted[-1]:
                    ids.append(splitted[0])
                    first = False
                    current_edge = current_edge+splitted[1:-1]
                elif splitted[-1] == '\\' and not first:
                    current_edge = current_edge+splitted[0:-1]
                elif "*/" in splitted[-1]:
                    # last line, save
                    if first:
                        ids.append(splitted[0])
                        current_edge = current_edge+splitted[1:-2]
                    else:
                        current_edge = current_edge+splitted[0:-2]
                    edges.append([int(e) for e in current_edge])
                    current_edge = []
                    first = True
        
        cells['id'] = ids
        cells['edges'] = edges
        cells["pressures"] = pressure_dict.values()

        return cells
    
    def get_pressures(self) -> dict:
        """Get the values of the ground truth pressure of the cells

        :return: Ground truth pressure dictionary.
        :rtype: dict
        """
        pressures = {}
        _, _, _, index_p = self.get_first_last()
        with open(os.path.join(self.fname), "r") as f:
            lines = f.readlines()

            for i in range(index_p[0] + 1, index_p[1]):
                splitted = lines[i].split()
                pressures[int(splitted[0])] = float(splitted[7])
        return pressures

