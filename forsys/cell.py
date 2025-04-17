from dataclasses import dataclass
import numpy as np
from typing import Tuple
import scipy.optimize as sco
import circle_fit as cfit

import forsys.virtual_edges as ve

@dataclass 
class Cell:
    """
    Class representation of the cells objects.

    :param id: Unique identifier of the cell
    :type id: int
    :param vertices: Vertices that form the cell
    :type vertices: list
    :param is_border: True if the cell is in the border , defaults to False
    :type is_border: float
    :param gt_pressure: Ground truth pressure of the cell, defaults to None
    :type gt_pressure: float
    :param pressure: Inferred pressure of the cell, defaults to None
    :type pressure: float
    """
    id: int
    vertices: list
    is_border: bool = False
    ctype: str = None

    gt_pressure: float = None
    pressure: float = None
    center_method: str = "dlite"


    def __post_init__(self):
        """Constructor method
        """
        self.center_x = None
        self.center_y = None
        self.neighbors = None

        for v in self.vertices:
            v.add_cell(self.id)

        self.center_x, self.center_y = ve.calculate_circle_center(vertices=self.vertices, method=self.center_method)

    def __del__(self):
        for v in self.vertices:
            v.remove_cell(self.id)

    def get_cell_vertices(self) -> list:
        """
        Get the list of vertices corresponding to the cell

        :return: List of the cell's vertices
        :rtype: list
        """
        return self.vertices

    def get_next_vertex(self, v: object) -> object:
        """
        Get the next vertex from a given one in the cell ordering

        :param v: Current vertex after which the next is calculated
        :type cid: object
        :return: Object reference for the next vertex in the ordering
        :rtype: object
        """
        return self.vertices[(self.vertices.index(v) + self.get_area_sign()) % len(self.vertices)]

    def get_previous_vertex(self, v: object) -> object:
        """
        Get the previous vertex from a given one in the cell ordering

        :param v: Current vertex after which the previous is calculated
        :type cid: object
        :return: Object reference for the previous vertex in the ordering
        :rtype: object
        """
        return self.vertices[(self.vertices.index(v) - self.get_area_sign())% len(self.vertices)]

    def get_cm(self) -> list:
        """
        Get the centroid of the cell

        :return:[x, y] list with the centroid position
        :rtype: list
        """
        return [np.mean([v.x for v in self.vertices]), np.mean([v.y for v in self.vertices])]

    def get_area_sign(self) -> int:
        """
        Get the sign of the area, after the shoelace algorithm

        :return: -1 means the cell is ordered clockwise, +1 counterclockwise.
        :rtype: int
        """
        return int(np.sign(self.get_area()))

    def get_area(self) -> float:
        """
        Get the area of cell, using the shoelace formula

        :return: Cell's area
        :rtype: float
        """
        x = [i.get_coords()[0] for i in self.vertices]
        y = [i.get_coords()[1] for i in self.vertices]
        return 0.5 * (np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x, 1)))

    def get_perimeter(self) -> float:
        """
        Get the perimeter of the cell

        :return: Cell's perimeter
        :rtype: float
        """
        perimeter = 0
        for i in self.vertices:
            diffx = i.x - self.get_next_vertex(i).x
            diffy = i.y - self.get_next_vertex(i).y
            perimeter +=  np.sqrt(diffx**2 + diffy**2)
        return perimeter

    def replace_vertex(self, vold: object, vnew: object) -> None:
        """
        Replace vertex vold with vnew in the vertices list and
        add this cell to that vertex. If the vertex is already
        in the cell, remove it.

        :param vold: Old vertex object to replace
        :type vold: object
        :param vnew: New vertex to replace in vold positions
        :type vnew: object
        """
        vertices_ids = [v.id for v in self.vertices]
        # self.vertices[vertices_ids.index(vold.id)].remove_cell(self.id)
        if vnew.id in vertices_ids:
            # print(f"Warning: Vertex {vnew.id} already in cell {self.id}. Removing vertex {vold.id}")
            self.vertices.remove(vold)
        else:
            self.vertices[vertices_ids.index(vold.id)] = vnew
            # add cell to vertex
            vnew.add_cell(self.id)


    def calculate_neighbors(self) -> list:
        """
        Get all neighboring cell's ids and set the corresponding class variable

        :return: IDs of neighboring cells
        :rtype: list
        """
        # TODO Functionalize
        current_cells = set()
        for vertex in self.vertices:
            [current_cells.add(cell_id) for cell_id in vertex.ownCells]
        current_cells = list(current_cells)
        
        current_cells.remove(self.id)
        self.neighbors = current_cells
        return self.neighbors

    def get_edges(self) -> list:
        cell_edges = []
        for vnum in range(0, len(self.vertices) - 1):
            local_edges = list(set(self.vertices[vnum].ownEdges) &
                               set(self.vertices[vnum + 1].ownEdges))
            cell_edges.append(local_edges[0])
        return cell_edges
