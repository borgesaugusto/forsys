from dataclasses import dataclass, field
import numpy as np
from typing import Tuple

import forsys.virtual_edges as ve

@dataclass
class SmallEdge:
    """
    Class representation for small edges. This are defined as the lines joining two vertices.

    :param id: Unique identifier of the small edge
    :type id: int
    :param v1: First vertex of the edge
    :type v1: object
    :param v2: Second vertex of the edge
    :type v2: object
    :param tension: Inferred tension in this edge. Defaults to 0
    :type tension: float
    :param gt: Ground truth tension of this edge. Defaults to 0
    :type gt: float
    """
    id: int
    v1: object
    v2: object
    tension: float = 0
    gt: float = 0

    def __post_init__(self):
        """ Constructor method
        """
        for v in self.get_vertices_array():
            v.add_edge(self.id)
        assert self.v1.id != self.v2.id, f"edge {self.id} with the same vertex twice"

    def __del__(self):
        for v in self.get_vertices_array():
            if self.id in v.ownEdges:
                v.remove_edge(self.id)

    def get_vertices_id(self) -> list:
        """
        Get IDs of the vertices that form the edge

        :return: Array with the two IDs of the vertices
        :rtype: list
        """
        return [self.v1.id, self.v2.id]

    def get_other_vertex_id(self, one: int) -> int:
        """
        Get the vertex ID of the other vertex in the edge, given one of them

        :param one: ID of the cell to be added
        :type one: int
        :return: ID of the vertex
        :rtype: int
        """
        assert one in self.get_vertices_id(), "vertex not in asked edge"
        return self.v1.id if one == self.v2.id else self.v2.id

    def get_vertices_array(self) -> list:
        """
        Get the IDs of the vertices that form the edge

        :return: [id_1, id_2] array for both vertices' IDs
        :rtype: list
        """
        return [self.v1, self.v2]

    def replace_vertex(self, vold: object, vnew: object) -> None:
        """
        Replace one of the vertices in the edge with a new one

        :param vold: Vertex object to be replaced
        :type vold: object
        :param vnew: Vertex object to be added
        :type vnew: object
        """
        who = 0 if self.v1.id == vold.id else 1
        self.get_vertices_array()[who].remove_edge(self.id)
        if who == 0:
            self.v1 = vnew
        else:
            self.v2 = vnew
        self.get_vertices_array()[who].add_edge(self.id)

    def get_vector(self) -> list:
        """
        Get the vector connecting the two vertices of the edge.

        :return: Vector in the (0, 0) frame of reference
        :rtype: list
        """
        vector = [self.v2.x - self.v1.x, self.v2.y - self.v1.y]
        return vector


@dataclass
class BigEdge:
    """
    Class representation for big edges. This are defined as a set of SmallEdges.
    Could be only one small edge in special cases. This allows for curved representations
    of the cellular membrane.

    :param big_edge_id: Unique identifier of the big edge
    :type big_edge_id: int
    :param vertices: List of vertices form the big edge
    :type vertices: list
    :param tension: Inferred tension in this edge. Defaults to 0
    :type tension: float
    :param gt: Ground truth tension of this edge. Defaults to 0
    :type gt: float
    :param external: True if this big edge is external to the tissue
    :type external: bool
    """
    big_edge_id: int
    vertices: list

    tension: float = 0.0
    gt: float = 0.0
    external: bool = False

    def __post_init__(self):
        """ Constructor method
        """
        for vertex in self.vertices:
            vertex.add_big_edge(self.big_edge_id)

        self.xs = [vertex.x for vertex in self.vertices]
        self.ys = [vertex.y for vertex in self.vertices]
        self.edges = [list(set(self.vertices[vid].ownEdges) &
                           set(self.vertices[vid + 1].ownEdges))[0]
                      for vid in range(0, len(self.vertices) - 1)]

        if len(self.vertices) == 2:
            self.own_cells = list(set(self.vertices[0].ownCells) & set(self.vertices[1].ownCells))
        else:
            self.own_cells = self.vertices[(len(self.vertices) - 1) // 2].ownCells
        vertices_own_cells = [len(vertex.ownCells) < 2 for vertex in self.vertices]
        has_junction_at_extremes = True if len(self.vertices[0].ownCells) > 2 \
                                           or len(self.vertices[-1].ownCells) > 2 else False
        self.external = True if np.any(vertices_own_cells) or not has_junction_at_extremes \
            else False
        

    def calculate_curvature(self) -> float:
        """
        Get the curvature of the big edge

        :return: Object reference for the next vertex in the ordering
        :rtype: object
        """
        dx_dt = np.gradient(self.xs)
        dy_dt = np.gradient(self.ys)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / ((dx_dt**2 + dy_dt**2)**1.5)

        return curvature

    def calculate_total_curvature(self, normalized=True) -> float:
        """
        Get the total curvature of the big edge normalized by its length

        :param normalized: Current vertex after which the next is calculated
        :type normalized: bool
        :return: Object reference for the next vertex in the ordering
        :rtype: object
        """
        curvatures = self.calculate_curvature()
        ds = np.sqrt(np.diff(self.xs)**2 + np.diff(self.ys)**2)
        total_curvature = np.sum((curvatures[1:] + curvatures[:-1])/2 * ds)
        if normalized:
            curve_length = np.sum(np.sqrt(np.diff(self.xs)**2 + np.diff(self.ys)**2))
            total_curvature = total_curvature / curve_length
        return total_curvature


    def get_versor_from_vertex(self, vid: int, method: str = "edge", cell: object = None, fit_method:str = "dlite") -> list:
        """
        Get the versor corresponding to the vectorial representation of the big edge.
        
        :param vid: The ID of the vertex.
        :type vid: int
        :param method: The method used to calculate the vector representation of the big edge. Default is "edge".
        :type method: str, optional
        :param cell: The cell object. Default is None.
        :type cell: object, optional
        :param fit_method: The fit method used. Default is "dlite".
        :type fit_method: str, optional
        :return: The versor corresponding to the vectorial representation of the big edge.
        :rtype: list
        """
        vector = self.get_vector_from_vertex(vid, method, cell, fit_method)
        versor = vector / np.linalg.norm(vector)
        return versor

    def get_vector_from_vertex(self, vid: int, method: str = "edge", cell: object = None, fit_method: str="dlite") -> list:
        """
        Get the vector representation for the big edge

        :param vid: ID of the vertex from where the vector is defined
        :type vid: int
        :param method: Method to calculate representation. "edge" gives the circular fit to the edge, \
            "cell" and passing a cell object gives a circular fit to the given cell.
        :type method: str
        :param cell: Object reference of the cell to calculate the curvature if necessary
        :type cell: object
        :return: x, y coordinates of the vector
        :rtype: list
        """
        vobject = self.get_vertex_object_by_id(vid)
        if method == "edge":
            xc, yc = ve.calculate_circle_center(self.vertices, method=fit_method)
        elif method == "cell" and cell:
            xc, yc = cell.center_x, cell.center_y
        else:
            raise Exception("Method for versor doesn't exist or cell missing")

        vector = np.array((- (vobject.y - yc), (vobject.x - xc)))

        correct_sign = self.get_versor_sign(vid)
        if np.any(np.sign(vector) != correct_sign):
            correction = correct_sign * np.sign(vector)
            vector = vector * correction
        return vector

    def get_vertex_object_by_id(self, vid: int) -> object:
        """
        Get the vertex object corresponding to the given ID

        :param vid: Current vertex after which the next is calculated
        :type vid: int
        :return: Object reference for required vertex
        :rtype: object
        """
        vertex_object = list(filter(lambda x: x.id == vid, self.vertices))
        assert len(vertex_object) == 1, "More than one or no vertex with the same ID"
        return vertex_object[0]

    def get_vertices_ids(self) -> list:
        """
        Get all IDs of the vertices that form the big edge

        :return: Array with all vertices' IDs
        :rtype: list
        """
        return [vertex.id for vertex in self.vertices]

    def get_versor_sign(self, vid: int) -> list:
        """
        Get the correction sign for each
        componenet of the vector

        :param vid: Vertex to calculate the sign from
        :type vid: int
        :return: Array with the sign corrections
        :rtype: list
        """
        versor = self.get_straight_edge_versor_from_vid(vid)
        signs = np.sign(versor)
        return [1.0 if sign == 0 else sign for sign in signs]


    def get_straight_edge_versor_from_vid(self, vid: int) -> list:
        all_vertices_ids = self.get_vertices_ids()
        v0 = self.get_vertex_object_by_id(vid)
        # connected to ?
        if all_vertices_ids[0] == vid:
            next_vid = all_vertices_ids[1]
        elif vid == all_vertices_ids[-1]:
            next_vid = all_vertices_ids[-2]
        else:
            raise ("Vertex ID is not in a junction")

        v1 = self.get_vertex_object_by_id(next_vid)
        return [v1.x - v0.x, v1.y - v0.y]

    def get_length(self):
        """
        Get the length of the big edge

        :return: Length of the big edge
        :rtype: float
        """
        return np.sum(np.sqrt(np.diff(self.xs)**2 + np.diff(self.ys)**2))
