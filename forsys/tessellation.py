import numpy as np
import scipy.spatial as scspace
from copy import deepcopy
import itertools as itert

import forsys.vertex as fvertex
import forsys.edge as fedge
import forsys.cell as fcell

def create_lattice(vertices_voronoi, edges_voronoi, cells_voronoi, **kwargs) -> tuple:
    vertices = {}
    edges = {}
    cells = {}
    for vid, vertex in vertices_voronoi.items():
        vertices[vid] = fvertex.Vertex(vid, vertex[0], vertex[1])
    
    for eid, edge in edges_voronoi.items():
        edges[abs(eid)] = fedge.SmallEdge(abs(eid), vertices[edge[0]], vertices[edge[-1]])

    for cid, cell in cells_voronoi.items():
        vertices_in_cell = []
        for eid in cell:
            which = 0 if eid > 0 else -1
            current_vertex = vertices[edges[abs(eid)].get_vertices_id()[which]]
            vertices_in_cell.append(current_vertex)

        vertices_in_cell = vertices_in_cell[::-1] if cid < 0 else vertices_in_cell
        cells[abs(cid)] = fcell.Cell(abs(cid), vertices_in_cell)

    return vertices, edges, cells

def create_lattice_elements(cell_centers: list, **kwargs) -> tuple:
    """
    Create the vertices, edges and cells from the scipy.spatial.Voronoi() objects

    :param cell_centers: Centroids of the objects 
    :type cell_centers: list
    :return: Vertices, edges and cell's list
    :rtype: tuple
    """
    # if kwargs.get("add_layer", False):
    #     centroid = np.mean(cell_centers)
        
    tessellation = scspace.Voronoi(cell_centers)

    new_vertices = {}
    new_cells = {}
    new_edges = {}
    regions = deepcopy(tessellation.regions)
    big_edge = []

    cnum = 1

    regions = remove_infinite_regions(tessellation, regions, max_distance=kwargs.get("max_distance", 75))

    for c in regions:
        temp_for_cell = []
        temp_vertex_for_cell = []
        if len(c) != 0 and -1 not in c:
            # add first to close the cell_vertices
            c.append(c[0])
            for ii in range(0, len(c) - 1):
                temp_big_edge = []
                x_coordinate = np.around(np.linspace(round(tessellation.vertices[c[ii]][0], 3),
                                                    round(tessellation.vertices[c[ii + 1]][0], 3), 2), 3)
                y_coordinate = np.around(line_eq(tessellation.vertices[c[ii]],
                                                    tessellation.vertices[c[ii + 1]],
                                                    x_coordinate), 3)

                new_edge_vertices = list(zip(x_coordinate, y_coordinate))
                # add new edges to the global list
                for v in range(0, len(new_edge_vertices) - 1):
                    v0 = new_edge_vertices[v]
                    v1 = new_edge_vertices[v + 1]

                    vertex_number_1 = get_vertex_number(v0, new_vertices)

                    vertex_number_2 = get_vertex_number(v1, new_vertices)

                    enum = get_enum([vertex_number_1, vertex_number_2], new_edges)

                    temp_big_edge.append(enum)
                    temp_for_cell.append(enum)
                    temp_vertex_for_cell.append(vertex_number_1)
                    temp_vertex_for_cell.append(vertex_number_2)

                big_edge.append(temp_big_edge)

            area_sign = get_cell_area_sign(temp_vertex_for_cell, new_vertices)
            new_cells[-1 * cnum * area_sign] = temp_for_cell
            cnum += 1

    return new_vertices, new_edges, new_cells


def remove_infinite_regions(tessellation, regions: list, max_distance: float = 50) -> list:
    """
    Remove regions that have a vertex too far away to be part of the tissue. This solves vertices placed in
    'infinity' by the tessellation

    :param regions: List of the voronoi regions
    :type regions: list
    :param max_distance: Maximum distance of a vertex in a polygon
    :type max_distance: float, optional
    :return: List of regions without the offending ones
    :rtype: list
    """
    to_delete = []
    used_vertices = []
    for c in regions:
        if len(c) != 0 and -1 not in c:
            polygon_vertices = [list(tessellation.vertices[vid]) for vid in c]
            matrix = distance_matrix(polygon_vertices)
            if np.max(matrix) > max_distance:
                to_delete.append(c)
    for c in to_delete:
        regions.remove(c)
    return regions


def line_eq(p0: float, p1: float, x: np.ndarray) -> list:
        """
        Get the value of linear function that goes through p0 and p1 at x

        :param p0: First point to join
        :type p0: float
        :param p1: Second point to join
        :type p0: float
        :param x: Independent variable of the equation
        :type p0: list
        :return: Value of y(x) for a line that goes through p0 and p1
        """
        p0 = np.around(p0, 3)
        p1 = np.around(p1, 3)
        m = (p1[1] - p0[1]) / (p1[0] - p0[0])
        return p0[1] + m * (x - p0[0])


def get_cell_area_sign(cell: list, all_vertices: dict) -> int:
    """
    Get the orientation of the polygon for a cell_vertices using the sign of the area through gauss formula

    :param cell: Vertices of the cell_vertices to determine orientation
    :type cell: list
    :param all_vertices: Dictionary with all the vertices in the lattice
    :type all_vertices: dict
    :return: Sign of the cell_vertices area to determine orientation of the polygon
    :rtype: int
    """
    return int(np.sign(get_cell_area(cell, all_vertices)))


def get_cell_area(cell_vertices: list, vertices: dict) -> float:
    """
    Area of a polygon using the shoelace formula

    :param cell_vertices: List of vertex's id of the polygon
    :param vertices: Dictionary of vertices in the system
    :return: Area of the polygon. Positive number indicates clockwise orientation, negative number \
        counterclockwise.
    """
    x = [vertices[i][0] for i in cell_vertices]
    y = [vertices[i][1] for i in cell_vertices]
    return 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_vertex_number(vertex: list, vertices: dict) -> int:
    """
    Get id of the vertex from the vertex position or new possible ID if the vertex is already known

    :param vertex: Vertex position
    :type vertex: list
    :param vertices: Dictionary of the vertices currently identified
    :type vertices: dict
    :return: Vertex id if the vertex exists, if not the next possible id to assign
    :rtype: int
    """
    if vertex in vertices.values():
        vertex_number = list(vertices.keys())[list(vertices.values()).index(vertex)]
    else:
        if len(vertices) > 0:
            vertex_number = max(vertices.keys()) + 1
        else:
            vertex_number = 1
        vertices[vertex_number] = vertex
    return vertex_number

def get_enum(edge: list, edges: dict) -> int:
    """
    Get id of the edge from the vertex position or new possible ID if the edge is already known

    :param edge: id of the vertices that create the edge
    :type edge: list
    :param edges: Dictionary of the edges currently identified
    :type edges: dict
    :return: edge id if the edge exists, if not the next possible id to assign
    :rtype: int
    """
    if edge in edges.values():
        enum = list(edges.keys())[list(edges.values()).index(edge)]
    elif edge[::-1] in edges.values():
        enum = - list(edges.keys())[list(edges.values()).index(edge[::-1])]
    else:
        if len(edges) > 0:
            enum = max(edges.keys()) + 1
        else:
            enum = 1
        edges[enum] = [edge[0], edge[1]]
    return enum


def add_voronoi_centers(cell_centers):
    matrix = distance_matrix(cell_centers)
    ave_distance = np.median(matrix) / 10
    max_x = np.max(list(zip(*cell_centers))[0])
    min_x = np.min(list(zip(*cell_centers))[0])
    max_y = np.max(list(zip(*cell_centers))[1])
    min_y = np.min(list(zip(*cell_centers))[1])

    range_x = np.linspace(min_x - ave_distance, max_x + ave_distance, 10)
    range_y = np.linspace(min_y - ave_distance, max_y + ave_distance, 10)
    new_cell_centers = []
    for ii in range_x:
        new_cell_centers.append((ii, min_y - ave_distance))
        new_cell_centers.append((ii, max_y + ave_distance))
    for kk in range_y:
        new_cell_centers.append((min_x - ave_distance, kk))
        new_cell_centers.append((max_x + ave_distance, kk))
    
    return new_cell_centers


def distance_matrix(vertices: list) -> list:
    matrix = np.zeros((len(vertices), len(vertices)))
    for (element_1, element_2) in itert.product(vertices, repeat=2):
        ii = vertices.index(element_1)
        jj = vertices.index(element_2)
        distance_element = np.array(element_1) - np.array(element_2)
        matrix[jj, ii] = np.linalg.norm(distance_element)
    return matrix