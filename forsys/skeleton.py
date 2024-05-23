from dataclasses import dataclass
import numpy as np
from PIL import Image
import cv2
import itertools as iter
from collections import Counter
from typing import Tuple

import forsys.vertex as vertex
import forsys.edge as edge
import forsys.cell as cell
import forsys.wkt as fwkt
import forsys.virtual_edges as fvedges

@dataclass
class Skeleton:
    """
    Reads and parses TIFF skekeletons using OpenCV implementation to detect polygons

    :param fname: Path to segmented image
    :type fname: str
    :param mirror_y: False by default. Inverts the y-axis if true.
    :type mirror_y: bool, optional
    """
    fname: str

    mirror_y: bool = False

    def __post_init__(self):
        with Image.open(self.fname).convert("L") as curr_image_file:
            self.img_arr = np.array(curr_image_file)[1:-1, 1:-1]

        # self.contours, _ = cv2.findContours(self.img_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours, _ = cv2.findContours(self.img_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.contours = [np.vstack(p).squeeze() for p in self.contours][1:]
        # check size of all areas and filter small ones
        all_cell_areas = [self.calculate_area(polygon) for polygon in self.contours]
        # if a cell has more than 5 times the average cell area (withouth the maximum)
        # remove it
        ave_cell_area = np.mean(np.sort(all_cell_areas)[:-1])
        self.contours = [polygon for polygon in self.contours if self.calculate_area(polygon) < 5 * ave_cell_area]
        # initialize the counters
        self.vertex_id = 0
        self.edge_id = 0
        self.cell_id = 0

        if self.mirror_y:
            self.max_y = float(max(list(iter.chain.from_iterable(
                [list(zip(*c))[1] for c in self.contours]))))

    def create_lattice(self, **kwargs) -> Tuple[dict, dict, dict]:
        """
        Creates vertices, edges and cells of the system in three dictioinaries

        :return: Vertices, edges and cells created from parsing the provided image
        :rtype: tuple
        """
        self.vertices = {}
        self.edges = {}
        self.cells = {}

        self.coords_to_key = {}

        self.edges_added = []

        rescale = kwargs.get("rescale", [1, 1])
        offset = kwargs.get("offset", [0, 0])

        for _, polygon in enumerate(self.contours):
            cell_vertices_list = []
            for coords in polygon:
                if self.mirror_y:
                    coords[1] = self.max_y - coords[1]
                vid = self.get_vertex_id_by_position(coords)

                if vid == -1:
                    vid = self.vertex_id
                    self.vertices[vid] = vertex.Vertex(int(vid),
                                                       (coords[0] - offset[0]) / rescale[0],
                                                       (coords[1] - offset[1]) / rescale[1])
                    self.vertex_id += 1

                    self.coords_to_key[tuple(coords)] = vid
                    # add vertex to this cell's list
                cell_vertices_list.append(self.vertices[vid])

            # join vertices with the edges
            number_of_vertices = len(polygon)
            for n in range(1, number_of_vertices):
                self.create_edge(n - 1, n, polygon)
            # Join last vertex with first
            self.create_edge(n, 0, polygon)

            # create cells now
            self.cells[self.cell_id] = cell.Cell(self.cell_id, cell_vertices_list, {})
            self.cell_id += 1
        
        if kwargs.get("reduce_amount", False):
            self.vertices, self.edges, self.cells = fwkt.reduce_amount(self.vertices,
                                                                    self.edges,
                                                                    self.cells)

        self.all_big_edges = fvedges.create_edges_new(self.vertices,
                                                      self.cells)

        for current_cell in self.cells.values():
            # if any of the vertices in the cell has a vertex with only one cell, that is an external cell
            if np.any([len(v.ownCells) == 1 for v in current_cell.vertices]):
                current_cell.is_border = True
        for e in self.edges.values():
            # if at least one of the vertices have only one cell, this is an external edge
            if len(e.v1.ownCells) == 1 or len(e.v2.ownCells) == 1:
                e.external = True
            else:
                e.external = False

        # triangles in the middle
        inner_edge_triangles = []
        num_elements = len(self.all_big_edges)
        first_part = [(e[0], e[-1]) for e in self.all_big_edges]
        last_part = [(e[-1], e[0]) for e in self.all_big_edges]
        first_last = first_part + last_part
        inner_edge_triangles = [k for k, v in Counter(first_last).items() if v > 1]
        
        visited = []
        for index in range(len(inner_edge_triangles) - 1):
            if (inner_edge_triangles[index] in visited or
                inner_edge_triangles[index][::-1] in visited):
                continue

            edge_0_index = first_last.index(inner_edge_triangles[index]) % num_elements
            edge_1_index = first_last.index(inner_edge_triangles[(index + 1)]) % num_elements
            edge_0 = self.all_big_edges[edge_0_index]
            if len(edge_0) > 3:
                continue
            edge_1 = self.all_big_edges[edge_1_index]
            vertex_id_to_delete = np.setdiff1d(edge_0, edge_1)[0]

            its_cells = self.vertices[vertex_id_to_delete].ownCells
            for cell_id in its_cells:
                self.cells[cell_id].replace_vertex(self.vertices[vertex_id_to_delete],
                                                   self.vertices[edge_0[0]])

            its_edges = self.vertices[vertex_id_to_delete].ownEdges
            for edge_id in its_edges:
                del self.edges[edge_id]

            del self.vertices[vertex_id_to_delete]
            visited.append(inner_edge_triangles[index])

        # get artifacts from the contour and apply T3 transitions to them
        all_artifact_vertices = list(self.get_artifacts())

        artifacts = []
        while len(all_artifact_vertices) != 0:
            current_artifact = [all_artifact_vertices[0]]
            current_len0 = -1
            while current_len0 != len(current_artifact):
                current_artifact = self.add_vertices_to_current(all_artifact_vertices, current_artifact)
                current_len0 = len(current_artifact)
            # remove vertices already in artifact from all_artifact_vertices
            for ii in current_artifact:
                all_artifact_vertices.remove(ii)

            artifacts.append(current_artifact)

        # now make a T3 transition for each triangle:
        for artifact in artifacts:
            self.do_t3_transition(artifact)

        # if all vertices in a cell only have that cell as own, delete cell
        cells_to_remove = []
        for c in self.cells.values():
            vertices_le_one = [len(v.ownCells) <= 1 for v in c.vertices]

            if np.all(vertices_le_one):
                cells_to_remove.append(c.id)
                for v in c.vertices:
                    # remove edges
                    for e in v.ownEdges:
                        del self.edges[e]
                    # remove vertices
                    try:
                        del self.vertices[v.id]
                    except KeyError:
                        print(f"**WARNING** vertex {v.id} not in list of vertices")
        # remove the cell
        for cid in cells_to_remove:
            del self.cells[cid]

        return self.vertices, self.edges, self.cells

    def add_vertices_to_current(self, artifact_vertices: list, current_artifact: list) -> list:
        """
        Creates the list of vertices of a given artifact
        :param artifact_vertices: All vertices belonging to artifacts
        :type artifact_vertices: list
        :param current_artifact: List to append the vertices of the current artifact
        :type current_artifact: list

        :return: Vertices, edges and cells created from parsing the provided image
        :rtype: list
        """
        v0 = current_artifact[-1]
        for eid in self.vertices[v0].ownEdges:
            other_vertex = self.edges[eid].get_other_vertex_id(v0)
            if other_vertex in artifact_vertices and other_vertex not in current_artifact:
                current_artifact.append(other_vertex)
        return current_artifact

    def do_t3_transition(self, artifact: list):
        """
        Perform a T3 transition in all the identified artifacts

        :param artifact: List with all the artifacts detected in the system
        :type artifact: list
        :return: None
        """
        # get the mid point and create a vertex there
        ave_x = np.mean([self.vertices[ii].x for ii in artifact])
        ave_y = np.mean([self.vertices[ii].y for ii in artifact])

        new_vid = self.get_new_vid()
        self.vertices[new_vid] = vertex.Vertex(int(new_vid),
                                               ave_x,
                                               ave_y)

        # get the edge and join them to the new point)
        for v in artifact:
            edge_to_remove = []
            edge_to_replace = []
            for e in self.vertices[v].ownEdges:
                if (self.edges[e].v1.id in artifact and self.edges[e].v2.id in artifact) or \
                        (self.edges[e].v2.id in artifact and self.edges[e].v1.id in artifact):
                    edge_to_remove.append(e)
                else:
                    edge_to_replace.append(e)

            for ii in range(0, len(edge_to_remove)):
                del self.edges[edge_to_remove[ii]]

            for ii in range(0, len(edge_to_replace)):
                edge_object = self.edges[edge_to_replace[ii]]
                # if new_vid != edge_object.v1.id and new_vid != edge_object.v2.id:
                edge_object.replace_vertex(self.vertices[v], self.vertices[new_vid])

            # add the cells corresponding
            for c in self.vertices[v].ownCells:
                # cell_vertex_ids = [v.id for v in self.cells[c]]
                self.cells[c].replace_vertex(self.vertices[v], self.vertices[new_vid])

        for jj in range(0, len(artifact)):
            if len(self.vertices[artifact[jj]].ownEdges) == 0:
                del self.vertices[artifact[jj]]
            else:
                print("WARNING: BAD VERTEX DELETION IN TRIANGLE - MORE THAN ZERO EDGE")

    def get_artifacts(self) -> list:
        """
        Create list with all the artifacts detected in the processing of the tissue

        :return: List of vertices belonging to artifacts
        :rtype: list
        """
        # triangles with TJs
        candidate_artifact_vertices = []
        for v in self.vertices.values():
            if len(v.ownEdges) == 3 and len(v.ownCells) == 2:
                # this is part of a triangle
                candidate_artifact_vertices.append(v.id)

        # list of external vertices:
        external_vertices = []
        for e in self.edges.values():
            if e.external:
                external_vertices.append(e.v1.id)
                external_vertices.append(e.v2.id)
        artifact_vertices = np.setdiff1d(candidate_artifact_vertices, external_vertices)

        return artifact_vertices

    def create_edge(self, n0: int , n1: int, polygon: list):
        """
        Add tuples of vertices identified by the vertices that from them and create the SmallEdge object

        :param n0: ID in the polygon list of the first vertex
        :type n0: int
        :param n1: ID in the polygon list of the second vertex
        :type n1: int
        :param n0: List of vertices of the current polygon
        :type polygon: int
        """
        # TODO: Functionalize function.
        v1 = self.vertices[self.get_vertex_id_by_position(polygon[n0])]
        v2 = self.vertices[self.get_vertex_id_by_position(polygon[n1])]
        if not (v1.id, v2.id) in self.edges_added and not (v2.id, v1.id) in self.edges_added:
            self.edges[self.edge_id] = edge.SmallEdge(self.edge_id, v1, v2)
            self.edge_id += 1
            self.edges_added.append((v1.id, v2.id))

    def get_vertex_id_by_position(self, coordinates: list) -> int:
        """
        Check whether vertices were already created using a hashmap
        
        :param coordinates: tuple with the coordiantes of the vertex to check
        :type coordinates: list

        :return: Current key of the given coordinates or -1 if there are no vertices at that position
        :rype: int
        """
        try:
            return self.coords_to_key[tuple(coordinates)]
        except KeyError:
            return -1

    def get_new_vid(self) -> int:
        """
        Returns the best possible ID for the creation of a new vertex

        :return: unused ID in the vertices dictionary
        :rtype: int
        """
        new_vid = max(self.vertices.keys()) + 1
        while new_vid in self.vertices.keys():
            new_vid += 1
        return new_vid

    @staticmethod
    def calculate_area(vertices: list) -> float:
        """
        Calculate the area of a given polygon using the shoelace formula

        :param vertices: array of [x,y] positions of the ordered vertices in a polygon
        :type vertices: list
        :return: Area of the polygon described by the vertices
        :rtype: float
        """
        x = [i[0] for i in vertices]
        y = [i[1] for i in vertices]
        return abs(0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    @staticmethod
    def get_distance(v1: vertex.Vertex, v2: vertex.Vertex) -> float:
        """
        Get the distance between two vertices in the system

        :param v1: First vertex
        :type v1: vertex.Vertex
        :param v2: Second vertex
        :type v2: vertex.Vertex
        :return: Distance between the vertices
        :rtype: float
        """
        return np.linalg.norm(np.array([v1.x, v1.y]) - np.array([v2.x, v2.y]))
