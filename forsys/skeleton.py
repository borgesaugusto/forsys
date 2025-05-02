from dataclasses import dataclass
import itertools as iter
from collections import Counter
from typing import Tuple
from warnings import warn
import scipy.spatial as spspatial
import skimage as skimage
import collections
import numpy as np
import cv2
from PIL import Image

import forsys.vertex as vertex
import forsys.edge as edge
import forsys.cell as cell
import forsys.wkt as fwkt
import forsys.virtual_edges as fvedges

import forsys as fs

import matplotlib.pyplot as plt

@dataclass
class Skeleton:
    """
    Reads and parses TIFF skeletons or CellPose npy files
    using OpenCV implementation to detect polygons

    :param fname: Path to segmented image
    :type fname: str
    :param mirror_y: False by default. Inverts the y-axis if true.
    :type mirror_y: bool, optional
    """
    fname: str
    mirror_y: bool = False
    minimum_distance: float = 0
    expand: int = 0

    def __post_init__(self):
        # TODO: Unify NPY and TIF, as they use similar OpenCV methods
        if self.fname.endswith(".tif") or self.fname.endswith(".tiff"):
            self.format = "tif"
            with Image.open(self.fname).convert("L") as curr_image_file:
                self.img_arr = np.array(curr_image_file)[1:-1, 1:-1]

            self.contours, _ = cv2.findContours(self.img_arr,
                                                cv2.RETR_TREE,
                                                # cv2.CHAIN_APPROX_SIMPLE,
                                                cv2.CHAIN_APPROX_NONE)
            self.contours = [np.vstack(p).squeeze() for p in self.contours][1:]
        elif self.fname.endswith(".npy"):
            self.format = "npy"
            np_data = np.load(self.fname,
                              allow_pickle=True).item()
            masks = np_data["masks"]
            border_id = -1
            if self.expand > 0:
                # add an additional mask of everything
                border_contour = skimage.measure.label(masks != 0)
                border_expand = skimage.segmentation.expand_labels(border_contour,
                                                                  distance=2)
                binary_mask_border = border_expand == 1
                polygon_contour, _ = cv2.findContours(binary_mask_border.astype(np.uint8),
                                                 cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_NONE)
                which = np.argmax([c.shape[0] for c in polygon_contour])
                border_polygon = polygon_contour[which].astype(int).squeeze()

                border_id = np.max(masks) + 1
                for element in border_polygon:
                    masks[element[1], element[0]] = border_id

                masks = skimage.segmentation.expand_labels(masks,
                                                           distance=self.expand)

            self.contours = []
            for cell_n in np.unique(masks)[1:]:
                if cell_n == border_id:
                    continue
                binary_mask = masks == cell_n
                if binary_mask.sum() > 0:
                    current_contour, _ = cv2.findContours(binary_mask.astype(np.uint8),
                                                     cv2.RETR_TREE,
                                                     cv2.CHAIN_APPROX_NONE)

                # assert len(current_contour) == 1, f"More than one contour found for cell {cell_n}"
                which = np.argmax([c.shape[0] for c in current_contour])
                to_add = current_contour[which].astype(int).squeeze()
                self.contours.append(to_add)
        else:
            raise ValueError("File format not supported")

        # initialize the counters
        self.vertex_id = 0
        self.edge_id = 0
        self.cell_id = 0

        if self.mirror_y:
            warn("Legacy function, soon to be deprecated", DeprecationWarning, 2)
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

        if kwargs.get("max_cell_size", 0) != 0:
            all_cell_areas = [self.calculate_area(polygon)
                             for polygon in self.contours]
            ave_cell_area = np.mean(np.sort(all_cell_areas)[:-1])
            self.contours = [polygon for polygon in self.contours
                             if self.calculate_area(polygon) <
                             kwargs.get("max_cell_size") * ave_cell_area]

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
            warn("Legacy function, soon to be deprecated", DeprecationWarning, 2)
            self.vertices, self.edges, self.cells = fwkt.reduce_amount(self.vertices,
                                                                    self.edges,
                                                                    self.cells)


        # self.plot_here("pre_map")
        if self.minimum_distance > 0:
            new_vertices = {}
            new_edges = {}
            map_mains_to_nn = {}
            to_connect = {}
            cells_vertices = {}
            new_vid = 0
            # new_eid = 0
            created_edges = []
            # TODO: It would be more efficient to use a dynamic KDtree
            self.main_vertices_map = {}
            all_points = np.array([[v.x, v.y] for v in self.vertices.values()])
            self.kdtree = spspatial.KDTree(all_points)
            already_mapped = set()
            for vid, vobj in self.vertices.items():
                if vid in already_mapped:
                    continue
                # get all neighbors in the given distance, and join them to vid
                # in one vertex
                neighbors = self.kdtree.query_ball_point([vobj.x, vobj.y],
                                                         self.minimum_distance)
                self.main_vertices_map[vid] = None
                new_vertex_x = np.mean([self.vertices[nn].x for nn in neighbors])
                new_vertex_y = np.mean([self.vertices[nn].y for nn in neighbors])
                new_vertices[new_vid] = vertex.Vertex(int(new_vid),
                                                      new_vertex_x,
                                                      new_vertex_y)
                map_mains_to_nn[new_vid] = neighbors
                # Check cell inside the bubble
                cells_in = np.unique(np.array([self.vertices[vid].ownCells for vid in neighbors]).reshape(-1))
                for cell_id in cells_in:
                    if cell_id not in cells_vertices.keys():
                        cells_vertices[cell_id] = []
                    cells_vertices[cell_id].append(new_vid)
                # identify which are the edges connected to vertices outside
                # the bubble
                vertices_to_connect = [nn for nn in neighbors
                                       if nn in to_connect.keys()]
                if vid in to_connect.keys() and vid not in vertices_to_connect:
                    vertices_to_connect.append(vid)
                for vconnect_id in vertices_to_connect:
                    if vconnect_id in already_mapped:
                        continue
                    new_edges, created_id = self.create_edge_mindis(to_connect[vconnect_id],
                                                                    new_vid,
                                                                    new_vertices,
                                                                    new_edges,
                                                                    created_edges)
                for nn in neighbors:
                    if nn in already_mapped:
                        continue
                    for e_id in self.vertices[nn].ownEdges:
                        both_ids = self.edges[e_id].get_vertices_id()
                        if (both_ids[0] not in neighbors or
                           both_ids[1] not in neighbors) and \
                           vid not in both_ids and \
                           nn not in to_connect.keys():
                            # Connected outside the bubble
                            other = self.edges[e_id].get_other_vertex_id(nn)
                            to_connect[other] = new_vid
                new_vid += 1
                already_mapped.update(neighbors)
                already_mapped.add(vid)
            self.cells = {}
            for cell_id, vertices in cells_vertices.items():
                self.cells[cell_id] = cell.Cell(cell_id,
                                                [new_vertices[v] for v in vertices])

            self.vertices = new_vertices
            self.edges = new_edges
            print("To connect: ", to_connect)
            print("Number of edges: ", len(self.edges))
            self.plot_here("after_mapping")

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

        # self.all_big_edges = fvedges.create_edges_new(self.vertices,
        #                                               self.cells)

        self.triangular_holes()
        self.remove_artifacts()

        return self.vertices, self.edges, self.cells

    def plot_here(self, name):
        for v in self.vertices.values():
            plt.scatter(v.x, v.y, s=0.5, color="blue")
            plt.annotate(str(v.id), [v.x, v.y], fontsize=2)

        for e in self.edges.values():
            plt.plot([e.v1.x, e.v2.x],
                     [e.v1.y, e.v2.y],
                     color="black",
                     linewidth=0.5,
                     alpha=0.6)

        for c in self.cells.values():
            cm = c.get_cm()
            cxs = [v.x for v in c.vertices]
            cys = [v.y for v in c.vertices]

            plt.fill(cxs, cys, alpha=0.25)
            plt.annotate(str(c.id), [cm[0], cm[1]], fontsize=5)

        plt.tight_layout()
        plt.savefig(f"../res_refactor/{name}.png", dpi=600)
        plt.close()

    @staticmethod
    def create_edge_mindis(v1_id: int,
                           v2_id: int,
                           new_vertices: dict,
                           new_edges: dict,
                           created_edges: list) -> Tuple[dict, int]:
        e_vert = [v1_id, v2_id]
        try:
            new_eid = max(new_edges.keys()) + 1
        except ValueError:
            new_eid = 0
        if frozenset(e_vert) in created_edges:
            # print(f"Edge {e_vert} already created,"
            #       f"joining {v1_id} and {v2_id}")
            return new_edges, -1
        new_edges[new_eid] = edge.SmallEdge(new_eid,
                                            new_vertices[e_vert[0]],
                                            new_vertices[e_vert[1]])
        created_edges.append(set(e_vert))
        return new_edges, new_eid

    def stragglers(self):
        # if there are vertices with only one connection,
        # join them to the nearset and disappear
        ...

    def triangular_holes(self):
        vertices_of_triangles = {}
        for v in self.vertices.values():
            if len(v.ownEdges) >= 3:
                # this is could be a triangular hole
                # is it connected to two other TJ with only two cells ?
                cells_per_nn = self.is_vertex_in_hole(v)
                if len(cells_per_nn) > 0:
                    # if all are TJs, join in one point
                    vertices_of_triangles[v.id] = list(cells_per_nn.keys())
        # see if key is in at least 2 others
        appearances = {}
        for key, nns in vertices_of_triangles.items():
            appeard_in = []
            for key_2, nns2 in vertices_of_triangles.items():
                if key in nns2 and key != key_2:
                    appeard_in.append(key_2)
            appearances[key] = appeard_in
        triangles = []
        middle_edge = []
        for who, where in appearances.items():
            triangle = []
            if len(where) >= 2:
                triangle = sorted([who] + where)
                if triangle not in triangles:
                    triangles.append(triangle)
            elif len(where) == 0:
                # this is triangle in the middle of an edge
                middle_edge.append(who)
        # remove the one that are in the middle of the edge
        for middle in middle_edge:
            for ee in self.vertices[middle].ownEdges:
                vertices_ids = self.edges[ee].get_vertices_id()
                vertex_to_delete = vertices_ids[0] if vertices_ids[0] != middle else vertices_ids[1]
                if len(self.vertices[vertex_to_delete].ownCells) == 1:
                    # this is the one
                    break

            for cell_id in self.vertices[vertex_to_delete].ownCells:
                self.cells[cell_id].replace_vertex(self.vertices[vertex_to_delete],
                                                   self.vertices[middle])
            its_edges = list(self.vertices[vertex_to_delete].ownEdges)
            for edge_id in its_edges:
                del self.edges[edge_id]

            del self.vertices[vertex_to_delete]

        # Join the vertices in the triangles in the centroid
        for triangle in triangles:
            # get the centroid
            centroid = np.mean([self.vertices[v].get_coords() for v in triangle], axis=0)
            new_vid = self.get_new_vid()
            self.vertices[new_vid] = vertex.Vertex(int(new_vid),
                                                   centroid[0],
                                                   centroid[1])
            # replace the vertices with the new one
            for v in triangle:
                its_edges = list(self.vertices[v].ownEdges)
                for edge_id in its_edges:
                    if self.edges[edge_id].get_vertices_id()[0] in triangle and \
                        self.edges[edge_id].get_vertices_id()[1] in triangle:
                        del self.edges[edge_id]
                    else:
                        self.edges[edge_id].replace_vertex(self.vertices[v],
                                                           self.vertices[new_vid])
                for cell_id in self.vertices[v].ownCells:
                    self.cells[cell_id].replace_vertex(self.vertices[v],
                                                       self.vertices[new_vid])
                del self.vertices[v]

    def is_vertex_in_hole(self, v: fs.vertex.Vertex) -> dict:
        connected_edges_vertices = [self.edges[e].get_vertices_id() for e in v.ownEdges]
        connected_to = np.array(connected_edges_vertices).reshape(-1)
        connected_to = list(filter(lambda x: x != v.id, connected_to))
        # do at least two of them have only two cells?
        cells_per_nn = {vv: self.vertices[vv] for vv in connected_to}
        leq_2 = len(list(filter(lambda x: len(x.ownCells) <= 2, cells_per_nn.values())))
        eq_1 = len(list(filter(lambda x: len(x.ownCells) == 1, cells_per_nn.values())))
        if leq_2 >= 2 and eq_1 != 2 and len(v.ownCells) == 2:
            return cells_per_nn
        else:
            return {}

    def get_cell_intersection(self, vobj) -> list:
        connected_edges_vertices = [self.edges[e].get_vertices_id() for e in vobj.ownEdges]
        connected_to = np.unique(np.array(connected_edges_vertices).reshape(-1))
        cells_belonging = [set(self.vertices[v].ownCells) for v in connected_to]
        intersection = set.intersection(*cells_belonging)
        return list(intersection)

    # def triangles_in_the_middle(self) -> None:
    #     # triangles in the middle
    #     inner_edge_triangles = []
    #     num_elements = len(self.all_big_edges)
    #     first_part = [(e[0], e[-1]) for e in self.all_big_edges]
    #     last_part = [(e[-1], e[0]) for e in self.all_big_edges]
    #     first_last = first_part + last_part
    #     inner_edge_triangles = [k for k, v in Counter(first_last).items() if v > 1]
    #
    #     visited = []
    #     for index in range(len(inner_edge_triangles) - 1):
    #         if (inner_edge_triangles[index] in visited or
    #             inner_edge_triangles[index][::-1] in visited):
    #             continue
    #
    #         edge_0_index = first_last.index(inner_edge_triangles[index]) % num_elements
    #         edge_1_index = first_last.index(inner_edge_triangles[(index + 1)]) % num_elements
    #         edge_0 = self.all_big_edges[edge_0_index]
    #         if len(edge_0) > 3:
    #             continue
    #         edge_1 = self.all_big_edges[edge_1_index]
    #         vertex_id_to_delete = list(set(edge_0).symmetric_difference(set(edge_1)))[0]
    #         # vertex_id_to_delete = np.intersect1d(edge_0, edge_1)[0]
    #
    #         its_cells = self.vertices[vertex_id_to_delete].ownCells
    #         for cell_id in its_cells:
    #             self.cells[cell_id].replace_vertex(self.vertices[vertex_id_to_delete],
    #                                                self.vertices[edge_0[0]])
    #
    #         its_edges = list(self.vertices[vertex_id_to_delete].ownEdges)
    #         for edge_id in its_edges:
    #             del self.edges[edge_id]
    #
    #         del self.vertices[vertex_id_to_delete]
    #         visited.append(inner_edge_triangles[index])
    #     print(f"Deleted {len(visited)} triangles in the middle of the system")

    def remove_artifacts(self) -> None:
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
            if len(v.ownEdges) > len(v.ownCells):
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
