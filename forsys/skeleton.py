from dataclasses import dataclass
from hashlib import algorithms_available
import numpy as np
from PIL import Image
import cv2

import forsys.vertex as vertex
import forsys.edge as edge
import forsys.cell as cell


@dataclass
class Skeleton:
    fname: str

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
        # areas = [self.calculate_area(polygons) for polygons in self.contours)]
        self.contours = [polygon for polygon in self.contours if self.calculate_area(polygon) < 5 * ave_cell_area]
        # initialize the counters
        self.vertex_id = 0
        self.edge_id = 0
        self.cell_id = 0
    
    def create_lattice(self):
        self.vertices = {}
        self.edges = {}
        self.cells = {}

        # create vertices
        self.edges_added = []
        for polygon in self.contours:
            # create all vertices
            # xcoords = [(coords[0] > 0) and (coords[0] < 700) for coords in polygon]
            # if not np.all(xcoords):
            #     continue
            cell_vertices_list = []
            for coords in polygon:
                vid = self.get_vertex_id_by_position(coords)

                if vid == -1:
                    vid = self.vertex_id

                    self.vertices[vid] = vertex.Vertex(int(vid), 
                                                        coords[0], 
                                                        coords[1])
                    self.vertex_id += 1

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

        # get artifacts from the contour and apply T3 transitions to them
        all_artifact_vertices = list(self.get_artifacts())
        # all_triangle_vertices_remaining = all_triangle_vertices.copy()
        # max_number_of_triangles = len(all_artifact_vertices) // 3
        # 5 px of distance maximum
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
        v0 = current_artifact[-1]
        for eid in self.vertices[v0].ownEdges:
            other_vertex = self.edges[eid].get_other_vertex_id(v0)
            if other_vertex in artifact_vertices and other_vertex not in current_artifact:
                current_artifact.append(other_vertex)
        return current_artifact            


    def do_t3_transition(self, artifact : list):
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
                print("WARNING: BAD VERTEX DELETION IN TRINANGLE - MORE THAN ZERO EDGE")


    def get_artifacts(self) -> list:
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
        # print("Artifact triangular:", candidate_triangular_vertices, external_vertices)
        artifact_vertices = np.setdiff1d(candidate_artifact_vertices, external_vertices)

        return artifact_vertices




    def create_edge(self, n0, n1, polygon):
        v1 = self.vertices[self.get_vertex_id_by_position(polygon[n0])]
        v2 = self.vertices[self.get_vertex_id_by_position(polygon[n1])]
        if not (v1.id, v2.id) in self.edges_added and not (v2.id, v1.id) in self.edges_added:
            self.edges[self.edge_id] = edge.Edge(self.edge_id, v1, v2)
            self.edge_id += 1
            self.edges_added.append((v1.id, v2.id))


    def get_vertex_id_by_position(self, coordinates: list) -> int:
        # Check all vertices to see if we already have one in that position
        # if the distance is =< than some value, it is the same vertex
        for _, vertex in self.vertices.items():
            if abs(coordinates[0] - vertex.x) <= 0 and abs(coordinates[1] - vertex.y) <= 0:
                return vertex.id
        return -1
    
    def get_new_vid(self) -> int:
        new_vid = max(self.vertices.keys()) + 1
        while new_vid in self.vertices.keys():
            new_vid += 1
        return new_vid

    @staticmethod
    def calculate_area(vertices: list) -> float:
        x = [i[0] for i in vertices]
        y = [i[1] for i in vertices]
        return abs(0.5 * (np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x, 1))))

    @staticmethod
    def get_distance(v1, v2):
        return np.linalg.norm(np.array([v1.x, v1.y]) - np.array([v2.x, v2.y]))

