from dataclasses import dataclass
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
        self.curr_image_file = Image.open(self.fname).convert("L")
        self.img_arr = np.array(self.curr_image_file)[1:-1, 1:-1]

        self.contours, _ = cv2.findContours(self.img_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # initialize the counters
        self.vertex_id = 0
        self.edge_id = 0
        self.cell_id = 0
    
    def create_lattice(self):
        self.vertices = {}
        self.edges = {}
        self.cells = {}

        # create vertices
        for polygon in self.contours:
            # create all vertices
            cell_vertices_list = []
            for coords in polygon:
                vid = self.get_vertex_id_by_position(coords[0])

                if vid == -1:
                    vid = self.vertex_id

                    self.vertices[vid] = vertex.Vertex(int(vid), 
                                                        coords[0][0], 
                                                        coords[0][1])
                    self.vertex_id += 1        
                # add vertex to this cell's list
                cell_vertices_list.append(self.vertices[vid])

            # join vertices with the edges
            number_of_vertices = len(polygon)
            for n in range(1, number_of_vertices):
                v1 = self.vertices[self.get_vertex_id_by_position(polygon[n-1][0])]
                v2 = self.vertices[self.get_vertex_id_by_position(polygon[n][0])]

                self.edges[self.edge_id] = edge.Edge(self.edge_id, v1, v2)
                self.edge_id += 1

            # create cells now
            self.cells[self.cell_id] = cell.Cell(self.cell_id, cell_vertices_list, {})
            self.cell_id += 1
        
        return self.vertices, self.edges, self.cells

    def get_vertex_id_by_position(self, coordinates: list) -> int:
        # Check all vertices to see if we already have one in that position
        for _, vertex in self.vertices.items():
            if float(coordinates[0]) == float(vertex.x) and float(coordinates[1]) == float(vertex.y):
                return vertex.id
        return -1