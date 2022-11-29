from dataclasses import dataclass
import numpy as np

@dataclass 
class Cell:
    id: int
    vertices: list
    ctype: dict
    is_border: bool = False


    def __post_init__(self):
        # check for repeated elements
        # vids = [v.id for v in self.vertices]
        # assert len(vids) == len(set(vids)), f"There are repeated vertices in cell {self.id}"
        for v in self.vertices:
            v.add_cell(self.id)
        
    def __del__(self):
        for v in self.vertices:
            v.remove_cell(self.id)
        
    def get_cell_vertices(self) -> list:
        return self.vertices

    def get_next_vertex(self, v: object) -> object:
        return self.vertices[(self.vertices.index(v) + self.get_area_sign()) % len(self.vertices)]
    
    def get_previous_vertex(self, v: object) -> object:
        return self.vertices[(self.vertices.index(v) - self.get_area_sign())% len(self.vertices)]
        
    def get_cm(self) -> list:
        return [np.mean([v.x for v in self.vertices]), np.mean([v.y for v in self.vertices])]

    def get_area_sign(self) -> int:
        """
        x and y are the arrays with Xs and Ys of the vertices
        """
        x = [i.get_coords()[0] for i in self.vertices]
        y = [i.get_coords()[1] for i in self.vertices]

        return int(np.sign(self.get_area()))

    def get_area(self) -> float:
        x = [i.get_coords()[0] for i in self.vertices]
        y = [i.get_coords()[1] for i in self.vertices]
        return 0.5 * (np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x, 1)))

    def get_perimeter(self) -> float:
        perimeter = 0
        nVertex = len(self.vertices)
        for i in self.vertices:
            diffx = i.x - self.get_next_vertex(i).x
            diffy = i.y - self.get_next_vertex(i).y
            perimeter +=  np.sqrt(diffx**2 + diffy**2)
        return perimeter

    def replace_vertex(self, vold: object, vnew: object):
        """
        Replace vertex vold with vnew in the vertices list and
        add this cell to that vertex. If the vertex is already
        in the cell, remove it.
        """
        vertices_ids = [v.id for v in self.vertices]
        if vnew.id in vertices_ids:
            self.vertices.remove(vold)
        else:
            self.vertices[vertices_ids.index(vold.id)] = vnew
            # add cell to vertex
            vnew.add_cell(self.id)
