from dataclasses import dataclass, field
import numpy as np


@dataclass
class SmallEdge:
    id: int
    v1: object
    v2: object
    tension: float = 0
    gt: float = 0

    def __post_init__(self):
        self.verticesArray = [self.v1, self.v2]
        for v in self.verticesArray:
            v.add_edge(self.id)
        assert self.v1.id != self.v2.id, f"edge {self.id} with the same vertex twice"

    def __del__(self):
        for v in self.verticesArray:
            v.remove_edge(self.id)

    def get_vertices_id(self) -> list:
        # return map(lambda x: x.id, [self.v1, self.v2])
        return [self.v1.id, self.v2.id]

    def get_other_vertex_id(self, one) -> int:
        assert one in self.get_vertices_id(), "vertex not in asked edge"
        return self.v1.id if one == self.v2.id else self.v2.id

    def get_vertices_array(self) -> list:
        return [self.v1, self.v2]

    def replace_vertex(self, vold: object, vnew: object):
        who = 0 if self.v1.id == vold.id else 1
        self.verticesArray[who].ownEdges.remove(self.id)
        if who == 0:
            self.v1 = vnew
        else:
            self.v2 = vnew
        self.verticesArray[who] = vnew
        self.verticesArray[who].add_edge(self.id)

    def get_vector(self) -> list:
        vector = [self.v2.x - self.v1.x, self.v2.y - self.v1.y]
        # norm = np.linalg.norm(vector)
        return vector


@dataclass
class BigEdge:
    big_edge_id: int
    vertices: list

    tension: float = 0.0
    gt: float = 0.0
    external: bool = False

    def __post_init__(self):
        for vertex in self.vertices:
            vertex.add_big_edge(self.big_edge_id)

        self.xs = [vertex.x for vertex in self.vertices]
        self.ys = [vertex.y for vertex in self.vertices]
        self.edges = [list(set(self.vertices[vid].ownEdges) &
                           set(self.vertices[vid + 1].ownEdges))[0]
                      for vid in range(0, len(self.vertices) - 1)]
        # check if I am external
        vertices_own_cells = [len(vertex.ownCells) < 2 for vertex in self.vertices]
        has_junction_at_extremes = True if len(self.vertices[0].ownCells) > 2 \
                                           or len(self.vertices[-1].ownCells) > 2 else False
        self.external = True if np.any(vertices_own_cells) or not has_junction_at_extremes \
            else False

    def calculate_curvature(self):
        dxdt = np.gradient(self.curve[:, 0])
        dydt = np.gradient(self.curve[:, 1])

        d2xdt2 = np.gradient(dxdt)
        d2ydt2 = np.gradient(dydt)
        curvature = np.abs(d2xdt2 * dydt - dxdt * d2ydt2) / (dxdt ** 2 + dydt ** 2) ** 1.5

        return curvature

    def get_circle_parameters(self):
        import scipy.optimize as sco
        # x = big_edge.xs
        # y = big_edge.ys
        # TODO: identificar residuos para ver si excluir

        def objective_f(c):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            distances = np.sqrt((self.xs - c[0]) ** 2 + (self.ys - c[1]) ** 2)
            return distances - distances.mean()

        center_2, _ = sco.leastsq(objective_f, (np.mean(self.xs), np.mean(self.ys)))
        return center_2[0], center_2[1]

    def get_versor_from_vertex(self, vid, method="edge", cell=None) -> list:
        vobject = self.get_vertex_object_by_id(vid)
        if method == "edge":
            xc, yc = self.get_circle_parameters()
        elif method == "cell" and cell:
            xc, yc = cell.center_x, cell.center_y
        else:
            raise Exception("Method for versor doesn't exist or cell missing")
        
        versor = np.array((- (vobject.y - yc), (vobject.x - xc)))
        versor = versor / np.linalg.norm(versor)
        # correct for the sign
        correct_sign = self.get_versor_sign(vid)
        if np.any(np.sign(versor) != correct_sign):
            correction = correct_sign * np.sign(versor)
            versor = versor * correction
        return versor

    def get_vertex_object_by_id(self, vid: int) -> object:
        vertex_object = list(filter(lambda x: x.id == vid, self.vertices))
        assert len(vertex_object) == 1, "More than one or no vertex with the same ID"
        return vertex_object[0]

    def get_vertices_ids(self) -> list:
        return [vertex.id for vertex in self.vertices]

    def get_versor_sign(self, vid: int):
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
        versor = [v1.x - v0.x, v1.y - v0.y]
        signs = np.sign(versor) 
        return [1.0 if sign == 0 else sign for sign in signs]
