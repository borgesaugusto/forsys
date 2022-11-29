from dataclasses import dataclass 

@dataclass 
class SmallEdge():
    id: int     
    v1: object
    v2: object
    tension: float=0
    gt: float=0

    def __post_init__(self):
        super().__init__()
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

    def replace_vertex(self, vold: object, vnew:object):
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