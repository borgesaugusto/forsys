from dataclasses import dataclass

@dataclass 
class Edge:
    id: int
    v1: object
    v2: object

    def __post_init__(self):
        self.verticesArray = [self.v1, self.v2]
        for v in self.verticesArray:
            v.add_edge(self.id)

    def get_vertices_id(self) -> list:
        # return map(lambda x: x.id, [self.v1, self.v2])
        return [self.v1.id, self.v2.id]

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
