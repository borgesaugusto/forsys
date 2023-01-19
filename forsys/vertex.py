from dataclasses import dataclass, field

@dataclass 
class Vertex:
    id: int
    x: float
    y: float
    ownEdges: list = field(default_factory=list)
    ownCells: list = field(default_factory=list)
    own_big_edges: list = field(default_factory=list)
    
    def get_coords(self) -> list:
        return [self.x, self.y]

    def add_cell(self, cid: int) -> bool:
        if cid in self.ownCells:
            return False
        else:
            self.ownCells.append(cid)
            return True
        
    def remove_cell(self, cid: int) -> list:
        self.ownCells.remove(cid)
        return self.ownCells

    def add_edge(self, eid: int) -> bool:
        if eid in self.ownEdges:
            print(eid, self.ownEdges)
            print("edge already in vertex")
            return False
        else:
            self.ownEdges.append(eid)
            return True
    
    def remove_edge(self, eid: int):
        self.ownEdges.remove(eid)

    def add_big_edge(self, beid: int) -> bool:
        if beid in self.own_big_edges:
            print(beid, self.own_big_edges)
            print("edge already in vertex")
            return False
        else:
            self.own_big_edges.append(beid)
            return True
    
    def remove_big_edge(self, beid: int):
        self.own_big_edges.remove(beid)