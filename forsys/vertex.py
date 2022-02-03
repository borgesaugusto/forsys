from dataclasses import dataclass, field

@dataclass 
class Vertex:
    id: int
    x: float
    y: float
    ownEdges: list = field(default_factory=list)
    ownCells: list = field(default_factory=list)
    
    def get_coords(self) -> list:
        return [self.x, self.y]

    def add_cell(self, cid: int) -> bool:
        if cid in self.ownCells:
            return False
        else:
            self.ownCells.append(cid)
            return True
        
    def add_edge(self, eid: int) -> bool:
        if eid in self.ownEdges:
            print("edge already in vertex")
            return False
        else:
            self.ownEdges.append(eid)
            return True
    
    # def remove_edge(self, eid: int):
    #     self.ownEdges.remove(eid)
