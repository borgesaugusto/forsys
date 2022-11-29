from dataclasses import dataclass, field
import pandas as pd
import forsys.time_series as ts
import forsys.solve as solver
import forsys.frames as fsframes
@dataclass
class ForSys():
    frames: dict

    cm: bool=True
    initial_guess: list = field(default_factory=list)

    def __post_init__(self):
        # if there is more than 1 time, generate time series
        if len(self.frames) > 1:
            self.mesh = ts.TimeSeries(self.frames, cm=self.cm, 
                                        initial_guess=self.initial_guess)
            
            self.times_to_use = self.mesh.times_to_use()

    def solve(self, when: int, term: str, time_series:dict = None, metadata:dict = {}) -> None:
        res = solver.equations(self.frames[when].vertices, 
                                self.frames[when].edges, 
                                self.frames[when].cells,
                                self.frames[when].big_edges_list,
                                timeseries=time_series,
                                at_time=when,
                                # externals_to_use = self.frames[when].border_vertices,
                                externals_to_use = 'none',
                                term=term,
                                metadata=metadata)
        
        self.frames[when].forces = res[0]
        self.frames[when].earr = res[2]
        self.frames[when].position_map = res[3]
        # self.frames[when].versors = res[3]

        self.frames[when].assign_tensions_to_big_edges()
        
    def get_accelerations():
        raise(NotImplementedError)
    
    def get_velocities():
        raise(NotImplementedError)

    def log_force(self, when: int):
        df = pd.DataFrame.from_dict(self.frames[when].forces, orient='index')
        df['is_border'] = [False if isinstance(x, int) else True for x in df.index]
        return df

    def get_edge_force(self, v0, v1, t0=-1, tmax=-1):
        """
        get lambda value for the edge given two vertices of the edge
        """
        edge_force = []
        if tmax == -1 or t0 == -1:
            range_values = self.mesh.mapping.keys()
        else:
            range_values = range(t0, tmax)

        for t in range_values:
            current_v0 = self.mesh.get_point_id_by_map(v0, t0, t)
            current_v1 = self.mesh.get_point_id_by_map(v1, t0, t)
            for e in self.frames[t].earr:
                if current_v0 in e and current_v1 in e:
                    edge_id = self.frames[t].earr.index(e)
            edge_force.append(self.frames[t].forces[edge_id])

        return edge_force          
            
            
    def remove_outermost_edges(self, frame_number: int, layers: int = 1) -> tuple:
        if layers == 0:
            return True
        elif layers > 1:
            raise(NotImplementedError)
        else:
            # vertex_to_delete = []
            border_cells_ids = [cell.id for cell in self.frames[0].cells.values() if cell.is_border]
            for cell_id in border_cells_ids:
                ### hardcoded remove after!
                if (self.frames[frame_number].time == 3 and cell_id == 43) \
                or (self.frames[frame_number].time == 17 and cell_id == 50) \
                or (self.frames[frame_number].time == 24 and cell_id == 9) \
                or (self.frames[frame_number].time == 28 and cell_id == 35) \
                or (self.frames[frame_number].time == 11 and cell_id == 39):
                    pass
                else:
                    self.remove_cell(frame_number, cell_id)            
                ###
                # self.remove_cell(frame_number, cell_id)            
            return self.frames[frame_number].vertices, self.frames[frame_number].edges, self.frames[frame_number].cells


    def remove_cell(self, frame_number: int, cell_id: int):
        vertex_to_delete = []
        for vertex in self.frames[frame_number].cells[cell_id].vertices:
            # if all(cid in border_cells_ids for cid in vertex.ownCells):
            if len(vertex.ownCells) < 2:
                assert vertex.ownCells[0] == self.frames[frame_number].cells[cell_id].id, "Vertex incorrectly placed in cell"
                for ii in vertex.ownEdges.copy():
                    del self.frames[frame_number].edges[ii]

                vertex_to_delete.append(vertex.id)
                # vertex.ownCells.remove(self.frames[frame_number].cells[cell_id].id)
                # vertex.remove_cell(cell_id)
        
        for vertex_id in list(set(vertex_to_delete)):
            del self.frames[frame_number].vertices[vertex_id]

        del self.frames[0].cells[cell_id]

        self.frames[frame_number] = fsframes.Frame(self.frames[frame_number].vertices, 
                                                    self.frames[frame_number].edges, 
                                                    self.frames[frame_number].cells, 
                                                    time=self.frames[frame_number].time,
                                                    gt=self.frames[frame_number].gt,
                                                    surface_evolver=self.frames[frame_number].surface_evolver)
        return self.frames[frame_number]
