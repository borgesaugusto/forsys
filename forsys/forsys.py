from dataclasses import dataclass, field
import pandas as pd
# from forsys import virtual_edges as ve
import forsys.time_series as ts
# from  forsys.exceptions import DifferentTissueException
import forsys.solve as solver
# import forsys.frames as fframe
# import forsys.plot as fplot
# import forsys.surface_evolver as se 
@dataclass
class ForSys():
    frames: dict

    cm: bool=True

    def __post_init__(self):
        # if there is more than 1 time, generate time series
        if len(self.frames) > 1:
            self.mesh = ts.TimeSeries(self.frames, cm=self.cm)
            
            self.times_to_use = self.mesh.times_to_use()

    def solve(self, when: int, term: str, time_series:dict = None, metadata:dict = {}) -> None:
        res = solver.equations(self.frames[when].vertices, 
                                self.frames[when].edges, 
                                self.frames[when].cells,
                                self.frames[when].big_edges_list,
                                timeseries=time_series,
                                at_time=when,
                                externals_to_use = self.frames[when].border_vertices,
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

