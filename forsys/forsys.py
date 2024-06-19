from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import forsys.time_series as ts
import forsys.fmatrix as fmatrix
import forsys.pmatrix as pmatrix
import forsys.frames as fsframes

@dataclass
class ForSys():
    """
    Class wrapper containing the interface with the ForSys solver

    :param frames: Unique identifier of the cell
    :type id: dict
    :param cm: True if the each frame should be moved the the center of mass \
        system, defaults to False
    :type cm: bool
    :param initial_guess: Dictionary pairing vertex IDs in different timepoints \
        to force ForSys to connect them through time
    :type initial_guess: list
    """
    frames: dict

    cm: bool = False
    initial_guess: list = field(default_factory=list)


    def __post_init__(self):
        """Constructor method        
        """
        # if there is more than 1 time, generate time series
        if len(self.frames) > 1:
            self.mesh = ts.TimeSeries(self.frames, cm=self.cm, 
                                        initial_guess=self.initial_guess)
            
            self.times_to_use = self.mesh.times_to_use()
        else:
            self.mesh = {}
        
        self.force_matrices = {}
        self.pressure_matrices = {}
        self.forces = {k: None for k in range(len(self.frames))}
        self.pressures = {k: None for k in range(len(self.frames))}

    
    def build_force_matrix(self, when: int = 0, term: str = "none", metadata:dict = {}, **kwargs) -> None:
        """
        Interface to create the matrix system to solve for the stresses

        :param when: Time at which the matrix should be constructed, defaults to 0
        :type when: int, optional
        :param term: Define behaviour for external vertices. By default they are not \
            considered, defaults to "none"
        :type term: str, optional
        :param metadata: Extra arguments to supply to the matrix builder, defaults to {}
        :type metadata: dict, optional
        """
        # TODO: add matrix caching
        self.force_matrices[when] = fmatrix.ForceMatrix(self.frames[when],
                                # externals_to_use = self.frames[when].border_vertices,
                                externals_to_use = 'none',
                                term=term,
                                metadata=metadata,
                                timeseries=self.mesh,
                                angle_limit=kwargs.get("angle_limit", np.pi),
                                circle_fit_method=kwargs.get("circle_fit_method", "dlite"),)


    def build_pressure_matrix(self, when: int = 0):
        """
        Interface to create the matrix system to solve for the pressures. 
        Must be called after stresses were already found.

        :param when: Time at which the matrix should be constructed, defaults to 0
        :type when: int, optional
        """
        # TODO: add matrix caching
        self.pressure_matrices[when] = pmatrix.PressureMatrix(self.frames[when],
                                                              timeseries=self.mesh)
        

    def solve_stress(self, when: int = 0, **kwargs) -> None:
        """
        Interface to call the solver method for the stresses.
        The matrix must already exist.

        :param when: Time at which to calculate the solution, defaults to 0
        :type when: int, optional
        """
        self.forces[when] = self.force_matrices[when].solve(self.mesh, **kwargs)
        self.frames[when].forces = self.forces[when]
        self.frames[when].assign_tensions_to_big_edges()


    def solve_pressure(self, when: int=0, **kwargs):
        """Interface to call the solver method for the pressures.
        The matrix must already exist.

        :param when: Time at which to calculate the solution, defaults to 0
        :type when: int, optional
        """
        assert type(self.forces[when]) is not None, "Forces must be calculated first"
        self.pressures = self.pressure_matrices[when].solve_system(**kwargs)
        self.frames[when].assign_pressures(self.pressures, self.pressure_matrices[when].mapping_order)


    def log_force(self, when: int) -> pd.DataFrame:
        """Create dataframe with the solution for the stresses at a 
        given time.

        :param when: Time at which to create the dataframe
        :type when: int
        :return: Dataframe with the stresses per edge
        :rtype: pd.DataFrame
        """
        df = pd.DataFrame.from_dict(self.frames[when].forces, orient='index')
        df['is_border'] = [False if isinstance(x, int) else True for x in df.index]
        return df


    def get_edge_force(self, v0: int, v1: int, t0: int = -1, tmax: int = -1) -> list:
        """
        Return the edge tension between two timepoints for an edge, given two vertices.
        If no time is provided, the whole evolution is assumed.

        :param v0: First vertex of the edge
        :type v0: int
        :param v1: Second vertex of the edge
        :type v1: int
        :param t0: Initial time at which the edge's stress is required, defaults to -1. \
            Default returns the value for each frame.
        :type t0: int, optional
        :param tmax: Final time for the edge's stress, defaults to -1.
        :type tmax: int, optional
        :return: Array with the values of the edge's stress in time
        :rtype: list
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
        """Remove outermost layer of cells from a tissue. 
        WARNING: This function is experimental

        :param frame_number: Frame to perform the removal at
        :type frame_number: int
        :param layers: Number of layers of cells to remove, defaults to 1
        :type layers: int, optional
        :return: New vertices, edges and cells dictionaries after removal
        :rtype: tuple
        """
        if layers == 0:
            return True
        elif layers > 1:
            raise(NotImplementedError)
        else:
            # vertex_to_delete = []
            border_cells_ids = [cell.id for cell in self.frames[0].cells.values() if cell.is_border]
            for cell_id in border_cells_ids:
                self.remove_cell(frame_number, cell_id)            
            
            return self.frames[frame_number].vertices, self.frames[frame_number].edges, self.frames[frame_number].cells


    def remove_cell(self, frame_number: int, cell_id: int) -> fsframes.Frame:
        """Remove a cell from the tissue

        :param frame_number: Timepoint to perform removal
        :type frame_number: int
        :param cell_id: ID of the cell to be removed
        :type cell_id: int
        :return: New state of the frame
        :rtype: fsframes.Frame
        """
        vertex_to_delete = []
        for vertex in self.frames[frame_number].cells[cell_id].vertices:
            if len(vertex.ownCells) < 2:
                assert vertex.ownCells[0] == self.frames[frame_number].cells[cell_id].id, "Vertex incorrectly placed in cell"
                for ii in vertex.ownEdges.copy():
                    del self.frames[frame_number].edges[ii]

                vertex_to_delete.append(vertex.id)
        
        for vertex_id in list(set(vertex_to_delete)):
            del self.frames[frame_number].vertices[vertex_id]

        del self.frames[0].cells[cell_id]

        self.frames[frame_number] = fsframes.Frame(frame_number, 
                                                   self.frames[frame_number].vertices, 
                                                   self.frames[frame_number].edges, 
                                                   self.frames[frame_number].cells, 
                                                   time=self.frames[frame_number].time,
                                                   gt=self.frames[frame_number].gt)
        return self.frames[frame_number]


    def get_system_velocity_per_frame(self, time_interval: list=None, angle_limit=np.inf) -> list:
        """Calculate the average velocity of the frame. Useful for 
        normalization purposes. When this functions is called, the system's
        matrix is automatically generated.

        :param time_interval: Times to do the calculation. Default is every frame, defaults to None
        :type time_interval: list, optional
        :param angle_limit: Angle limit to ignore triple junctions in the inference, defaults to np.inf
        :type angle_limit: _type_, optional
        :return: List of velocity averages per frame
        :rtype: list
        """
        velocities_per_frame = []
        if type(time_interval) is not list:
            time_interval = range(len(self.frames))
        for time in time_interval:
            self.build_force_matrix(when=time,
                                    angle_limit=angle_limit)
            _, ave_velocity = self.force_matrices[time].set_velocity_matrix(self.mesh,
                                                                            b_matrix="velocity",
                                                                            adimensional_velocity=True)
            velocities_per_frame.append(ave_velocity)

        return velocities_per_frame