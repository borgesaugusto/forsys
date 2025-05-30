import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple
import itertools as itert
import scipy.optimize as scop
import copy
import forsys.virtual_edges as ve
import forsys.borders as borders
from forsys.exceptions import BigEdgesBadlyCreated
import warnings
@dataclass
class ForceMatrix:
    """
    Class to build and solve the force matrix

    :param frame: Frame object from which matrix system is created
    :type frame: object
    :param externals_to_use: List of external vertices to use, or method to determine them. 
    :type externals_to_use: str or list
    :param term: If externals are incorporated, a term can be added to account for the third \
        force in the balance. area, perimeter, area-perimeter and ext are possible.
    :type term: str
    :param metadata: Extra parameters required by functions in the class. 
    :type metadata: dict
    :param timeseries: Time series dictionary of frames
    :type timeseries: dict
    """
    # TODO: Incorporate stress solution to the GeneralMatrix paradigm
    frame: object
    externals_to_use: Union[str, list]
    term: str
    metadata: dict
    timeseries: dict
    angle_limit: float = np.inf
    circle_fit_method: str = "dlite"

    def __post_init__(self):
        """Constructor method
        """
        self.map_vid_to_row = {}
        self.map_edge_to_column = {}
        self.deletes = set()

        if type(self.externals_to_use) == str:
            if self.externals_to_use == 'all':
                # self.externals_to_use = list(np.array(ve.get_border_edge(self.frame.big_edges_list, 
                #                                                         self.frame.vertices)).flatten())
                raise(NotImplementedError)
            elif self.externals_to_use == 'ext':
                self.externals_to_use = ve.get_border_from_angles_new(self.frame.big_edges_list, 
                                                                        self.frame.vertices)
                self.big_edges_to_use = self.frame.big_edges_list
            else:
                # Only solve on the triple junctions
                self.externals_to_use = []
                # TODO: make it work with the general matrix solver using self.frame.internal_big_edges
                # self.big_edges_to_use = self.frame.internal_big_edges_vertices
                self.big_edges_to_use, _ = self.get_angle_limited_edges()
        else:
            raise(NotImplementedError)

        tj_vertices = set()
        for big_edge in self.big_edges_to_use:
            tj_vertices.add(big_edge[0])
            tj_vertices.add(big_edge[-1])

        self.tj_vertices = list(tj_vertices)
        self.matrix = self._build_matrix()
        self.rhs = None
        self.velocity_matrix = None


    def _build_matrix(self) -> np.ndarray:
        """Build the stress inference matrix row by row

        :return: Stress inference matrix before least squares is applied
        :rtype: Matrix
        """
        max_rows = len(self.tj_vertices) * 2
        cols = len(self.big_edges_to_use) + len(self.externals_to_use) * 2
        mat = np.empty(shape=(max_rows, cols))
        position_index = 0
        for vid in self.tj_vertices:
            row_x, row_y = self.get_row(vid)
            non_zero_x = np.count_nonzero(row_x)
            non_zero_y = np.count_nonzero(row_y)
            at_least_three = non_zero_x >= 3 or non_zero_y >= 3
            less_than_four = non_zero_x < 4 and non_zero_y < 4
            less_than_four_condition = less_than_four if self.metadata.get("ignore_four", False) else True

            if at_least_three and less_than_four_condition:
                self.map_vid_to_row[vid] = position_index
                mat[position_index] = row_x
                mat[position_index + 1] = row_y
                position_index += 2

        # return matrix with populated rows
        return mat[:position_index]

    def get_row(self, vid: int) -> Tuple:
        """Create the general row by appending inference rows to external terms

        :param vid: ID of the vertex for the required row
        :type vid: int
        :return: The rows corresponding to the x and y components.
        :rtype: _type_
        """
        # create the two rows of the A matrix (row_x, row_y)
        arrx, arry = self.get_vertex_equation(vid)
        arrxF, arryF = self.get_external_term(vid)
        return np.concatenate((arrx, arrxF)), np.concatenate((arry, arryF))
        
    def get_external_term(self, vid: int) -> Tuple:
        """Get the external forces corresponding to the giving vertex in row form to append
        to the stress inference matrix

        :param vid: Current vertex ID
        :type vid: int
        :return: X and Y component of the external forces for the required vertex
        :rtype: Tuple
        """
        arrxF = np.zeros(len(self.externals_to_use)*2)
        arryF = np.zeros(len(self.externals_to_use)*2)
        if vid in self.externals_to_use:
            if self.term == 'area':
                fxVersor, fyVersor = borders.get_versors_area(self.frame.vertices, 
                                                                self.frame.edges, 
                                                                self.frame.cells, 
                                                                vid, 
                                                                self.metadata)
                # print("vID:", vid, " force : ", fxVersor, fyVersor)
            elif self.term == 'perimeter':
                fxVersor, fyVersor = borders.get_versors_perimeter(self.frame.vertices, 
                                                                    self.frame.edges, 
                                                                    self.frame.cells, 
                                                                    vid, 
                                                                    self.metadata)
            elif self.term == 'area-perimeter':
                axVersor, ayVersor = borders.get_versors_area(self.frame.vertices, 
                                                                self.frame.edges, 
                                                                self.frame.cells,
                                                                vid, 
                                                                self.metadata)
                pxVersor, pyVersor = borders.get_versors_perimeter(self.frame.vertices, 
                                                                    self.frame.edges, 
                                                                    self.frame.cells, 
                                                                    vid, 
                                                                    self.metadata)
                fxVersor = axVersor + pxVersor
                fyVersor = ayVersor + pyVersor
            elif self.term == 'none':
                fxVersor = 0
                fyVersor = 0
            else: # term='ext' is the default, also works for term='timeseries-velocity'
                fxVersor = 0
                fyVersor = 0
                for el in ve.get_versors(self.frame.vertices, self.frame.edges, vid):
                    fxVersor -= round(el[0], 3)
                    fyVersor -= round(el[1], 3)
            current_border_id = np.where(np.array(self.externals_to_use).astype(int) == vid)[0]

            arrxF[current_border_id*2] = round(fxVersor, 3)
            arryF[current_border_id * 2 + 1] = round(fyVersor, 3)
        return arrxF, arryF

    def get_vertex_equation(self, vid: int) -> Tuple:
        """Generate the stress inference row corresponding to the given vertex ID 

        :param vid: ID of the required vertex's equation
        :type vid: int
        :return: X and Y componenet of the stress inference matrix for the required vertex
        :rtype: Tuple
        """
        arrx = np.zeros(len(self.big_edges_to_use))
        arry = np.zeros(len(self.big_edges_to_use))
        vertex = self.frame.vertices[vid]
        vertex_big_edges = [self.frame.big_edges[beid] for beid in vertex.own_big_edges]
        vertex_big_edges_versors = [big_edge.get_versor_from_vertex(vid, fit_method=self.circle_fit_method) for big_edge in vertex_big_edges]
        # Find the three angles
        # TODO: Should something be done for 4-fold junctions ?
        # if len(vertex_big_edges) == 3:
        #     combinations = itert.combinations(vertex_big_edges_versors, r=2)
        #     angles = [np.arccos(np.dot(*combination)) for combination in combinations]
        #     if np.max(angles) >= self.angle_limit:
        #         self.deletes.add(vid)
        #         vertex_big_edges_versors = np.zeros((3, 2))

        for index, big_edge in enumerate(vertex_big_edges):
            if not big_edge.external and len(vertex.ownCells) > 2:
                try:
                    pos = ve.eid_from_vertex(self.big_edges_to_use, big_edge.get_vertices_ids())
                    versor = vertex_big_edges_versors[index]
                    arrx[pos] = versor[0]
                    arry[pos] = versor[1]
                except BigEdgesBadlyCreated:
                    continue

        return arrx, arry
    
    def get_angle_limited_edges(self):
        big_edges_vertices = [big_edge.get_vertices_ids() for big_edge in self.frame.internal_big_edges]
        tj_vertices = set([big_edge[0] for big_edge in big_edges_vertices])
        last = set([big_edge[-1] for big_edge in big_edges_vertices])
        tj_vertices.update(last)
        for vid in tj_vertices:
            vertex = self.frame.vertices[vid]
            vertex_big_edges = [self.frame.big_edges[beid] for beid in vertex.own_big_edges]
            vertex_big_edges_versors = [big_edge.get_versor_from_vertex(vid, fit_method=self.circle_fit_method) for big_edge in vertex_big_edges]
            # Find the three angles
            # TODO: Should something be done for 4-fold junctions ?
            # if len(vertex_big_edges) == 3:
            combinations = itert.combinations(vertex_big_edges_versors, r=2)
            angles = [np.arccos(np.dot(*combination)) for combination in combinations]
            if np.max(angles) >= self.angle_limit:
                self.deletes.add(vid)
        
        big_edges_to_use = copy.copy(self.frame.internal_big_edges_vertices)
        
        for _, big_edge in enumerate(big_edges_vertices):
            if (big_edge[0] in self.deletes) and (big_edge[-1] in self.deletes):
                big_edges_to_use.remove(big_edge)

        return big_edges_to_use, self.deletes


    def solve(self, timeseries: Union[str, list] = None, **kwargs) -> dict:
        """Solve the system of equations. The stress inference matrix must be built beforehand.

        :param timeseries: If a dynamical inference is required, the timeseries must be provided, defaults to None
        :type timeseries: Union[str, list], optional
        :return: Dictionary of edges and inferred stress. External forces are included if they were required
        :rtype: dict
        """
        np.seterr(all='raise')
        tote = len(self.big_edges_to_use)

        b, average_velocity = self.set_velocity_matrix(timeseries, **kwargs)

        solver_method = kwargs.get("method", None)
        if solver_method == "fix_stress":
            mprime, b, removed_index = self.fix_one_stress(b)
        elif solver_method == "lsq_linear":
            mprime, b = self.add_mean_one_before(b)
            removed_index = None
        elif solver_method == "lsq":
            mprime, b = self.add_mean_one(b)
            removed_index = None
        else:
            mprime, b = self.add_mean_one(b)
            removed_index = None

        b = b.astype(np.float64)

        self.rhs = b.T.round(4)

        mprime = mprime.astype(np.float64)
        # flatten b to convert it to a vector. Rounding is to keep old behavior (not sure if it's useful)
        b = b.astype(np.float64).flatten().round(3)
        try:
            if solver_method == "lsq_linear":
                solutions = scop.lsq_linear(mprime,
                                            b,
                                            bounds=(0.0, np.inf))
                xres = solutions["x"]
            elif solver_method == "lsq":
                try:
                    import lmfit as lmf
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(f'lmfit is required for {solver_method=}')
                arguments = (mprime, b)
                x0_original = kwargs.get("initial_condition", np.ones(len(self.frame.internal_big_edges)))
                x0, removed_indices = self.get_new_initial_condition(x0_original, what="other")
                x0 = [val for val in x0 if val > 0]
                x0.append(1)

                def lmfit_cost(params, A, b):
                    _x = [params[name].value for name in params]
                    a_x = np.dot(np.array(A), np.array(_x)) 
                    vectorial_differences = a_x - b
                    return vectorial_differences

                def lmfit_cost_std(params, A, b):
                    _x = np.array([params[name].value for name in params])
                    a_x = np.dot(A, _x)
                    vectorial_differences = a_x - b
                    return vectorial_differences * (1 + 0.5 * _x.std())

                parameters = lmf.Parameters()
                for index, val in enumerate(x0):
                    naming = f"_{index}"
                    parameters.add(naming, val)
                    parameters[naming].min = 0

                if kwargs.get("use_std", False):
                    solution = lmf.minimize(lmfit_cost_std,
                                            params=parameters,
                                            args=arguments)
                else:
                    solution = lmf.minimize(lmfit_cost,
                                            params=parameters,
                                            args=arguments)
                # TODO: replace Matrix by ndarray in this code
                xres = [solution.params[name].value for name in solution.params]

                # reinsert all the removed spaces
                for index in removed_indices:
                    xres = xres.insert(index, -1)
            else:
                try:
                    xres = np.linalg.inv(mprime) @ b
                except np.linalg.LinAlgError:
                    raise ValueError("Singular matrix")
                
                if np.any([x < 0 for x in xres[:-1]]) and not kwargs.get("allow_negatives", True):
                    raise ValueError("Negative values detected")
        except (ValueError, np.linalg.LinAlgError, TypeError) as e:
            warnings.warn(f"Numerically solving due to the following error: {e}")
            xres, _ = scop.nnls(mprime, b, maxiter=kwargs.get("nnls_max_iter"))

        if kwargs.get("verbose", False):
            print("Residuals ||AX - B||: ", np.linalg.norm(mprime @ xres - b))

        if removed_index is not None:
            xres = np.insert(xres, removed_index, 1.)

        for index, element in enumerate(self.big_edges_to_use):
            edges_to_use = [list(set(self.frame.vertices[element[vid]].ownEdges) & 
                            set(self.frame.vertices[element[vid+1]].ownEdges))[0]
                            for vid in range(0, len(element)-1)]
            for e in edges_to_use:
                self.frame.edges[e].tension = float(xres[index])

        xres = xres[:-1]
        xres = self.get_solution_no_discarded(xres)
        self.force_dictionary = {}
        # TODO: Reincorporate external forces
        for index, value in enumerate(xres):
        # for i in range(0, tote):
        #     if i < tote:
        #         val = float(xres[i])
            self.force_dictionary[index] = value
        i = 0

        if type(self.externals_to_use) == list or \
            self.externals_to_use == "all" or \
            self.externals_to_use == "ext":
            extForces = {}
            for vid in self.externals_to_use:
                current_border_id = np.where(np.array(self.frame.border_vertices).astype(int) == vid)[0]
                row_idx = self.map_vid_to_row[vid]
                col_idx = int(tote + current_border_id * 2)
                extForces[vid] = self.matrix[row_idx, col_idx: col_idx + 2].tolist()

            for index, _ in extForces.items():
                val1 = round(xres[tote+i], 3) * extForces[index][0]
                val2 = round(xres[tote+i+1], 3) * extForces[index][1]
                self.force_dictionary[f'F{index}x'] = val1
                self.force_dictionary[f'F{index}y'] = val2
                i += 1

        return self.force_dictionary

    def add_mean_one(self, b: np.ndarray) -> Tuple:
        """Add the lagrange multiplier required the average of stresses values equal to one.

        :param b: Right hand side matrix. If in a statical modality it is the null column, \
            in dynamcal modality, has the components of the velocity for each edge.
        :type b: Matrix
        :return: The new LHS and RHS matrices
        :rtype: Tuple
        """
        mprime = self.matrix
        total_edges = mprime.shape[1]
        total_vertices = mprime.shape[0]
        total_borders = len(self.externals_to_use)

        cMatrix = np.array([1.] * total_edges + [0.] * (total_borders * 2))
        mprime = np.vstack((mprime, cMatrix))
        # Now insert the two corresponding cols for the lagrange multipliers
        cMatrix = np.array([1.] * total_vertices + [0.] * (total_borders * 2) + [0.])
        mprime = np.hstack((mprime, cMatrix.reshape(-1, 1)))
        b = np.vstack((b, np.zeros(b.shape[1])))
        b[-1, -1] = total_edges

        if total_borders != 0:
            # cmatrix forces
            cMatrix = np.array([0.] * total_edges + [1.] * (total_borders * 2) + [0.])
            mprime = np.hstack((mprime, cMatrix.reshape(-1, 1)))
            cMatrix = np.concatenate((cMatrix, np.zeros(1)))
            mprime = np.vstack((mprime, cMatrix))

            b = np.vstack((b, np.zeros(b.shape[1])))

        return mprime, b
    
    def add_mean_one_before(self, b: np.ndarray) -> Tuple:
        """Add the lagrange multiplier required the average of stresses values equal to one.

        :param b: Right hand side matrix. If in a statical modality it is the null column, \
            in dynamcal modality, has the components of the velocity for each edge.
        :type b: Matrix
        :return: The new LHS and RHS matrices
        :rtype: Tuple
        """
        mprime = self.matrix.T @ self.matrix
        b = self.matrix.T @ b
        total_edges = mprime.shape[1]
        total_borders = len(self.externals_to_use)

        cMatrix = np.array([1.] * total_edges + [0.] * (total_borders * 2))
        mprime = np.hstack((mprime, cMatrix.reshape(-1, 1)))

        # Now insert the two corresponding cols for the lagrange multpliers
        cMatrix = np.concatenate((cMatrix, np.zeros(1)))
        mprime = np.vstack((mprime, cMatrix))

        b = np.vstack((b, np.zeros(b.shape[1])))
        b[-1, -1] = total_edges

        if total_borders != 0:
            # cmatrix forces
            cMatrix = np.array([0.] * total_edges + [1.] * (total_borders * 2) + [0.])
            mprime = np.hstack((mprime, cMatrix.reshape(-1, 1)))

            cMatrix = np.concatenate((cMatrix, np.zeros(1)))
            mprime = np.vstack((mprime, cMatrix))

            b = np.vstack((b, np.zeros(b.shape[1])))

        return mprime, b

    def fix_one_stress(self, b: np.ndarray, column_to_fix: int = None, value_to_fix_to: float = 1) -> Tuple:
        """Fix one of the stresses values. This is helpful for establishing an scale.

        :param b: Right hand side matrix. Zero in the statical case.
        :type b: Matrix
        :param column_to_fix: Column number to be fixed, i.e edge stress. If None uses \
            the vertex with the most connections, defaults to None
        :type column_to_fix: int, optional
        :param value_to_fix_to: Value to fix the stress to, defaults to 1
        :type value_to_fix_to: float, optional
        :return: New LHS and RHS matrix as well as the index of the fixed stress.
        :rtype: Tuple
        """
        # Replace column with the most connections
        if not column_to_fix:
            non_zero_count = np.count_nonzero(self.matrix, axis=0)
            max_index = np.argmax(non_zero_count)
            b = b - value_to_fix_to * self.matrix[:, max_index]
            self.matrix = np.delete(self.matrix, max_index, 1)
        else:
            raise(NotImplementedError)
        
        mprime = self.matrix.T @ self.matrix
        b = self.matrix.T @ b

        return mprime, b, max_index

    def set_velocity_matrix(self, timeseries: Union[str, list] = None, **kwargs):
        vector_of_vectors = []
        b = np.zeros((self.matrix.shape[0], 1))
        b_matrix = kwargs.get("b_matrix", None)
        if timeseries and (b_matrix == "velocity" or b_matrix == "acceleration"):
            for vid in self.map_vid_to_row.keys():
                if b_matrix == "velocity":
                    value = timeseries.calculate_velocity(vid, self.frame.frame_id)
                elif b_matrix == "acceleration":
                    value = timeseries.calculate_acceleration(vid, self.frame.frame_id)
                    if np.any(np.isnan(value)):
                        value = [0, 0]
                else:
                    raise NotImplementedError(f'{b_matrix=}')
                j = self.map_vid_to_row[vid]
                b[j, 0] = value[0]
                b[j + 1, 0] = value[1]
                vector_of_vectors.append(value)

        self.velocity_normalization = kwargs.get("velocity_normalization", 0.1)
        if len(vector_of_vectors) != 0 and kwargs.get("adimensional_velocity",  False):
            average_velocity = np.mean([np.linalg.norm(vector) for vector in vector_of_vectors])
        else:
            average_velocity = 1

        self.velocity_matrix_dimensional = b.T.astype(np.float64).round(4)
        
        b = (b / average_velocity) * self.velocity_normalization
        
        self.velocity_matrix = b.T.astype(np.float64).round(4)

        return b, average_velocity
    
    def get_solution_no_discarded(self, xres: np.ndarray) -> list:
        """Get new solution adding -1 to the big edges that no longer exist

        :param x0: Original solution
        :type x0: list
        :return: New list with solution for the inexesitant edges equal to -1
        :rtype: list
        """
        big_edges_vertices = [big_edge.get_vertices_ids() for big_edge in self.frame.internal_big_edges]
        if len(big_edges_vertices) == len(xres):
            return xres

        xres_i = 0  # pointer to next value in xres
        xres_new = np.empty(len(big_edges_vertices), dtype=xres.dtype)
        for be_index, big_edge in enumerate(big_edges_vertices):
            if (big_edge[0] in self.deletes) and (big_edge[-1] in self.deletes):
                xres_new[be_index] = -1
            else:
                xres_new[be_index] = xres[xres_i]
                xres_i += 1
        return xres_new
    
    def get_new_initial_condition(self, x0: list, what="zero") -> list:
        """Get new initial condition adding zeros to or removing the big edges that no longer exist

        :param x0: Original list
        :type x0: list
        :return: New list with initial condition for the inexesitant edges equal to zero
        :rtype: list
        """
        list_of_big_edges = [big_edge.get_vertices_ids() for big_edge in self.frame.internal_big_edges]
        both_count = 0
        removed_indices = {}
        for index, big_edge in enumerate(list_of_big_edges):
            if (big_edge[0] in self.deletes) and (big_edge[-1] in self.deletes):
                removed_indices[index] = x0[index]
                if what == "zero":
                    x0[index] = 0
                else:
                    x0[index] = -1
                both_count += 1
        return x0, removed_indices
