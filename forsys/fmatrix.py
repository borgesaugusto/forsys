import numpy as np
from sympy import Matrix
from dataclasses import dataclass, field
from typing import Union, Tuple
from mpmath import mp
import scipy.optimize as scop

import forsys.virtual_edges as ve
import forsys.borders as borders

@dataclass
class ForceMatrix:
    """
    Class to build and solve the force matrix

    :param frame: Frame object from which matrix system is created
    :type frame: object
    :param externals_to_use: List of external vertices to use, or method to determine them. 
    :type externals_to_use: str or list
    :param term: If externals are incorporated, a term can be added to account for the third
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

    def __post_init__(self):
        """Constructor method
        """
        self.map_vid_to_row = {}
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
                self.big_edges_to_use = self.frame.internal_big_edges_vertices
        else:
            raise(NotImplementedError)


        tj_vertices = set([big_edge[0] for big_edge in self.big_edges_to_use])
        last = set([big_edge[-1] for big_edge in self.big_edges_to_use])
        tj_vertices.update(last)

        self.tj_vertices = list(tj_vertices)
        self.matrix = self._build_matrix()
        self.rhs = None
        self.velocity_matrix = None


    def _build_matrix(self) -> Matrix:
        """Build the stress inference matrix row by row

        :return: Stress inference matrix before least squares is applied
        :rtype: Matrix
        """
        position_index = 0
        self.matrix = Matrix(())
        for vid in self.tj_vertices:
            row_x, row_y = self.get_row(vid)
            non_zero_x = np.count_nonzero(row_x)
            non_zero_y = np.count_nonzero(row_y)
            at_least_three = non_zero_x >= 3 or non_zero_y >= 3
            less_than_four = non_zero_x < 4 and non_zero_y < 4
            less_than_four_condition = less_than_four if self.metadata.get("ignore_four", False) else True

            if at_least_three and less_than_four_condition:
                self.map_vid_to_row[vid] = position_index
                position_index += 2
                if len(self.matrix) == 0:
                    self.matrix = Matrix(( [row_x] ))
                    self.matrix = self.matrix.row_insert(self.matrix.shape[0], Matrix(( [row_y] )))
                else:
                    self.matrix = self.matrix.row_insert(self.matrix.shape[0], Matrix(([row_x])))
                    self.matrix = self.matrix.row_insert(self.matrix.shape[0], Matrix(([row_y])))

        return self.matrix

    def get_row(self, vid: int) -> Tuple:
        """Create the general row by appending inference rows to external terms

        :param vid: ID of the vertex for the required row
        :type vid: int
        :return: The rows corresponding to the x and y components.
        :rtype: _type_
        """
        # create the two rows of the A matrix.
        arrx, arry = self.get_vertex_equation(vid)
        arrxF, arryF = self.get_external_term(vid)

        row_x = list(arrx)+list(arrxF)
        row_y = list(arry)+list(arryF)

        return row_x, row_y
        
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
        for big_edge in vertex_big_edges:
            if not big_edge.external and len(vertex.ownCells) > 2:
                pos = ve.eid_from_vertex(self.big_edges_to_use, big_edge.get_vertices_ids())
                versor = big_edge.get_versor_from_vertex(vid)
    
                arrx[pos] = versor[0]
                arry[pos] = versor[1]

        return arrx, arry

    def solve(self, timeseries: Union[str, list] = None, **kwargs) -> dict:
        """Solve the system of equations. The stress inference matrix must be built beforehand.

        :param timeseries: If a dynamical inference is required, the timeseries must be provided, defaults to None
        :type timeseries: Union[str, list], optional
        :return: Dictionary of edges and inferred stress. External forces are included if they were required
        :rtype: dict
        """
        np.seterr(all='raise')
        tote = len(self.big_edges_to_use)

        shapeM = self.matrix.shape
        vector_of_vectors = []
        b = Matrix(np.zeros(shapeM[0]))
        b_matrix = kwargs.get("b_matrix", None)
        if timeseries and  (b_matrix == "velocity" or b_matrix == "acceleration"):
            for vid in self.map_vid_to_row.keys():
                if b_matrix == "velocity":
                    value = timeseries.calculate_velocity(vid, self.frame.frame_id)
                elif b_matrix == "acceleration":
                    value = timeseries.calculate_acceleration(vid, self.frame.frame_id)
                    if np.any(np.isnan(value)):
                        value = [0, 0]
                j = self.map_vid_to_row[vid]
                b[j] = value[0]
                b[j+1] = value[1]
                vector_of_vectors.append(value)

        self.velocity_normalization = kwargs.get("velocity_normalization", 1)
        if len(vector_of_vectors) != 0 and kwargs.get("adimensional_velocity",  False):
            ave_velocities = np.mean([np.linalg.norm(vector) for vector in vector_of_vectors])
        else:
            ave_velocities = 1
        b = (b / ave_velocities) * self.velocity_normalization
        self.velocity_matrix = np.array(list(b.T), dtype=np.float64).round(4)

        if kwargs.get("method", None) == "fix_stress":
            mprime, b, removed_index = self.fix_one_stress(b)
        else:
            mprime, b = self.add_mean_one(b)
            removed_index = None
        b = Matrix([np.round(float(val), 3) for val in b])
        rounded_b = np.array(list(b.T), dtype=np.float64).round(4)
        self.rhs = rounded_b

        mprime = np.array(mprime).astype(np.float64)
        b = np.array(mp.matrix(b)).astype(np.float64)

        try:
            xres = Matrix(np.linalg.inv(mprime) * Matrix(b))
            
            if np.any([x<0 for x in xres[:-1]]) and not kwargs.get("allow_negatives", True):
                print("Numerically solving due to negative values")
                # negative_edges = [self.big_edges_to_use[x_id] for x_id, x_val in enumerate(xres[:-1]) if x_val < 0]
                # print(f"Negatives edges: ", negative_edges)
                xres, _ = scop.nnls(mprime, b, maxiter=100000)
                xres = Matrix(xres)
        except np.linalg.LinAlgError:
            # then try with nnls
            print("Numerically solving due to singular matrix")
            xres, _ = scop.nnls(mprime, b, maxiter=100000)
            xres = Matrix(xres)
        if removed_index is not None:
            xres = xres.row_insert(removed_index, Matrix([1]))
        for index, element in enumerate(self.big_edges_to_use):
            edges_to_use = [list(set(self.frame.vertices[element[vid]].ownEdges) & 
                            set(self.frame.vertices[element[vid+1]].ownEdges))[0]
                            for vid in range(0, len(element)-1)]
            for e in edges_to_use:
                self.frame.edges[e].tension = float(xres[index])
        
        self.force_dictionary = {}
        for i in range(0, tote):
            if i < tote:
                val = float(xres[i])
            self.force_dictionary[i] = val
        i = 0

        if type(self.externals_to_use) == list or \
            self.externals_to_use == "all" or \
            self.externals_to_use == "ext":
            extForces = {}
            for vid in self.externals_to_use:
                current_border_id = np.where(np.array(self.frame.border_vertices).astype(int) == vid)[0]
                current_row = self.matrix.row(self.map_vid_to_row[vid])
                index = int(tote+current_border_id*2)
                extForces[vid] = [current_row[index], current_row[int(index+1)]]

            for index, _ in extForces.items():
                name1 = "F"+str(index)+"x"
                name2 = "F"+str(index)+"y"
                val1 = round(xres[tote+i], 3) * extForces[index][0]
                val2 = round(xres[tote+i+1], 3) * extForces[index][1]
                self.force_dictionary[name1] = val1
                self.force_dictionary[name2] = val2
                i += 1

        return self.force_dictionary

    def add_mean_one(self, b: Matrix) -> Tuple:
        """Add the lagrange multiplier required the average of stresses values equal to one.

        :param b: Right hand side matrix. If in a statical modality it is the null column,
        in dynamcal modality, has the components of the velocity for each edge.
        :type b: Matrix
        :return: The new LHS and RHS matrices
        :rtype: Tuple
        """
        mprime = self.matrix.T * self.matrix
        b = self.matrix.T * b
        # total_edges = len(self.big_edges_to_use)
        total_edges = mprime.shape[1]
        total_borders = len(self.externals_to_use)
        ones = np.ones(total_edges)
        zeros = np.zeros(total_borders*2)
        additional_zero = np.zeros(1)
        cMatrix = Matrix(np.concatenate((ones, zeros), axis=0))
        mprime = mprime.row_insert(mprime.shape[0], cMatrix.T)
        ## Now insert the two corresponding cols for the lagrange multpliers
        cMatrix = Matrix(np.concatenate((ones, zeros, additional_zero), axis=0))
        mprime = mprime.col_insert(mprime.shape[1], cMatrix)

        zeros = Matrix(np.zeros(b.shape[1]))
        b = b.row_insert(b.shape[0]+1, zeros)
        b[b.shape[0]-1,b.shape[1]-1] = total_edges

        if total_borders != 0:
            # cmatrix forces
            ones = np.ones(total_borders*2)
            zeros = np.zeros(total_edges)
            cMatrix = Matrix(np.concatenate((zeros, ones, additional_zero), axis=0))
            mprime = mprime.row_insert(mprime.shape[0]+1, cMatrix.T)
            
            ones = np.ones(total_borders*2)
            zeros = np.zeros(total_edges)
            additional_zeros = np.zeros(2) # +2 comes from the two inserted rows
            cMatrix = Matrix(np.concatenate((zeros, ones, additional_zeros), axis=0))
            mprime = mprime.col_insert(mprime.shape[1], cMatrix)

            zeros = Matrix(np.zeros(b.shape[1]))
            b = b.row_insert(b.shape[0]+1, zeros)
            b[b.shape[0]-1,b.shape[1]-1] = 0

        return mprime, b

    def fix_one_stress(self, b: Matrix, column_to_fix: int = None, value_to_fix_to: float = 1) -> Tuple:
        """Fix one of the stresses values. This is helpful for establishing an scale.

        :param b: Right hand side matrix. Zero in the statical case.
        :type b: Matrix
        :param column_to_fix: Column number to be fixed, i.e edge stress. If None uses
        the vertex with the most connections, defaults to None
        :type column_to_fix: int, optional
        :param value_to_fix_to: Value to fix the stress to, defaults to 1
        :type value_to_fix_to: float, optional
        :return: New LHS and RHS matrix as well as the index of the fixed stress.
        :rtype: Tuple
        """
        # Replace column with the most connections
        if not column_to_fix:
            non_zero_count = [np.count_nonzero(self.matrix.col(col_id)) 
                            for col_id in range(0, self.matrix.shape[1])]
            max_index = np.argmax(non_zero_count)
            b = b - value_to_fix_to * self.matrix.col(max_index)
            self.matrix.col_del(max_index)
        else:
            raise(NotImplementedError)
        
        mprime = self.matrix.T * self.matrix
        b = self.matrix.T * b

        return mprime, b, max_index

