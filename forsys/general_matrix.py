import numpy as np
from sympy import Matrix
from dataclasses import dataclass, field
from mpmath import mp
import scipy.optimize as scop
from typing import Tuple


@dataclass
class GeneralMatrix:
    """Top level class to generalize solving the matrix system. Serves as parent for 
    pressure and stress matrix

    :param frame: Frame object from which matrix system is created
    :type frame: object
    :param timeseries: Time series dictionary of frames
    :type timeseries: dict
    """
    frame: object
    timeseries: dict

    def __post_init__(self):
        """Constructor method
        """
        self.map_vid_to_row = {}
        
        self.externals_to_use = []
        self.big_edges_to_use = self.frame.internal_big_edges
        self.mapping_order = {}

        self.lhs_matrix = None
        self.rhs_matrix = None


    def solve_system(self, **kwargs) -> list:
        """Solve the system of equations. The LHS and RHS have to be set by the child classes beforehand.

        :return: Returns a list with the values for the solution.
        :rtype: list
        """
        np.seterr(all='raise')
        assert self.lhs_matrix is not None, "LHS matrix not set"
        assert self.rhs_matrix is not None, "RHS matrix not set"

        self.lhs_matrix, self.rhs_matrix = self.get_least_squares_matrices(self.lhs_matrix, 
                                                                           self.rhs_matrix)

        removed_index = None
        if kwargs.get("method", None) == "fix_stress":
            self.lhs_matrix, self.rhs_matrix, removed_index = self.fix_one_stress(self.rhs_matrix)
        elif kwargs.get("method", None) == "lagrange_pressure":
            self.lhs_matrix, self.rhs_matrix = self.add_lagrange_multiplier(self.lhs_matrix, self.rhs_matrix, 0)
        
        self.rhs_matrix = Matrix([np.round(float(val), 3) for val in self.rhs_matrix])

        self.lhs_matrix = np.array(self.lhs_matrix).astype(np.float64)
        self.rhs_matrix = np.array(mp.matrix(self.rhs_matrix)).astype(np.float64)

        try:
            xres = Matrix(np.linalg.inv(self.lhs_matrix) * Matrix(self.rhs_matrix))
            
            if np.any([x<0 for x in xres[:-1]]) and not kwargs.get("allow_negatives", True):
                print("Numerically solving due to negative values")
                xres, _ = scop.nnls(self.lhs_matrix, self.rhs_matrix, maxiter=100000)
                xres = Matrix(xres)
        except np.linalg.LinAlgError:
            # then try with nnls
            print("Numerically solving due to singular matrix")
            xres, _ = scop.nnls(self.lhs_matrix, self.rhs_matrix, maxiter=100000)
            xres = Matrix(xres)

        if removed_index is not None:
            xres = xres.row_insert(removed_index, Matrix([1]))
        
        self.solution = xres[:-1]
        for _, val in self.mapping_order.items():
            if val in self.removed_columns:
                self.solution.insert(val, 0)

        return self.solution

    def add_lagrange_multiplier(self, lhs_matrix: Matrix, rhs_matrix: Matrix, constraint: float) -> Tuple:
        """Add a constraint to the matrix system through a Lagrange multiplier. 
        LHS and RHS matrices should be already in least squares form.

        :param lhs_matrix: LHS matrix
        :type lhs_matrix: Matrix
        :param rhs_matrix: RHS matrix
        :type rhs_matrix: Matrix
        :param constraint: Value to give as constraint
        :type constraint: float
        :return: LHS and RHS matrices with the additional Lagrange multiplier.
        :rtype: Tuple
        """
        lhs_matrix = Matrix(lhs_matrix)
        rhs_matrix = Matrix(rhs_matrix)
        ones = np.ones(lhs_matrix.shape[1])
        cMatrix = Matrix(ones)
        additional_zero = np.zeros(1)
        lhs_matrix = lhs_matrix.row_insert(lhs_matrix.shape[0], cMatrix.T)
        ## Now insert the two corresponding cols for the lagrange multpliers
        cMatrix = Matrix(np.concatenate((ones, additional_zero), axis=0))
        lhs_matrix = lhs_matrix.col_insert(lhs_matrix.shape[1], cMatrix)

        zeros = Matrix(np.zeros(rhs_matrix.shape[1]))
        rhs_matrix = rhs_matrix.row_insert(rhs_matrix.shape[0]+1, zeros)
        if type(constraint) is float or int:
            rhs_matrix[rhs_matrix.shape[0]-1, rhs_matrix.shape[1]-1] = constraint

        return lhs_matrix, rhs_matrix

    def fix_one_stress(self, column_to_fix: int = None, value_to_fix_to: float =1) -> Tuple:
        """Fix one of the values in the solution. This could be used to give a scale to the solution.

        :param column_to_fix: ID of unknown to fix, defaults to None
        :type column_to_fix: int, optional
        :param value_to_fix_to: Value to give the fixed unknown, defaults to 1
        :type value_to_fix_to: float, optional
        :return: Returns the values of the LHS and RHS matrix and the new number of unknowns
        :rtype: Tuple
        """
        assert self.lhs_matrix is not None, "LHS matrix not set"
        assert self.rhs_matrix is not None, "RHS matrix not set"
        # Replace column with the most connections
        if not column_to_fix:
            non_zero_count = [np.count_nonzero(self.lhs_matrix.col(col_id)) 
                            for col_id in range(0, self.lhs_matrix.shape[1])]
            max_index = np.argmax(non_zero_count)
            self.rhs_matrix = self.rhs_matrix - value_to_fix_to * self.lhs_matrix.col(max_index)
            self.lhs_matrix.col_del(max_index)
        else:
            raise(NotImplementedError)

        return self.lhs_matrix, self.rhs_matrix, max_index
    
    @staticmethod
    def get_least_squares_matrices(lhs_matrix: Matrix, rhs_matrix: Matrix)-> Tuple:
        """Create the least square system by multiplying the LHS and RHS matrix by 
        the transpose of the LHS.

        :param lhs_matrix: Left hand side matrix
        :type lhs_matrix: Matrix
        :param rhs_matrix: Right hand side matrix
        :type rhs_matrix: Matrix
        :return: Returns the new value for the LHS and RHS matrices.
        :rtype: Tuple
        """
        least_squares_lhs_matrix = Matrix(lhs_matrix).T * Matrix(lhs_matrix)
        least_squares_rhs_matrix = Matrix(lhs_matrix).T * Matrix(rhs_matrix)

        return least_squares_lhs_matrix, least_squares_rhs_matrix

