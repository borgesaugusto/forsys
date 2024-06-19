import numpy as np
from dataclasses import dataclass
import scipy.optimize as scop
from typing import Tuple, Optional


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

        ls_lhs_matrix = self.lhs_matrix.T @ self.lhs_matrix
        ls_rhs_matrix = self.lhs_matrix.T @ self.rhs_matrix

        removed_index = None
        if kwargs.get("method") == "fix_stress":
            ls_lhs_matrix, ls_rhs_matrix, removed_index = self.fix_one_stress(ls_lhs_matrix, ls_rhs_matrix)
        elif kwargs.get("method") == "lagrange_pressure":
            ls_lhs_matrix, ls_rhs_matrix = self.add_lagrange_multiplier(ls_lhs_matrix, ls_rhs_matrix, 0)
        
        ls_lhs_matrix = ls_lhs_matrix.astype(np.float64)
        ls_rhs_matrix = ls_rhs_matrix.astype(np.float64)

        try:
            xres = np.linalg.inv(ls_lhs_matrix) @ ls_rhs_matrix

            if np.any(xres[:-1] < 0) and not kwargs.get("allow_negatives", True):
                print("Numerically solving due to negative values")
                xres, _ = scop.nnls(ls_lhs_matrix, ls_rhs_matrix, maxiter=kwargs.get("nnls_max_iter"))

        except np.linalg.LinAlgError:
            # then try with nnls
            print("Numerically solving due to singular matrix")
            xres, _ = scop.nnls(self.lhs_matrix, self.rhs_matrix, maxiter=kwargs.get("nnls_max_iter"))

        if removed_index is not None:
            xres = np.insert(xres, removed_index, 1.)
        
        self.solution = xres[:-1].tolist()
        for val in self.mapping_order.values():
            if val in self.removed_columns:
                self.solution.insert(val, 0.)

        return self.solution

    def add_lagrange_multiplier(self,
                                ls_lhs_matrix: np.ndarray,
                                ls_rhs_matrix: np.ndarray,
                                constraint: Optional[float] = None
                                ) -> Tuple:
        """Add a constraint to the matrix system through a Lagrange multiplier. 
        LHS and RHS matrices should be already in least-squares form.

        :param ls_lhs_matrix: LHS matrix in least-squares form
        :type ls_lhs_matrix: np.ndarray
        :param ls_rhs_matrix: RHS matrix in least-squares form
        :type ls_lhs_matrix: np.ndarray
        :param constraint: Value to give as constraint
        :type constraint: float
        :return: LHS and RHS matrices with the additional Lagrange multiplier.
        :rtype: Tuple
        """
        lhs_rows, lhs_cols = ls_lhs_matrix.shape
        ls_lhs_matrix = np.vstack((ls_lhs_matrix, np.ones(lhs_cols)))

        # Now insert the two corresponding cols for the lagrange multipliers
        cMatrix = np.array([1.] * lhs_rows + [0.])
        ls_lhs_matrix = np.hstack((ls_lhs_matrix, cMatrix.reshape(-1, 1)))

        ls_rhs_matrix = np.vstack((ls_rhs_matrix, np.zeros(ls_rhs_matrix.shape[1])))
        if constraint is not None:
            ls_rhs_matrix[-1, -1] = constraint

        return ls_lhs_matrix, ls_rhs_matrix

    @staticmethod
    def fix_one_stress(ls_lhs_matrix: np.ndarray,
                       ls_rhs_matrix: np.ndarray,
                       column_to_fix: Optional[int] = None,
                       value_to_fix_to: float = 1.
                       ) -> Tuple:
        """Fix one of the values in the solution. This could be used to give a scale to the solution.

        :param ls_lhs_matrix: LHS matrix in least-squares form
        :type ls_lhs_matrix: np.ndarray
        :param ls_rhs_matrix: RHS matrix in least-squares form
        :type ls_rhs_matrix: np.ndarray
        :param column_to_fix: ID of unknown to fix, defaults to None
        :type column_to_fix: int, optional
        :param value_to_fix_to: Value to give the fixed unknown, defaults to 1
        :type value_to_fix_to: float, optional
        :return: Returns the values of the LHS and RHS matrix and the new number of unknowns
        :rtype: Tuple
        """
        # Replace column with the most connections
        if column_to_fix is None:
            non_zero_count = np.count_nonzero(ls_lhs_matrix, axis=0)
            max_index = np.argmax(non_zero_count)
            ls_rhs_matrix = ls_rhs_matrix - value_to_fix_to * ls_lhs_matrix[:, max_index]
            ls_lhs_matrix = np.delete(ls_lhs_matrix, max_index, 1)
        else:
            raise NotImplementedError(f'{column_to_fix=}')

        return ls_lhs_matrix, ls_rhs_matrix, max_index
