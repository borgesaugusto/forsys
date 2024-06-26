import forsys.general_matrix as forsys_general_matrix
import numpy as np
from typing import Tuple


class PressureMatrix(forsys_general_matrix.GeneralMatrix):
    """Child class to manage creating of the matrix system of equations
    to calculate pressure. The corresponding stresses need to be solved already.
    """
    
    def __init__(self, frame: object, timeseries: dict):
        """Constructor method

        :param frame: Frame object from which matrix system is created
        :type frame: object
        :param timeseries: Time series dictionary of frames.
        :type timeseries: dict
        """
        super().__init__(frame, timeseries)
        self._build_matrix()

    def _build_matrix(self) -> Tuple:
        """Generate the matrix system for pressure inference. 

        :return: Returns LHS matrix and RHS matrix
        :rtype: Tuple
        """
        n_edges, n_cells = len(self.big_edges_to_use), len(self.frame.cells)
        self.mapping_order = {key: enumid for enumid, key in enumerate(self.frame.cells)}

        self.rhs_matrix = np.empty(n_edges)
        self.lhs_matrix = np.empty((n_edges, n_cells))
        for position_id, big_edge in enumerate(self.big_edges_to_use):
            lhs_row, rhs_value = self.get_row(big_edge)
            self.rhs_matrix[position_id] = rhs_value
            self.lhs_matrix[position_id] = lhs_row

        # remove columns with all zeros
        self.removed_columns = np.nonzero(np.all(self.lhs_matrix == 0, axis=0))[0].tolist()  # list of column indices
        if self.removed_columns:
            self.lhs_matrix = np.delete(self.lhs_matrix, self.removed_columns, axis=1)

        return self.lhs_matrix, self.rhs_matrix


    def get_row(self, big_edge: object) -> Tuple:
        """Creates the row corresponding to a given big_edge for the LHS matrix 
        and the RHS one. The stress needs to be already inferred.

        :param big_edge: Object instance of the big edge that this row would correspond
        :type big_edge: object
        :return: Returns the left hand side row and the right hand side value.
        :rtype: Tuple
        """
        lhs_row = np.zeros(len(self.mapping_order))  # n_cells
        big_edge_cells = big_edge.own_cells
        if len(big_edge_cells) != 2:
            raise ValueError(f'big edge has own_cells={len(big_edge_cells)}, expecting 2')

        curvature = big_edge.calculate_total_curvature(normalized=False)
        rhs_value = big_edge.tension * curvature

        c1_position = self.mapping_order[big_edge_cells[0]]
        c2_position = self.mapping_order[big_edge_cells[1]]

        if self.frame.cells[big_edge_cells[0]].get_area_sign() > 0:
            lhs_row[c1_position] = 1
            lhs_row[c2_position] = -1
        else:
            lhs_row[c1_position] = -1
            lhs_row[c2_position] = 1

        return lhs_row, rhs_value
