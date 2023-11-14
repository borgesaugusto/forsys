from dataclasses import dataclass
import forsys.virtual_edges as ve
import forsys.general_matrix as forsys_general_matrix

from dataclasses import dataclass, field
from typing import Union
import numpy as np
from sympy import Matrix
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
        self.rhs_matrix = np.zeros(len(self.big_edges_to_use))

        self.mapping_order = {value: enumid for enumid, value in enumerate(self.frame.cells.keys())}

        for position_id, big_edge in enumerate(self.big_edges_to_use):
            lhs_row, rhs_value =  self.get_row(big_edge)

            self.rhs_matrix[position_id] = rhs_value
            if not self.lhs_matrix:
                self.lhs_matrix = Matrix(lhs_row).T
            else:
                self.lhs_matrix = self.lhs_matrix.row_insert(self.lhs_matrix.shape[0],
                                                             Matrix(lhs_row).T)
        
        self.removed_columns = []
        for column in range(self.lhs_matrix.shape[1]):
            if np.all([val == 0 for val in self.lhs_matrix.col(column)]):
                self.removed_columns.append(column)

        ii = 0
        for element in self.removed_columns:
            self.lhs_matrix.col_del(element - ii)
            # self.mapping_order["r_"+str(ii)] = element
            ii += 1

        return self.lhs_matrix, self.rhs_matrix


    def get_row(self, big_edge: object) -> Tuple:
        """Creates the row corresponding to a given big_edge for the LHS matrix 
        and the RHS one. The stress needs to be already inferred.

        :param big_edge: Object instance of the big edge that this row would correspond
        :type big_edge: object
        :return: Returns the left hand side row and the right hand side value.
        :rtype: Tuple
        """
        total_number_cells = len(self.frame.cells)
        lhs_row = np.zeros(total_number_cells)
        rhs_value = 0

        if len(big_edge.own_cells) > 2:
            raise(NotImplementedError)
        
        c1_position = list(self.frame.cells.keys()).index(big_edge.own_cells[0])
        c2_position = list(self.frame.cells.keys()).index(big_edge.own_cells[1])

        # self.mapping_order[big_edge.own_cells[0]] = c1_position
        # self.mapping_order[big_edge.own_cells[1]] = c2_position

        curvature = big_edge.calculate_total_curvature(normalized=False)
        rhs_value = big_edge.tension * curvature    

        if self.frame.cells[big_edge.own_cells[0]].get_area_sign() > 0:
            lhs_row[c1_position] = 1
            lhs_row[c2_position] = -1
        else:
            lhs_row[c1_position] = -1
            lhs_row[c2_position] = 1

        return lhs_row, rhs_value

