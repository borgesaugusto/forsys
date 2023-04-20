from dataclasses import dataclass
import forsys.virtual_edges as ve
import forsys.general_matrix as forsys_general_matrix

from dataclasses import dataclass, field
from typing import Union
import numpy as np
from sympy import Matrix

class PressureMatrix(forsys_general_matrix.GeneralMatrix):
    
    def __init__(self, frame, timeseries):
        super().__init__(frame, timeseries)
        self._build_matrix()

    def _build_matrix(self):
        # go over each big edge
        self.rhs_matrix = np.zeros(len(self.big_edges_to_use))
        for position_id, big_edge in enumerate(self.big_edges_to_use):
            lhs_row, rhs_value =  self.get_row(big_edge)
            self.rhs_matrix[position_id] = rhs_value
            if not self.lhs_matrix:
                self.lhs_matrix = Matrix(( [lhs_row] ))
            else:
                self.lhs_matrix = self.lhs_matrix.row_insert(self.lhs_matrix.shape[0], Matrix(([lhs_row])))

        return self.lhs_matrix, self.rhs_matrix


    def get_row(self, big_edge):
        # create the two rows of the A matrix.
        # arrx = self.get_big_edge_equation(beid)
        if len(big_edge.own_cells) > 2:
            raise(NotImplementedError)
        
        total_number_cells = len(self.frame.cells)
        c1_position = list(self.frame.cells.keys()).index(big_edge.own_cells[0])
        c2_position = list(self.frame.cells.keys()).index(big_edge.own_cells[1])

        curvature = big_edge.calculate_total_curvature(normalized=False)
        rhs_value = big_edge.tension * curvature    
        
        lhs_row = np.zeros(total_number_cells)
        # if curvature > 0:
        lhs_row[c1_position] = 1
        lhs_row[c2_position] = -1

        # print(f"Edge {big_edge.big_edge_id} with curvature {curvature} and tension {big_edge.tension} => DP = {rhs_value}")

        return lhs_row, rhs_value

