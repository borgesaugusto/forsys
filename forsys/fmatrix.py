import numpy as np
# from mpmath import mp
from sympy import *

import forsys.virtual_edges as ve
import forsys.borders as borders


def fmatrix(vertices, edges, cells, earr, externals_to_use, term, metadata):
    init_printing(use_unicode=True)
    m = Matrix(())
    if externals_to_use == 'all':
        print("Experimental function, may not work properly")
        inBorder = list(np.array(ve.get_border_edge(earr, vertices)).flatten())
        earrN = ve.new_virtual_edges(inBorder, earr)
        virtualDictionary, _, edgesNumber = ve.get_virtual_edges(earrN, vertices)
    elif type(externals_to_use) == list:
        inBorder = externals_to_use
        earrN = ve.new_virtual_edges(inBorder, earr)
        virtualDictionary, _, _ = ve.get_virtual_edges(earrN, vertices)
    elif externals_to_use == "none":
        earrN = earr
        virtualDictionary, _, _ = ve.get_virtual_edges(earr, vertices)
    else:
        inBorder = ve.get_border_from_angles_new(earr, vertices)
        earrN = ve.new_virtual_edges(inBorder, earr)
        virtualDictionary, _, _ = ve.get_virtual_edges(earrN, vertices)

    jj = 0
    # mapping from vid to row number
    position_map = {}
    for vid in virtualDictionary:
        # exit()
        position_map[vid] = jj
        jj += 2
        row_x, row_y = get_row(vid, earrN, inBorder, vertices, edges, cells, term, metadata)
 
        if len(m) == 0:
            m = Matrix(( [row_x] ))
            m = m.row_insert(m.shape[0], Matrix(( [row_y] )))
        else:
            m = m.row_insert(m.shape[0], Matrix(([row_x])))
            m = m.row_insert(m.shape[0], Matrix(([row_y])))

    return m, earrN, inBorder, position_map

def get_row(vid, earr, border_vertices, vertices, edges, cells, term='ext', metadata={}):
    # create the two rows of the A matrix.
    arrx, arry = get_vertex_equation(vertices, edges, vid, earr)
    arrxF, arryF = get_external_term(vertices, edges, cells, vid, border_vertices, term, metadata)

    row_x = list(arrx)+list(arrxF)
    row_y = list(arry)+list(arryF)

    return row_x, row_y
    
def get_external_term(vertices, edges, cells, vid, border_vertices, term='ext', metadata={}):
    arrxF = np.zeros(len(border_vertices)*2)
    arryF = np.zeros(len(border_vertices)*2)
    if vid in border_vertices:
        if term == 'area':
            fxVersor, fyVersor = borders.get_versors_area(vertices, edges, cells, vid, metadata)
            # print("vID:", vid, " force : ", fxVersor, fyVersor)
        elif term == 'perimeter':
            fxVersor, fyVersor = borders.get_versors_perimeter(vertices, edges, cells, vid, metadata)
        elif term == 'area-perimeter':
            axVersor, ayVersor = borders.get_versors_area(vertices, edges, cells, vid, metadata)
            pxVersor, pyVersor = borders.get_versors_perimeter(vertices, edges, cells, vid, metadata)
            fxVersor = axVersor + pxVersor
            fyVersor = ayVersor + pyVersor
        elif term == 'none':
            fxVersor = 0
            fyVersor = 0
        else: # term='ext' is the default, also works for term='timeseries-velocity'
            # bug!! this should sum all the parts... 
            # Solved ??
            fxVersor = 0
            fyVersor = 0
            for el in ve.get_versors(vertices, edges, vid):
                fxVersor -= round(el[0], 3)
                fyVersor -= round(el[1], 3)
        current_border_id = np.where(np.array(border_vertices).astype(int) == vid)[0]

        arrxF[current_border_id*2] = round(fxVersor, 3)
        arryF[current_border_id * 2 + 1] = round(fyVersor, 3)
    return arrxF, arryF

def get_vertex_equation(vertices, edges, vid, earr):
    arrx = np.zeros(len(earr))
    arry = np.zeros(len(earr))
    for el in ve.get_versors(vertices, edges, vid):
        pos = ve.eid_from_vertex(earr,  el[2])
        arrx[pos] = round(el[0], 3)
        arry[pos] = round(el[1], 3)
    return arrx, arry

