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
        true_earr = earrN
    elif type(externals_to_use) == list:
        inBorder = externals_to_use
        earrN = ve.new_virtual_edges(inBorder, earr)
        true_earr = earrN
        virtualDictionary, _, _ = ve.get_virtual_edges(earrN, vertices)
    elif externals_to_use == "none":
        earrN = earr
        inBorder = []
        border_edges = ve.get_border_edges(earr, vertices, edges)
        virtualDictionary, _, _ = ve.get_virtual_edges(earr, vertices)
        true_earr = earr.copy()
        for e in border_edges:
            true_earr.remove(e)
    else:
        inBorder = ve.get_border_from_angles_new(earr, vertices)
        earrN = ve.new_virtual_edges(inBorder, earr)
        true_earr = earrN
        virtualDictionary, _, _ = ve.get_virtual_edges(earrN, vertices)

    jj = 0
    # mapping from vid to row number
    position_map = {}

    for vid in virtualDictionary:
        # exit()
        row_x, row_y = get_row(vid, earrN, inBorder, vertices, edges, cells, true_earr, term, metadata)
        if not np.all((np.array(row_x) == 0)) or not np.all((np.array(row_y) == 0)):
            position_map[vid] = jj
            jj += 2
            if len(m) == 0:
                m = Matrix(( [row_x] ))
                m = m.row_insert(m.shape[0], Matrix(( [row_y] )))
            else:
                m = m.row_insert(m.shape[0], Matrix(([row_x])))
                m = m.row_insert(m.shape[0], Matrix(([row_y])))
    # return m, earrN, inBorder, position_map
    return m, true_earr, inBorder, position_map

def get_row(vid, earr, border_vertices, vertices, edges, cells, true_earr, term='ext', metadata={}):
    # create the two rows of the A matrix.
    arrx, arry = get_vertex_equation(vertices, edges, vid, true_earr)
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
            fxVersor = 0
            fyVersor = 0
            for el in ve.get_versors(vertices, edges, vid):
                fxVersor -= round(el[0], 3)
                fyVersor -= round(el[1], 3)
        current_border_id = np.where(np.array(border_vertices).astype(int) == vid)[0]

        arrxF[current_border_id*2] = round(fxVersor, 3)
        arryF[current_border_id * 2 + 1] = round(fyVersor, 3)
    return arrxF, arryF

def get_vertex_equation(vertices, edges, vid, true_earr):
    arrx = np.zeros(len(true_earr))
    arry = np.zeros(len(true_earr))
    for el in ve.get_versors(vertices, edges, vid):
        if not el[3] and len(vertices[vid].ownCells) > 2:
        # if True:
            pos = ve.eid_from_vertex(true_earr,  el[2])
            xc_2, yc_2 = ve.get_circle_params(true_earr[pos], vertices)
            v0 = vertices[vid]
            ####TODO: Correct the right angle !!

            versor = np.array((- (v0.y - yc_2), (v0.x - xc_2)))

            if el[2][0] != vid:
                versor = -1 * versor

            if np.sign(el[0]) != np.sign(versor[0]):
                versor[0] = versor[0] * -1
            if np.sign(el[1]) != np.sign(versor[1]):
                versor[1] = versor[1] * -1


            versor = versor / np.linalg.norm(versor)
            # print(pos)
            # arrx[pos] = round(el[0], 3)
            # arry[pos] = round(el[1], 3)
            arrx[pos] = versor[0]
            arry[pos] = versor[1]

            # print(vid, el)
    return arrx, arry

