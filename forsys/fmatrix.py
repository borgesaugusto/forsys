import numpy as np
# from mpmath import mp
from sympy import *

import forsys.virtual_edges as ve
import forsys.borders as borders

def fmatrix(vertices, edges, cells, external, term, metadata):
    init_printing(use_unicode=True)
    vertices, edges, cells, earr = ve.generate_mesh(vertices, edges, cells, 4)
    extForces = {}
    matrixPosForce = {}
    m = Matrix(())

    versorsUsed = {}

    if external == 'all':
        print("Experimental function, may not work properly")
        inBorder = list(np.array(ve.get_border_edge(earr, vertices)).flatten())
        earrN = ve.new_virtual_edges(inBorder, earr)
        virtualDictionary, _, edgesNumber = ve.get_virtual_edges(earrN, vertices)
    else:
        inBorder = ve.get_border_from_angles(earr, vertices)
        earrN = ve.new_virtual_edges(inBorder, earr)
        virtualDictionary, _, edgesNumber = ve.get_virtual_edges(earrN, vertices)
    print("There are: ", edgesNumber, " edges")
    te = len(earrN)
    # position of external forces:
    val = 0
    for e in inBorder:
        matrixPosForce[e] = val
        val += 2

    for vid in virtualDictionary:
        vuse = []
        arrx = np.zeros(te)
        arry = np.zeros(te)
        arrxF = np.zeros(len(inBorder)*2)
        arryF = np.zeros(len(inBorder)*2)
        # Check if vertex is in the border:
        fxVersor = 0
        fyVersor = 0
        for el in ve.get_versors(vertices, edges, vid):
            pos = ve.eid_from_vertex(earrN,  el[2])
            arrx[pos] = round(el[0], 3)
            arry[pos] = round(el[1], 3)
            vuse.append(el[2])

        if vid in inBorder:
            if term == 'area':
                fxVersor, fyVersor = borders.get_versors_area(vertices, edges, cells, vid, metadata)
            elif term == 'perimeter':
                fxVersor, fyVersor = borders.get_versors_perimeter(vertices, edges, cells, vid, metadata)
            elif term == 'area-perimeter':
                axVersor, ayVersor = borders.get_versors_area(vertices, edges, cells, vid, metadata)
                pxVersor, pyVersor = borders.get_versors_perimeter(vertices, edges, cells, vid, metadata)
                fxVersor = axVersor + pxVersor
                fyVersor = ayVersor + pyVersor
            elif term == 'timeseries-acceleration' or term=='timeseries-velocity':
                fxVersor = 0
                fyVersor = 0
            else: # term='ext' is the default
                fxVersor -= round(el[0], 3)
                fyVersor -= round(el[1], 3)

            print("vertex: ", vid, "forces: ", fxVersor, fyVersor)
            arrxF[matrixPosForce[vid]] = fxVersor
            arryF[matrixPosForce[vid] + 1] = fyVersor
            extForces[vid] = (round(fxVersor, 3), round(fyVersor, 3))

        versorsUsed[vid] = vuse
        mergeX = list(arrx)+list(arrxF)
        mergeY = list(arry)+list(arryF)

        if len(m) == 0:
            m = Matrix(( [mergeX] ))
            m = m.row_insert(m.shape[0], Matrix(( [mergeY] )))
        else:
            m = m.row_insert(m.shape[0], Matrix(([mergeX])))
            m = m.row_insert(m.shape[0], Matrix(([mergeY])))
    return m, (vertices, edges, cells), extForces, earrN, versorsUsed