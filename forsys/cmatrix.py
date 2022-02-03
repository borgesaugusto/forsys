import numpy as np
from mpmath import mp
from sympy import *


import forsys.virtual_edges as ve
import forsys.plot as plot

def cmatrix(vertices, edges, cells):
    init_printing(use_unicode=True)
    vertices, edges, cells, earr = ve.generate_mesh(vertices, edges, cells, 4)
    # configuration = Lattice(vertices, edges, cells)

    extForces = {}
    matrixPosForce = {}
    solutionName = {}
    m = Matrix(())
    fd = {}

    c = 0
    vseen = []
    versorsUsed = {}
    inBorder = []
    earrN = ve.new_virtual_edges(inBorder, earr)
    virtualDictionary, _, _ = ve.get_virtual_edges(earrN, vertices)
    print("virtualDictionary: ", virtualDictionary)
    te = len(earr)
    # position of external forces:
    val = 0
    for vid in virtualDictionary:
        vuse = []
        arrx = np.zeros(te)
        arry = np.zeros(te)
        print(vid)
        for el in ve.get_versors(vertices, edges, vid):
            pos = ve.eid_from_vertex(earrN,  el[2])
            arrx[pos] = round(el[0], 3)
            arry[pos] = round(el[1], 3)
            vuse.append(el[2])
            # solutionName[pos] = el[3]

        versorsUsed[vid] = vuse
        mergeX = list(arrx)
        mergeY = list(arry)

        if len(m) == 0:
            m = Matrix(( [mergeX] ))
            m = m.row_insert(m.shape[0], Matrix(( [mergeY] )))
        else:
            m = m.row_insert(m.shape[0], Matrix(([mergeX])))
            m = m.row_insert(m.shape[0], Matrix(([mergeY])))

    return m, extForces, earrN, versorsUsed, (vertices, edges, cells)



# def get_versors(vertices, edges, vid):
#     for eb in configuration.getVertices()[vid].getEdgesBelonging():
#         edge = configuration.getEdges()[eb]
#         deltaX, deltaY, modulus = edge.getDistances()
#         if vid == edge.getVerticesBelonging()[0]:
#             vx = - deltaX/modulus
#             vy = - deltaY/modulus
#         else:
#             vx = deltaX/modulus
#             vy = deltaY/modulus
#         # Force versor, to calculate modulus
#         yield (vx, vy, edge.getVerticesBelonging(), edge.edgeID)
