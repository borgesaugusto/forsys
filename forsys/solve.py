import numpy as np
import scipy.optimize as scop
import pandas as pd
from mpmath import mp
from sympy import *

import forsys.virtual_edges as eforce
import forsys.fmatrix as fmatrix

def equations(vertices, edges, cells, earr, timeseries=None, at_time=None, externals_to_use='', term='ext', metadata={}):
    np.seterr(all='raise')
    m, earr, border_vertices, position_map = fmatrix.fmatrix(vertices, edges, cells, earr, externals_to_use, term, metadata)
    tote = len(earr)
    shapeM = m.shape
    countBorder = int((shapeM[1] - tote) / 2)


    mprime = m.T * m

    b = Matrix(np.zeros(m.shape[0]))
    if timeseries != None:
        # for vid in versorsUsed.keys():
        for vid in position_map.keys():
            if term == "timeseries-velocity":
                value = timeseries.calculate_velocity(vid, at_time)
            else: # term == "timeseries-acceleration" is the default
                value = timeseries.calculate_acceleration(vid, at_time)
                if np.any(np.isnan(value)):
                    value = [0, 0]
            # print(vid, value)
            j = position_map[vid]
            b[j] = value[0]
            b[j+1] = value[1]


    # normalize b vector
    b_norm = np.linalg.norm(np.array(b).astype(float))
    b = m.T * b
    # if b==0, just do NNLS
    # if b_norm == 0:
    #     only_nnls = True
    # else:
    #     only_nnls = False

    mprime, b = add_mean_one(mprime, b, tote, countBorder)
    b = Matrix([np.round(float(val), 3) for val in b])

    bprime = np.zeros(len(b))
    bprime[-2] = b[-2]
    bprime[-1] = b[-1]
   
    # xres, rnorm = scop.nnls(mprime, mp.matrix(b), maxiter=200000)
    # xres_bacc, rnorm_bacc = scop.nnls(mprime, mp.matrix(b), maxiter=200000)
    mprime = np.array(mprime).astype(np.float64)
    b = np.array(mp.matrix(b)).astype(np.float64)

   
    try:
        xres = Matrix(np.linalg.inv(mprime) * Matrix(b))
        
        if np.any([x<0 for x in xres]):
            print("Numerically solving due to negative values")
            xres, _ = scop.nnls(mprime, b, maxiter=100000)
    except np.linalg.LinAlgError:
        # then try with nnls
        print("Numerically solving due to singular matrix")
        xres, _ = scop.nnls(mprime, b, maxiter=100000)


    for i in range(0, len(earr)):
        e = earr[i]
        edges_to_use = [list(set(vertices[e[vid]].ownEdges) & 
                        set(vertices[e[vid+1]].ownEdges))[0]
                        for vid in range(0, len(e)-1)]
        for e in edges_to_use:
            edges[e].tension = float(xres[i])
    
    # forces dictionary:
    f = True
    forceDict = {}
    for i in range(0, tote):
        if i < tote:
            val = float(xres[i])
        forceDict[i] = val
    i = 0

    extForces = {}
    for vid in border_vertices:
        current_border_id = np.where(np.array(border_vertices).astype(int) == vid)[0]
        current_row = m.row(position_map[vid])
        index = int(tote+current_border_id*2)
        extForces[vid] = [current_row[index], current_row[int(index+1)]]

    for index, _ in extForces.items():
        name1 = "F"+str(index)+"x"
        name2 = "F"+str(index)+"y"
        val1 = round(xres[tote+i], 3) * extForces[index][0]
        val2 = round(xres[tote+i+1], 3) * extForces[index][1]
        forceDict[name1] = val1
        forceDict[name2] = val2
        i += 1

    return forceDict, xres, earr, position_map

def log_force(fd):
    df = pd.DataFrame.from_dict(fd, orient='index')
    df['is_border'] = [False if isinstance(x, int) else True for x in df.index]
    return df


def cost_function(x, m, b):
    cost = np.sum(np.linalg.norm(np.dot(m, x) - b))
    return cost

def add_mean_one(mprime, b, total_edges, total_borders):
    
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