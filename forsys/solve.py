from turtle import position
import numpy as np
import scipy.optimize as scop
# import matplotlib.pyplot as plt
import warnings
import pandas as pd
from mpmath import mp
from sympy import *

import forsys.virtual_edges as eforce
import forsys.fmatrix as fmatrix
import forsys.time_series as ts

import time

def equations(vertices, edges, cells, earr, timeseries=None, at_time=None, externals_to_use='', term='ext', metadata={}):
    m, earr, border_vertices, position_map = fmatrix.fmatrix(vertices, edges, cells, earr, externals_to_use, term, metadata)
    # countBorder = len(extForces)
    # totv = m.shape[0]
    # tote = m.shape[1] - countBorder*2
    # tote = m.shape[1] - countBorder
    tote = len(earr)
    shapeM = m.shape
    countBorder = int((shapeM[1] - tote) / 2)

    # m = m.T * m
    print(shapeM, " with ", countBorder, " externals and ", tote, "edges")
    # print("Unkowns: ", tote + countBorder*2)
    # print("Equations: ", totv)

    mprime = m.T * m

    b = Matrix(np.zeros(m.shape[0]))

    print("Accelerations: ")
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
    # if b_norm != 0:
    #     b = b/b_norm
    #     mprime = mprime/b_norm
    
    b = m.T * b
    # if b==0, just do NNLS
    if b_norm == 0:
        only_nnls = True
    else:
        only_nnls = False

    mprime, b = add_mean_one(mprime, b, tote, countBorder)

    b = Matrix([np.round(float(val), 3) for val in b])


    # pprint(b)
    start = time.time()

    print("")
    print("||b||", b_norm)
    print("")
    print("NNLS iterating...")
    # print(b)
    bprime = np.zeros(len(b))
    bprime[-2] = b[-2]
    bprime[-1] = b[-1]
    # print("After: ")
    # print(bprime)


    # xres, rnorm = scop.nnls(mprime, mp.matrix(b), maxiter=200000)
    xres, rnorm = scop.nnls(mprime, mp.matrix(bprime), maxiter=200000)
    # xres_bacc, rnorm_bacc = scop.nnls(mprime, mp.matrix(b), maxiter=200000)
    mprime = np.array(mprime).astype(np.float64)
    b = np.array(mp.matrix(b)).astype(np.float64)

   
    xres, rnorm = scop.nnls(mprime, b, maxiter=200000)
    print("Cost for nnls with b!=0  ->   ", cost_function(xres, mprime, b), rnorm)

    # if not only_nnls:
    #     bounds = [(0, np.inf)]*len(b)

    #     x2 = scop.minimize(cost_function, x0=xres, method='L-BFGS-B', args=(mprime, b), bounds=bounds, tol=1e-25)
    #     # print("Cost for nnls b != 0", cost_function(xres_bacc, mprime, b), rnorm_bacc)
    #     # print("Cost for lsq_linear", cost_function(sol.x, mprime, b))

    #     print("Cost for minimize: ", cost_function(x2.x, mprime, b))

    #     xres=x2.x


    # res_list = list(np.around(xres, 3))
    # print("Result: ", [(ii, res_list[ii]) for ii in range(0, len(res_list))])
    # print("Index wanted:", list(np.around(xres, 3)).index(7.747))
    # print("\n")
    # print("Lambda 97:", x2.x[position_map[97]])
    # pprint(mprime * x2.x - b)

    # exit()

    # xres, rnorm = scop.nnls(mprime, b, maxiter=100000)
    end = time.time()
    # print("Forsys NNLS finished, took", end-start, " residuals: ", rnorm)
    print("Forsys LSQ Linear finished, took", end-start)
    # pprint(xres)
    # assign the corresponding value to each edge
    #the id is the one from bedge
    # for e in earr:
    for i in range(0, len(earr)):
        e = earr[i]
        edges_to_use = [list(set(vertices[e[vid]].ownEdges) & 
                        set(vertices[e[vid+1]].ownEdges))[0]
                        for vid in range(0, len(e)-1)]
        for e in edges_to_use:
            edges[e].tension = xres[i]
    
    if b_norm != 0:
        print("Lagrange multipliers: ", xres[-2], xres[-1])

    # forces dictionary:
    f = True
    forceDict = {}
    for i in range(0, tote):
        if i < tote:
            val = xres[i]
        forceDict[i] = val
        # for i in range(0)
    i = 0

    extForces = {}
    for vid in border_vertices:
        current_border_id = np.where(np.array(border_vertices).astype(int) == vid)[0]
        current_row = m.row(position_map[vid])
        index = int(tote+current_border_id*2)
        extForces[vid] = [current_row[index], current_row[int(index+1)]]

    for index, _ in extForces.items():
        # print("x", xres[tote+i])
        # print("y", xres[tote+i+1])
        name1 = "F"+str(index)+"x"
        name2 = "F"+str(index)+"y"
        val1 = round(xres[tote+i], 3) * extForces[index][0]
        val2 = round(xres[tote+i+1], 3) * extForces[index][1]
        forceDict[name1] = val1
        forceDict[name2] = val2
        i += 1

    # print("\n \n")
    # print("check: ")
    # pprint(mprime * Matrix(xres) - b)
    # print("----------------")
    # print("----------------")
    # condNew = 'NA'
    # lambdaVal = 'NA'
    # aux = pd.DataFrame()
    # aux['condition'] = [condNew]
    # aux['rnorm'] = [rnorm]
    # aux['lambda'] = [lambdaVal]

    # print("Forces dictionary: ")
    # print(forceDict)

    # print("Condition number: ", condNumber, condNew)
    # return forceDict, aux, xres, earr, (vertices, edges, cells), versorsUsed
    return forceDict, xres, earr, position_map
    # return forceDict, versorsUsed

def log_force(fd):
    df = pd.DataFrame.from_dict(fd, orient='index')
    df['is_border'] = [False if isinstance(x, int) else True for x in df.index]
    return df


def cost_function(x, m, b):
    cost = np.sum(np.linalg.norm(np.dot(m, x) - b))
    # print("cost: ", cost)
    return cost

def add_mean_one(mprime, b, total_edges, total_borders):
    ones = np.ones(total_edges)
    zeros = np.zeros(total_borders*2)
    cMatrix = Matrix(np.concatenate((ones, zeros), axis=0))
    mprime = mprime.row_insert(mprime.shape[0], cMatrix.T)
    # cmatrix forces
    ones = np.ones(total_borders*2)
    zeros = np.zeros(total_edges)
    cMatrix = Matrix(np.concatenate((zeros, ones), axis=0))
    mprime = mprime.row_insert(mprime.shape[0]+1, cMatrix.T)
    
    ## Now insert the two corresponding cols for the lagrange multpliers
    ones = np.ones(total_edges)
    zeros = np.zeros(total_borders*2 + 2) # +2 comes from the two inserted rows
    # zeros = np.zeros(countBorder*2 + 1) # +1 comes from the  inserted rows
    cMatrix = Matrix(np.concatenate((ones, zeros), axis=0))
    mprime = mprime.col_insert(mprime.shape[1], cMatrix)

    ones = np.ones(total_borders*2)
    zeros = np.zeros(total_edges)
    additional_zeros = np.zeros(2) # +2 comes from the two inserted rows
    cMatrix = Matrix(np.concatenate((zeros, ones, additional_zeros), axis=0))
    mprime = mprime.col_insert(mprime.shape[1], cMatrix)

    zeros = Matrix(np.zeros(b.shape[1]))

    b = b.row_insert(b.shape[0]+1, zeros)
    b = b.row_insert(b.shape[0]+1, zeros)
    
    b[b.shape[0]-2,b.shape[1]-2] = total_edges
    b[b.shape[0]-1,b.shape[1]-1] = total_borders*2

    return mprime, b