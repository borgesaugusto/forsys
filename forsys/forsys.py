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

def equations(vertices, edges, cells, timeseries=None, at_time=None, external='', term='ext', metadata={}):
    m, (vertices, edges, cells), extForces, earr, versorsUsed = fmatrix.fmatrix(vertices, edges, cells, external, term, metadata)
    countBorder = len(extForces)
    totv = m.shape[0]
    tote = m.shape[1] - countBorder*2
    shapeM = m.shape

    # m = m.T * m
    print(shapeM)
    print("Unkowns: ", tote + countBorder*2)
    print("Equations: ", totv)

    mprime = m.T * m
    ones = np.ones(tote)
    zeros = np.zeros(countBorder*2)
    cMatrix = Matrix(np.concatenate((ones, zeros), axis=0))
    mprime = mprime.row_insert(mprime.shape[0], cMatrix.T)
    # cmatrix forces
    ones = np.ones(countBorder*2)
    zeros = np.zeros(tote)
    cMatrix = Matrix(np.concatenate((zeros, ones), axis=0))
    mprime = mprime.row_insert(mprime.shape[0]+1, cMatrix.T)
    ## Now insert the two corresponding cols
    ones = np.ones(tote)
    zeros = np.zeros(countBorder*2 + 2) # +2 comes from the two inserted rows
    cMatrix = Matrix(np.concatenate((ones, zeros), axis=0))
    mprime = mprime.col_insert(mprime.shape[1], cMatrix)

    ones = np.ones(countBorder*2)
    zeros = np.zeros(tote + 2) # +2 comes from the two inserted rows
    cMatrix = Matrix(np.concatenate((zeros, ones), axis=0))
    mprime = mprime.col_insert(mprime.shape[1], cMatrix)

    # construct the b vector depending if it's a timeseries or not
    print("###########")
    print("Shape: ", m.shape)
    print("###########")
    b = Matrix(np.zeros(m.shape[0]))
    # b = Matrix(np.zeros(mprime.shape[0]))
    if timeseries != None:
        j = 0
        for vid in versorsUsed.keys():
            if term == "timeseries-velocity":
                value = timeseries.calculate_velocity(vid, at_time)
            else: # term == "timeseries-acceleration" is the default
                value = timeseries.calculate_acceleration(vid, at_time)
                if np.any(np.isnan(value)):
                    value = [0, 0]
            b[j] = value[0]
            b[j+1] = value[1]
            j += 2 
    zeros = Matrix(np.zeros(b.shape[1]))
    b = m.T * b
    b = b.row_insert(b.shape[0]+1, zeros)
    b = b.row_insert(b.shape[0]+1, zeros)
    
    b[b.shape[0]-2,b.shape[1]-2] = tote
    b[b.shape[0]-1,b.shape[1]-1] = countBorder*2

    # print("b matrix: ")
    # pprint(b)
    #
    # print("Shapes: ", mprime.shape, b.shape)

    # echelon, ech_pivots = maug.rref(simplify=True, iszerofunc = lambda x: abs(x)<1e-10)
    # rankM = m.rank()

    # print("Calculating SVD...")
    # U, S, V = mp.svd_r(mp.matrix(mprime), full_matrices=True)
    # roundedS = map(lambda p: round(p, 3), S)
    # nums = list(np.nonzero(list(roundedS))[0])
    # newS = []
    # for i in list(map(int, nums)):
    #     newS.append(S[i])
    start = time.time()
    print("NNLS iterating...")
    xres, rnorm = scop.nnls(mprime, mp.matrix(b), maxiter=100000)
    # xres, rnorm = scop.nnls(mprime, b, maxiter=100000)
    end = time.time()
    print("Forsys NNLS finished, took", end-start, "seconds for 100000 iterations")
    # pprint(xres)
    # forces dictionary:
    f = True
    forceDict = {}
    for i in range(0, tote):
        if i < tote:
            # name = "l"+str(i)
            # name = list(d.keys())[list(d.values()).index(i)]
            val = xres[i]
            # name = i
        forceDict[i] = val
        # for i in range(0)
    print("--------------")
    print("--------------")
    i = 0
    for index, value in extForces.items():
        # print("x", xres[tote+i])
        # print("y", xres[tote+i+1])
        name1 = "F"+str(index)+"x"
        name2 = "F"+str(index)+"y"
        val1 = round(xres[tote+i], 3) * extForces[index][0]
        val2 = round(xres[tote+i+1], 3) * extForces[index][1]
        forceDict[name1] = val1
        forceDict[name2] = val2
        i += 2

    # print("\n \n")
    # print("check: ")
    # pprint(mprime * Matrix(xres) - b)
    # print("----------------")
    # print("----------------")
    condNew = 'NA'
    lambdaVal = 'NA'
    aux = pd.DataFrame()
    aux['condition'] = [condNew]
    aux['rnorm'] = [rnorm]
    aux['lambda'] = [lambdaVal]

    # print("Forces dictionary: ")
    # print(forceDict)

    # print("Condition number: ", condNumber, condNew)
    return forceDict, aux, xres, earr, (vertices, edges, cells), versorsUsed

def log_force(fd):
    df = pd.DataFrame.from_dict(fd, orient='index')
    df['is_border'] = [False if isinstance(x, int) else True for x in df.index]
    return df
