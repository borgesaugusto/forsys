import numpy as np
import scipy.optimize as scop
import pandas as pd
from mpmath import mp
from sympy import *

import forsys.virtual_edges as eforce
import forsys.fmatrix as fmatrix
import forsys.cmatrix as cmatrix

import time

def equations(vertices, edges, cells):
    m, extForces, earr, versorsUsed, (vertices, edges, cells) = cmatrix.cmatrix(vertices, edges, cells)
    countBorder = len(extForces)
    totv = m.shape[0]
    tote = m.shape[1] - countBorder*2
    shapeM = m.shape

    print(shapeM)
    print("Unkowns: ", tote)
    print("Equations: ", totv)
    ###
    mprime = m.T * m
    cMatrix = Matrix(np.ones(tote))
    mprime = mprime.row_insert(mprime.shape[0], cMatrix.T)
    cMatrix = Matrix(np.ones(tote+1))
    mprime = mprime.col_insert(mprime.shape[1], cMatrix)
    # cmatrix forces
    mprime[mprime.shape[0]-1,mprime.shape[1]-1] = 0
    b = Matrix(np.zeros(mprime.shape[0]))
    b[b.shape[0]-1,b.shape[1]-1] = tote
    ##
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
    end = time.time()
    print("Forsys NNLS finished, took", end-start, "seconds for 100000 iterations")
    # pprint(xres)
    # forces dictionary:
    f = True
    forceDict = {}
    lambdaVal = xres[-1]
    xres = xres[:-1]
    for i in range(0, tote):
        if i < tote:
            val = xres[i]
        forceDict[i] = val
        # for i in range(0)
    print("--------------")
    print("--------------")
    condNew = 'NA'
    aux = pd.DataFrame()
    aux['condition'] = [condNew]
    aux['rnorm'] = [rnorm]
    aux['lambda'] = [lambdaVal]
    # print("Condition number: ", condNumber, condNew)
    return forceDict, aux, xres, earr, (vertices, edges, cells), versorsUsed

def log_force(fd, cnumber):
    df = pd.DataFrame.from_dict(fd, orient='index')
    return df
