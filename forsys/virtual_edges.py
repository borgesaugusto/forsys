
# from itertools import chain
import itertools as itertools
import logging as log
import numpy as np
import itertools as it
from collections import Counter

import forsys.vertex as fvertex
import forsys.edge as fedge
import forsys.cell as fcell

def create_edges(vertices, edges, cells):
    """
    Give virtual edges generated from the lattice
    """
    newEdges = []
    # print("Cells: ", len(cells))
    for _, cell in cells.items():
        vertexInEdge = []
        # print(cID, cell.get_cell_vertices())
        for v in cell.vertices:
            # find the first junction to iterate
            # vertex = vertices[vID]
            if len(v.ownEdges) > 2:
                start = cell.vertices.index(v)
                goArr = cell.vertices[start:]+cell.vertices[:start]+[cell.vertices[start]]
                # goArr = cell.vertices[start:]+cell.vertices[:start]+[cell.vertices[start]]
                # goArr = cell.get_cell_vertices()[start:]+cell.get_cell_vertices()[:start]+[cell.get_cell_vertices()[start]]
                # goArr = cell.get_cell_vertices()[:start]+cell.get_cell_vertices()[start:-1]+[cell.get_cell_vertices()[-1]]
                break
        try:
            if len(goArr) != len(cell.vertices)+1:
                print("ERROR: Different goArr size than cell: %d goArr vs %d cell_vertices in"
                            % (len(goArr),len(cell.vertices)))
                # print("goArr is: ", goArr )
                # print("goArr is: ", cell.get_cell_vertices())
                print("#########################")

            for v in goArr:
                if len(v.ownEdges) < 3:
                    vertexInEdge.append(v.id)
                else:
                    if len(vertexInEdge) != 0:
                        vertexInEdge.append(v.id)
                        # Check if that edge is already there
                        if not (vertexInEdge in newEdges) and not(vertexInEdge[::-1] in newEdges):
                            newEdges.append(vertexInEdge)
                    vertexInEdge = [v.id]
        except NameError:
            print("cell with no neighbour")
    return newEdges

def generate_mesh(vertices, edges, cells, ne=4):
    bedges = create_edges(vertices, edges, cells)
    # ldict = {}
    nEdgeArray = []
    alreadySeen = []
    edgeRange = range(0, ne)
    # print(bedges)
    for e in bedges:
        if len(e) > ne:
            if not e in alreadySeen or not reversed(e) in alreadySeen:
                each = len(e)/ne
                nEdge = []
                for i in edgeRange:
                    nEdge.append(e[int(each*i)])
                nEdge.append(e[-1])
                nEdgeArray.append(nEdge)
                alreadySeen.append(e)
        else:
            nEdgeArray.append(e)
            alreadySeen.append(e)
    # cellsToRemove = []
    # edgesToRemove = []
    vertexToRemove = []
    for vID, v in vertices.items():
        if not vID in list(itertools.chain.from_iterable(nEdgeArray)):
            if not vID in vertexToRemove:
                vertexToRemove.append(vID)
            # for eid in v.getEdgesBelonging():
            #     if not eid in edgesToRemove:
            #         edgesToRemove.append(eid)
            for cid in v.ownCells:
                cells[cid].vertices.remove(v)
                # cellsToRemove.append(cid)

    for eid, e in edges.items():
        e.v1.ownEdges.remove(eid)
        e.v2.ownEdges.remove(eid)
    # remove all edges
    del edges

    for vi in vertexToRemove:
        del vertices[vi]

    edges = {}
    # now join the vertices
    edgesNumber = 0
    for be in nEdgeArray:
        rango = range(0, len(be)-1)
        for n in rango:
            edges[edgesNumber] = fedge.Edge(edgesNumber, vertices[be[n]], vertices[be[n+1]])
            edgesNumber += 1


    return vertices, edges, cells, nEdgeArray

def eid_from_vertex(earr, vbel):
    # ret = []
    for j in range(0, len(earr)):
        if len(list(set(earr[j]) & set(vbel))) >= 2:
            return j
    log.warning("No big edge with both vectors")

def get_border_edge(earr, vertices):
    borderEdge = []
    for eid in earr:
        # print("Edge ", eid)
        for vid in eid:
            if len(vertices[vid].ownCells) < 2:
                borderEdge.append(eid)
                break
    return borderEdge

def get_border_from_angles(earr, vertices):
    inBorder = []
    bedge = get_border_edge(earr, vertices)
    for eid in bedge:
        # print("Now in ", eid)
        # Choose pivot vector
        anglesArr = []
        for i in range(1, len(eid) - 1):
            v1 = (vertices[eid[i-1]].x - vertices[eid[i]].x,
                vertices[eid[i-1]].y - vertices[eid[i]].y)
            v2 = (vertices[eid[i+1]].x - vertices[eid[i]].x,
                vertices[eid[i+1]].y - vertices[eid[i]].y)

            anglesArr.append(angle_between_two_vectors(v1, v2))
            # print(eid[i], " with angle ", round(np.degrees(anglesArr[-1]), 1))
        diff = 0
        for a, b in itertools.combinations(anglesArr, 2):
            if abs(a-b) > 0.3:
            # if abs(a-b) > np.pi/6:
                diff += 1
        if diff == 2 or diff==0:
            inBorder.append(eid[int(len(eid)/2)])
        else:
            inBorder.append(eid[1+anglesArr.index(min(anglesArr))])
    return inBorder

def new_virtual_edges(inBorder, earr):
    newEarr = []
    for eid in earr:
        border = False
        for vid in eid:
            if vid in inBorder:
                newEarr.append(earr[earr.index(eid)][:eid.index(vid)]+[vid])
                newEarr.append(earr[earr.index(eid)][eid.index(vid):])
                border = True
        if not border:
            newEarr.append(eid)
    return newEarr

def get_virtual_edges(earrN, vertices):
    vseen = []
    virtualEdges = {}
    edges = []
    for eid in earrN:
        vuse = []
        if eid[0] not in vseen:
            vseen.append(eid[0])
    for vid in vseen:
        virtualEdges[vid] = vertices[vid].ownEdges
        edges.append(virtualEdges[vid])
    edges = set(list(it.chain.from_iterable(edges)))
    edgesNumber = len(edges)

    return virtualEdges, edges, edgesNumber

def get_edge_from_earr(edges, element):
    for eid, eobj in edges.items():
        if Counter(eobj.get_vertices_id()) == Counter(element):
            return eid

def get_line_constant_from_bedge(edges, ele):
    edgeLC = []
    # print("Element: ", ele)
    for i in range(0, len(ele)-1):
        # print("Slice: ", ele[i:i+2])
        eid = get_edge_from_earr(edges, ele[i:i+2])
        edgeLC.append(edges[eid].getLineConstant())
    return edgeLC

def get_versors(vertices, edges, vid):
    for eb in vertices[vid].ownEdges:
        edge = edges[eb]
        deltaX = edge.v1.x - edge.v2.x
        deltaY = edge.v1.y - edge.v2.y
        modulus = np.sqrt(deltaX**2 + deltaY**2)
        if vid == edge.get_vertices_id()[0]:
            vx = - deltaX/modulus
            vy = - deltaY/modulus
        else:
            vx = deltaX/modulus
            vy = deltaY/modulus
        # Force versor, to calculate modulus
        yield (vx, vy, edge.get_vertices_id())


def angle_between_two_vectors(a, b):
    return np.arccos(np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b)))