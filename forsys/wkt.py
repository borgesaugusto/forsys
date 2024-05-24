import numpy as np
import pandas as pd
import os

import forsys.vertex as fvertex
import forsys.edge as fedge
import forsys.cell as fcell

def create_lattice(wkt):
    """
    Creates the lattice, give a wkt formated file, and returns
    vertex edges and cells as dictionaries to input into Lattice.
    """
    vertices = {}
    edges = {}
    cells = {}
    verticesNumber = 0
    edgesNumber = 0
    cellsNumber = 0
    edArr = []
    for cellNonParsed in wkt:
        # print("Creating cell ", cellsNumber)
        cellVertices = []
        arrPosition = []
        cell = read_vertexes(cellNonParsed)
        for vID in range(0, len(cell)-1):
            v = cell[vID].split(' ')
            # Check if the first vertex exists, if doesn't exist, create it
            oldVertex = is_vertex_created(v, vertices)
            if type(oldVertex) != int :  # Doesn't exist
                # vertices[verticesNumber] = fvertex.Vertex(verticesNumber, float(v[1]), float(v[2]))
                vertices[verticesNumber] = fvertex.Vertex(verticesNumber, float(v[1]), 1024 - float(v[2]))
                arrPosition.append(verticesNumber)
                verticesNumber += 1
            else:  # Save the global ID
                arrPosition.append(oldVertex)
        
        # vertexCellID = 0
        for v in range(0, len(arrPosition)):
            i = arrPosition[v]
            try:
                j = arrPosition[v+1]
            except IndexError:
                j = arrPosition[0]

            # cellVertices[vertexCellID] = vertices[i]
            cellVertices.append(vertices[i])
            # vertexCellID += 1
            if not (i, j) in edArr and not (j, i) in edArr:
                # edges[edgesNumber] = e.Edge(edgesNumber, vertices[i], vertices[j])
                edges[edgesNumber] = fedge.SmallEdge(edgesNumber, vertices[i], vertices[j])
                edArr.append((i, j))
                edgesNumber += 1
        # cellType = {'typeID':0,'targetArea': 600,'targetPerimeter':100,'areaElasticityConstant':1/2,'perimeterElasticityConstant':0.5/2,'cellHeight':1}
        cellType = {}
        cells[cellsNumber] = fcell.Cell(cellsNumber, cellVertices, cellType)
        cellsNumber += 1

    # vertices, edges, cells = reduce_amount(vertices, edges, cells)

    return vertices, edges, cells


def reduce_amount(vertices, edges, cells):
    for _, cell in cells.items():
        for j in cell.vertices:
            i = cell.get_next_vertex(j)
            # j = nv
            k = cell.get_previous_vertex(j)

            ipos = i.get_coords()
            jpos = j.get_coords()
            kpos = k.get_coords()

            deltaX1 = jpos[0]-ipos[0]
            deltaY1 = jpos[1]-ipos[1]
            deltaX2 = kpos[0]-jpos[0]
            deltaY2 = kpos[1]-jpos[1]

            if deltaX1 != 0:
                slope1 = deltaY1/deltaX1
            else:
                slope1 = np.inf
            if deltaX2 != 0:
                slope2 = deltaY2/deltaX2
            else:
                slope2 = np.inf
                
            if slope1 == slope2:
                toRemove = None
                toReplace = None

                if len(j.ownCells) < 3 and len(j.ownEdges) < 3:
                    for e in j.ownEdges:
                        if i.id in edges[e].get_vertices_id():
                            toReplace = e
                        elif k.id in edges[e].get_vertices_id():
                            toRemove = e
                    # edges[toRemove].v1.ownEdges.remove(toRemove)
                    # edges[toRemove].v2.ownEdges.remove(toRemove)
                    del edges[toRemove]
                    edges[toReplace].replace_vertex(j, k)
                    for c in j.ownCells:
                        cells[c].vertices.remove(j)
                    del vertices[j.id]

    return vertices, edges, cells


def read_vertexes(row):
    var = row.split('((')[1].split(',')
    var[0] = ' '+var[0]
    var[-1] = var[-1][:-2]
    # var = var[:-1] # Why is this line necessary ??
    return var


def is_vertex_created(v, vertices):
    # Check all vertices to see if we already have one in that position
    for _, vertex in vertices.items():
        if float(v[1]) == float(vertex.x) and 1024 - float(v[2]) == float(vertex.y):
            return vertex.id
    return False

def create_wkt(cells):
    """
    Creates a WKT-formated string from a lattice.
    POLYGON ((x0 y0, x1 y1, x2 y2))
    go through each vertex in a cell and save the position with round(*, 3)
    """
    final = str()
    for _, cell in cells.items():
        string = str()
        for v in cell.vertices:
          string += f"{str(v.x)} {str(v.y)}, "
        string = string[:-2]
        final += "POLYGON ((" + string+ "))"+"\n"

    return final

def save(wkt, folder="log/",step=0):
    """ Save the current state """
    with open(os.path.join(folder, str(step)+".wkt"), 'w') as f:
        f.write(wkt)
    # return wkt.to_csv(folder+"wkt-"+str(step)+".wkt", index=False, header=None)
