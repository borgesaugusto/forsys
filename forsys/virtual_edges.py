
# from itertools import chain
import itertools as itertools
import logging as log
import numpy as np
from collections import Counter

import scipy.optimize as sco

import forsys.vertex as fvertex
import forsys.edge as fedge
import forsys.cell as fcell

from  forsys.exceptions import SegmentationArtifactException, BigEdgesBadlyCreated

def create_edges_new(vertices, cells):
    """
    Create big edges by locating junctions with more that 2 lines connecting
    """
    earr_temp = []
    for cell in cells.values():
        # vertices = {v.id: v for v in cell.vertices}
        cell_vertex_ids = [v.id for v in cell.vertices]
        number_of_connections = [len(n.ownEdges) > 2 for n in cell.vertices]

        splitted = get_splitted(cell_vertex_ids, number_of_connections)
        if not number_of_connections[0]:
            new_vertex_order = np.concatenate((*splitted[1:], splitted[0]))
            new_number_of_connections = [len(vertices[vid].ownEdges) > 2 for vid in new_vertex_order]
            splitted = get_splitted(new_vertex_order, new_number_of_connections)

        splitted = splitted[1:]
        new_edges = [np.append(splitted[ii], splitted[((ii+1)%len(splitted))][0]).flatten().tolist() for ii in range(0, len(splitted))]

    
        earr_temp += new_edges

    earr = []
    for e in earr_temp:
        if e[::-1] not in earr and e not in earr:
            earr += [e]
    return earr
    
def get_splitted(cell_vertex_ids, number_connections):
    indices = [i for i in np.where(number_connections)[0]]
    splitted = np.split(cell_vertex_ids, indices)
    return splitted

def create_edges(cells):
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
    # bedges = create_edges(cells)
    bedges = create_edges_new(vertices, cells)
    # ldict = {}
    nEdgeArray = []
    alreadySeen = []
    vertices_to_join = []
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
            # if edge has only 2 vertices, join them 
            # TODO: Make it work with polygonal vertex model
            if len(e) == 2 and len(vertices[e[0]].ownCells) < 3 and len(vertices[e[1]].ownCells) < 3 :
                vertices_to_join.append(e)
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

    # for eid, e in edges.items():
    #     e.v1.ownEdges.remove(eid)
    #     e.v2.ownEdges.remove(eid)
    # remove all edges
    edges.clear()
    for vi in vertexToRemove:
        del vertices[vi]
    #### from here
    edges = {}
    # now join the vertices
    edgesNumber = 0
    for be in nEdgeArray:
        rango = range(0, len(be)-1)
        for n in rango:
            edges[edgesNumber] = fedge.SmallEdge(edgesNumber, vertices[be[n]], vertices[be[n+1]])
            edgesNumber += 1

    # if there is an empty cell, may be its an artifact, remove it
    cells_to_remove = []
    for c in cells.keys():
        if len(cells[c].vertices) == 0:
            cells_to_remove.append(c)

    for c in cells_to_remove:
        print(f"*** WARNING **** CELL {cells[c].id} remove due to having {len(cells[c].vertices)} vertices")
        del cells[c]
    
    if len(vertices_to_join) > 0:
        try:
            vertices, edges, cells = replace_short_edges(vertices_to_join, vertices, edges, cells)
        except KeyError:
            raise SegmentationArtifactException()

    # exit()

    return vertices, edges, cells, nEdgeArray

def eid_from_vertex(earr, vbel):
    # ret = []
    for j in range(0, len(earr)):
        if len(list(set(earr[j]) & set(vbel))) >= 2:
            return j
    raise BigEdgesBadlyCreated()
    
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

def get_border_from_angles_new(earr, vertices):
    # put the one in the middle
    inBorder = []
    for eid in get_border_edge(earr, vertices):
        # for i in range(1, len(eid) - 1):
        inBorder.append(eid[int(len(eid)/2)])
    return inBorder

def get_big_edges_se(frame):
    """
    Get the big edges by detecting changes in ground truth of Surface Evolver
    """
    new_big_edges = []
    middle_vertices = []

    for bedge in frame.earr:
        edges_to_use = frame.get_big_edge_edgesid(bedge)
        edges_tension = [frame.edges[eid].gt for eid in edges_to_use]

        a = [1 if edges_tension[b] != edges_tension[b+1] else 0 for b in range(0, len(edges_tension)-1)]

        cut_where = np.where(np.array(a).astype(int)==1)[0] + 1
        new_edges = np.split(bedge, cut_where)
        for enumber in range(0, len(new_edges)  - 1):
            new_big_edges.append(list(np.concatenate((new_edges[enumber], [new_edges[enumber+1][0]]))))
            middle_vertices.append(new_edges[enumber+1][0])
        new_big_edges.append(list(new_edges[-1]))

    return new_big_edges, middle_vertices

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
        if eid[0] not in vseen:
            vseen.append(eid[0])
    for vid in vseen:
        virtualEdges[vid] = vertices[vid].ownEdges
        edges.append(virtualEdges[vid])
    edges = set(list(itertools.chain.from_iterable(edges)))
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

def get_circle_params(big_edge, vertices):
    x = [vertices[vid].x for vid in big_edge]
    y = [vertices[vid].y for vid in big_edge]
    # TODO: identificar residuos para ver si excluir
    def objective_f(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        distances = np.sqrt((x - c[0])**2 + (y - c[1])**2)
        return distances - distances.mean()

    center_2, _ = sco.leastsq(objective_f, (np.mean(x), np.mean(y)))
    return center_2[0], center_2[1]



def get_versors(vertices, edges, vid):
    for eb in vertices[vid].ownEdges:
        # get versors average of the next two
        edge = edges[eb]
        deltaX = edge.v2.x - edge.v1.x
        deltaY = edge.v2.y - edge.v1.y
        modulus = np.sqrt(deltaX**2 + deltaY**2)      
        # check if the edge is in the border, and don't create a lambda if it is
        other_vertex = list(set(edge.get_vertices_id()) - set([vid]))
        if len(other_vertex) == 1:
            border = len(vertices[other_vertex[0]].ownCells) == 1
        else:
            border = False
        if vid == edge.get_vertices_id()[0]:
            vx = deltaX/modulus
            vy = deltaY/modulus
        else:
            vx = - deltaX/modulus
            vy = - deltaY/modulus
        # Force versor, to calculate modulus
        yield (vx, vy, edge.get_vertices_id(), border)

def get_border_edges(earr, vertices, edges):
    border_edges = []
    for e in earr:      
        if len(vertices[e[int(len(e)/2)]].ownCells) == 1:
            border_edges.append(e)
    return border_edges



def get_versors_average(vertices, edges, vid):
    for eb in vertices[vid].ownEdges:
        edge = edges[eb]

        other_vertex_id = list(set(edge.get_vertices_id()) - set([vid]))
        
        if len(other_vertex_id) == 1:
            border = len(vertices[other_vertex_id[0]].ownCells) == 1
        else:
            border = False

        # Next vertex
        next_edge = edges[list(set(vertices[other_vertex_id[0]].ownEdges) - set([eb]))[0]]
        
        vector1 = edge.get_vector()

        # deltaX = edge.v1.x - edge.v2.x
        # deltaY = edge.v1.y - edge.v2.y
        # modulus = np.sqrt(deltaX**2 + deltaY**2)

        sign1 = 1
        sign2 = 1
        if vid != edge.get_vertices_id()[0]:
            sign1 = -1
        if next_edge.get_vertices_id()[0] != other_vertex_id[0]:
            sign2 = -1


        vector2 = next_edge.get_vector()


        ret_versor = sign1 * np.array(vector1) + sign2 * np.array(vector2)
        ret_versor = ret_versor / np.linalg.norm(ret_versor)
        
        # yield (ave_versor[0], ave_versor[1], edge.get_vertices_id(), border)
        yield (ret_versor[0], ret_versor[1], edge.get_vertices_id(), border)


def angle_between_two_vectors(a, b):
    # anorm = np.(a, 2)
    # b = np.round(b, 2)
    clipped_dot_product = np.clip(np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b)), -1, 1)
    return np.arccos(clipped_dot_product)

def replace_short_edges(vertices_to_join, vertices, edges, cells):
    """ Join the two-vertex edges"""
    for e in vertices_to_join:
        x_cm = abs(vertices[e[0]].x + vertices[e[1]].x) / 2
        y_cm = abs(vertices[e[0]].y + vertices[e[1]].y) / 2

        # find the common edge
        common_edge = list(set(vertices[e[0]].ownEdges) & set(vertices[e[1]].ownEdges))[0]

        # TODO: Move this to a function
        new_id = len(vertices)
        i = 0
        while vertices.get(new_id) != None:
            new_id = len(vertices) + i
            i += 1

        new_vertex = fvertex.Vertex(new_id, x_cm, y_cm)
        vertices[new_id] = new_vertex
        # replace vertex in the cells
        # def replace_vertex(self, vold: object, vnew: object):
        for cid in vertices[e[0]].ownCells:
            cells[cid].replace_vertex(vertices[e[0]], new_vertex)
        for cid in vertices[e[1]].ownCells:
            cells[cid].replace_vertex(vertices[e[1]], new_vertex)

        # destroy edge
        del edges[common_edge]
        # replace vertex in the edges
        edges[vertices[e[0]].ownEdges[-1]].replace_vertex(vertices[e[0]], new_vertex)
        edges[vertices[e[0]].ownEdges[-1]].replace_vertex(vertices[e[0]], new_vertex)
        edges[vertices[e[1]].ownEdges[-1]].replace_vertex(vertices[e[1]], new_vertex)
        edges[vertices[e[1]].ownEdges[-1]].replace_vertex(vertices[e[1]], new_vertex)
        
        # destroy the two previous vertices
        del vertices[e[0]]
        del vertices[e[1]]

    return vertices, edges, cells

