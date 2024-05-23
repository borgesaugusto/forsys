import itertools as itertools
import numpy as np
from collections import Counter
from typing import Tuple

import scipy.optimize as sco
import circle_fit as cfit
import forsys.vertex as fvertex
import forsys.edge as fedge

from forsys.exceptions import SegmentationArtifactException, BigEdgesBadlyCreated


def create_edges_new(vertices, cells):
    """
    Create big edges by locating junctions with more that 2 lines connecting
    """
    earr_temp = []
    for cell in cells.values():
        # vertices = {v.id: v for v in cell.vertices}
        cell_vertex_ids = [v.id for v in cell.vertices]
        number_of_connections = [len(n.ownEdges) > 2 for n in cell.vertices]

        partitioned = get_partition(cell_vertex_ids, number_of_connections)
        if not number_of_connections[0]:
            new_vertex_order = np.concatenate((*partitioned[1:], partitioned[0]))
            new_number_of_connections = [len(vertices[vid].ownEdges) > 2 for vid in new_vertex_order]
            partitioned = get_partition(new_vertex_order, new_number_of_connections)

        partitioned = partitioned[1:]
        new_edges = [np.append(partitioned[ii], partitioned[((ii + 1) % len(partitioned))][0]).flatten().tolist() for ii in
                     range(0, len(partitioned))]

        earr_temp += new_edges

    earr = []
    for e in earr_temp:
        if e[::-1] not in earr and e not in earr:
            earr += [e]
    return earr


def get_partition(cell_vertex_ids, number_connections):
    indices = [i for i in np.where(number_connections)[0]]
    partitioned = np.split(cell_vertex_ids, indices)
    return partitioned


def generate_mesh(vertices, edges, cells, ne=4, **kwargs):
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
                each = len(e) / ne
                nEdge = []
                for i in edgeRange:
                    nEdge.append(e[int(each * i)])
                nEdge.append(e[-1])
                nEdgeArray.append(nEdge)
                alreadySeen.append(e)
        else:
            nEdgeArray.append(e)
            alreadySeen.append(e)
            # if edge has only 2 vertices, join them 
            # TODO: Make it work with polygonal vertex model
            if (len(e) == 2 and
                len(vertices[e[0]].ownCells) < 3 and
                len(vertices[e[1]].ownCells) < 3):
                vertices_to_join.append(e)

    vertexToRemove = []
    for vID, v in vertices.items():
        if not vID in list(itertools.chain.from_iterable(nEdgeArray)):
            if not vID in vertexToRemove:
                vertexToRemove.append(vID)
            for cid in v.ownCells:
                cells[cid].vertices.remove(v)
                # cellsToRemove.append(cid)

    # remove all edges
    edges.clear()
    for vi in vertexToRemove:
        del vertices[vi]
    #### from here
    edges = {}
    # now join the vertices
    edgesNumber = 0
    for be in nEdgeArray:
        rango = range(0, len(be) - 1)
        for n in rango:
            edges[edgesNumber] = fedge.SmallEdge(edgesNumber, vertices[be[n]], vertices[be[n + 1]])
            edgesNumber += 1

    # if there is an empty cell, may be its an artifact, remove it
    cells_to_remove = []
    for c in cells.keys():
        if len(cells[c].vertices) == 0:
            cells_to_remove.append(c)

    for c in cells_to_remove:
        print(f"*** WARNING **** CELL {cells[c].id} remove due to having {len(cells[c].vertices)} vertices")
        del cells[c]

    # if len(vertices_to_join) > 0 and kwargs.get("replace_short_edges", True):
    mapper = {}
    if kwargs.get("replace_short_edges", True):
        for edge_to_join in vertices_to_join:
            try:
                vertices, edges, cells, new_mapper = join_two_vertices(edge_to_join, vertices, edges, cells, mapper)
                mapper = mapper | new_mapper
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
        # Choose pivot vector
        anglesArr = []
        for i in range(1, len(eid) - 1):
            v1 = (vertices[eid[i - 1]].x - vertices[eid[i]].x,
                  vertices[eid[i - 1]].y - vertices[eid[i]].y)
            v2 = (vertices[eid[i + 1]].x - vertices[eid[i]].x,
                  vertices[eid[i + 1]].y - vertices[eid[i]].y)

            anglesArr.append(angle_between_two_vectors(v1, v2))
        diff = 0
        for a, b in itertools.combinations(anglesArr, 2):
            if abs(a - b) > 0.3:
                diff += 1
        if diff == 2 or diff == 0:
            inBorder.append(eid[int(len(eid) / 2)])
        else:
            inBorder.append(eid[1 + anglesArr.index(min(anglesArr))])
    return inBorder


def get_border_from_angles_new(earr, vertices):
    # put the one in the middle
    inBorder = []
    for eid in get_border_edge(earr, vertices):
        # for i in range(1, len(eid) - 1):
        inBorder.append(eid[int(len(eid) / 2)])
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

        a = [1 if edges_tension[b] != edges_tension[b + 1] else 0 for b in range(0, len(edges_tension) - 1)]

        cut_where = np.where(np.array(a).astype(int) == 1)[0] + 1
        new_edges = np.split(bedge, cut_where)
        for enumber in range(0, len(new_edges) - 1):
            new_big_edges.append(list(np.concatenate((new_edges[enumber], [new_edges[enumber + 1][0]]))))
            middle_vertices.append(new_edges[enumber + 1][0])
        new_big_edges.append(list(new_edges[-1]))

    return new_big_edges, middle_vertices


def non_border_big_edges(inBorder, earr):
    newEarr = []
    for eid in earr:
        border = False
        for vid in eid:
            if vid in inBorder:
                newEarr.append(earr[earr.index(eid)][:eid.index(vid)] + [vid])
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

    return virtualEdges, edges


def get_edge_from_earr(edges, element):
    for eid, eobj in edges.items():
        if Counter(eobj.get_vertices_id()) == Counter(element):
            return eid


def get_line_constant_from_bedge(edges, ele):
    edgeLC = []
    # print("Element: ", ele)
    for i in range(0, len(ele) - 1):
        # print("Slice: ", ele[i:i+2])
        eid = get_edge_from_earr(edges, ele[i:i + 2])
        edgeLC.append(edges[eid].getLineConstant())
    return edgeLC


def calculate_circle_center(vertices, method: str="dlite") -> Tuple:
        """
        Calculate the centroid of the cell through a circle fit 
        dlite invokes a modified from DLITE's implementation: 
        https://github.com/AllenCellModeling/DLITE
        if no argument is passed, the geometric center is returned

        :return: x and y coordiante of the cell's center from a circular fit
        :rtype: Tuple
        """        
        # print("Using as method: ", method)
        xs = [v.x for v in vertices]
        ys = [v.y for v in vertices]

        if method == "dlite":
            center = dlite_circle_method(xs, ys)
        elif method == "taubinSVD":
            zipped_coords = list(zip(xs, ys))
            if len(zipped_coords) < 3:
                center = dlite_circle_method(xs, ys)
            else:
                try:
                    xc, yc, r, sigma = cfit.taubinSVD(zipped_coords)
                    center = [xc, yc]
                except FloatingPointError:
                    center = dlite_circle_method(xs, ys)
        else:
            center = [np.mean(xs),  np.mean(ys)]
        
        return center[0], center[1]

def dlite_circle_method(xs, ys):
    def objective_f(c):
        """ 
        Distance between the vertices and the mean circle centered at c
        """
        distances = np.sqrt((xs - c[0]) ** 2 + (ys - c[1]) ** 2)
        return distances - distances.mean()

    center, _ = sco.leastsq(objective_f, (np.mean(xs), np.mean(ys)))
    return center



def get_versors(vertices, edges, vid):
    for eb in vertices[vid].ownEdges:
        # get versors average of the next two
        edge = edges[eb]
        deltaX = edge.v2.x - edge.v1.x
        deltaY = edge.v2.y - edge.v1.y
        modulus = np.sqrt(deltaX ** 2 + deltaY ** 2)
        # check if the edge is in the border, and don't create a lambda if it is
        other_vertex = list(set(edge.get_vertices_id()) - set([vid]))
        if len(other_vertex) == 1:
            border = len(vertices[other_vertex[0]].ownCells) == 1
        else:
            border = False
        if vid == edge.get_vertices_id()[0]:
            vx = deltaX / modulus
            vy = deltaY / modulus
        else:
            vx = - deltaX / modulus
            vy = - deltaY / modulus
        # Force versor, to calculate modulus
        yield (vx, vy, edge.get_vertices_id(), border, edge.id)


def get_border_edges(earr, vertices, edges):
    border_edges = []
    for e in earr:
        if len(vertices[e[int(len(e) / 2)]].ownCells) == 1:
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
    clipped_dot_product = np.clip(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)), -1, 1)
    return np.arccos(clipped_dot_product)


# def replace_short_edges(vertices_to_join, vertices, edges, cells):
def join_two_vertices(vertices_to_join, vertices, edges, cells, mapper={}):
    """ Join the two-vertex edges"""
    try:
        v0 = vertices[vertices_to_join[0]]
    except KeyError:
        v0 = vertices[mapper[vertices_to_join[0]]]
    try:
        v1 = vertices[vertices_to_join[1]]
    except KeyError:
        v1 = vertices[mapper[vertices_to_join[1]]]
    
    x_cm = abs(v0.x + v1.x) / 2
    y_cm = abs(v0.y + v1.y) / 2

    # find the common edge
    common_edge = list(set(v0.ownEdges) & set(v1.ownEdges))[0]

    new_id = get_unused_id(vertices)        
    new_vertex = fvertex.Vertex(new_id, x_cm, y_cm)
    vertices[new_id] = new_vertex
    mapper[vertices_to_join[0]] = new_id
    mapper[vertices_to_join[1]] = new_id
    # replace vertex in the cells
    list_of_cells_0 = [cid for cid in v0.ownCells]
    list_of_cells_1 = [cid for cid in v1.ownCells]
    for cid in list_of_cells_0:
        cells[cid].replace_vertex(vertices[v0.id], new_vertex)
    for cid in list_of_cells_1:
        cells[cid].replace_vertex(vertices[v1.id], new_vertex)
    # destroy edge
    del edges[common_edge]
    # replace vertex in the edges
    list_of_edges_0 = [eid for eid in v0.ownEdges]
    list_of_edges_1 = [eid for eid in v1.ownEdges]
    for edge_id in list_of_edges_0:
        edges[edge_id].replace_vertex(vertices[v0.id], new_vertex)
    for edge_id in list_of_edges_1:
        edges[edge_id].replace_vertex(vertices[v1.id], new_vertex)

    # destroy the two previous vertices
    del vertices[v0.id]
    del vertices[v1.id]

    return vertices, edges, cells, mapper


def get_unused_id(dictionary):
    new_id = len(dictionary)
    i = 0
    while dictionary.get(new_id) != None:
        new_id = len(dictionary) + i
        i += 1
    return new_id