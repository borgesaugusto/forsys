import numpy as np

def get_versors_area(vertices, edges, cells, vid, metadata):
    try:
        if metadata['area_target'] == 'min':
            areaTarget = get_minmax_area(cells)[0]
        elif metadata['area_target'] == 'max':
            areaTarget = get_minmax_area(cells)[1]
        elif metadata['area_target'] == 'ave':
            areaTarget = get_minmax_area(cells)[2]       
        else:
            areaTarget = metadata['area_target']
    except KeyError:
        print("***WARNING**** No area target defined, using average")
        areaTarget = get_minmax_area(cells)[2]
    
    areax = 0
    areay = 0
    print("Calculating area terms... vID: ", vid, "with area target: ", areaTarget)

    for cellID in vertices[vid].ownCells:
        cell = cells[cellID]
        area = abs(cell.get_area())
        deltay = cell.get_next_vertex(vertices[vid]).y - cell.get_previous_vertex(vertices[vid]).y
        deltax = cell.get_next_vertex(vertices[vid]).x - cell.get_previous_vertex(vertices[vid]).x
        # print("Deltas: ", deltax, deltay)
        # print("Deltas: ", deltax, deltay)
        # print("areas", area, areaTarget, area-areaTarget)
        areax -= (area-areaTarget) * deltay
        areay += (area-areaTarget) * deltax
   
    return areax, areay

def get_versors_perimeter(vertices, edges, cells, vid, metadata):
    try:
        if metadata['perimeter_target'] == 'min':
            perimeterTarget = get_minmax_perimeter(cells)[0]
        elif metadata['perimeter_target'] == 'max':
            perimeterTarget = get_minmax_perimeter(cells)[1]
        elif metadata['perimeter_target'] == 'ave':
            perimeterTarget = get_minmax_perimeter(cells)[2]
        else:
            perimeterTarget = metadata['perimeter_target']
    except KeyError:
        print("No area target defined, using average")
        perimeterTarget = get_minmax_perimeter(cells)[2]

    print("Calculating perimeter terms... vID: ", vid, "with perimeter target: ", perimeterTarget)
    perimeterx = 0
    perimetery = 0
    for cellID in vertices[vid].ownCells:
        cell = cells[cellID]
        perimeter = cell.get_perimeter()
        deltaxPrev = vertices[vid].x - cell.get_previous_vertex(vertices[vid]).x
        deltaxNext = vertices[vid].x - cell.get_next_vertex(vertices[vid]).x
        deltayPrev = vertices[vid].y - cell.get_previous_vertex(vertices[vid]).y
        deltayNext = vertices[vid].y - cell.get_next_vertex(vertices[vid]).y

        print("Deltas: ", deltaxPrev, deltaxNext)
        print("Deltas: ", deltaxNext, deltayNext)
        print("areas", perimeter, perimeterTarget, perimeter-perimeterTarget)
        print("----------------------------------------------------------------------------------------")

        modPrev = np.sqrt(deltaxPrev**2+deltayPrev**2)
        modNext = np.sqrt(deltaxNext**2+deltayNext**2)
        perimeterx += -2*(perimeter-perimeterTarget)*(deltaxPrev/modPrev+deltaxNext/modNext)
        perimetery += -2*(perimeter-perimeterTarget)*(deltayPrev/modPrev+deltayNext/modNext)

    return perimeterx, perimetery


def list_areas(cells):
    areas = []
    for _, cell in cells.items():
        areas.append(cell.get_area())
    return areas

def list_perimeters(cells):
    perim = []
    for _, cell in cells.items():
        perim.append(cell.get_perimeter())
    return perim

def get_minmax_area(cells):
    areas = list_areas(cells)
    return np.min(np.abs(areas)), np.max(np.abs(areas)), np.mean(np.abs(areas))

def get_minmax_perimeter(cells):
    perims = list_perimeters(cells)
    return min(perims), max(perims), np.mean(perims)