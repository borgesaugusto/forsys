from dataclasses import dataclass, field
import numpy as np
import json

from  forsys.exceptions import DifferentTissueException

import forsys.vertex as fvertex

### create a mapping between all meshes in a time series
@dataclass
class TimeSeries():
    mesh: dict
    cm: bool=False
    initial_guess: list = field(default_factory=list)

    def __post_init__(self):
        self.cutoff = 0.1
        self.accelerations = {}
        self.velocities = {}
        if self.initial_guess == []:
            # self.initial_guess = [{} for _ in range(0, len(self.mesh))]
            self.initial_guess = {k: {} for k in self.mesh.keys()}

        self.mapping = {}
        # for ii in range(0, len(self.mesh) - 1) :
        for ii in self.mesh.keys():
            # key = str(ii)+"-"+str(ii+1)
            key = ii
            print("meshing ", key)
            t0 = self.mesh[ii]
            # print("t0: ", t0)
            try:
                t1 = self.mesh[ii+1]
            except KeyError:
                break
            # print("t1: ", t1)
            try:
                self.mapping[key] = self.create_mapping(t0, t1, self.initial_guess[ii])
            except DifferentTissueException:
                print("Tissues between ", ii, " and ", ii+1, " too different, skipping...")
                self.mapping[key] = None


    def export_mapping(self, where='.'):
        with open(where, 'w') as f:
            json.dump(self.mapping, f)


    def create_mapping(self, t0, t1, initial_guess):
        # Mesh should have time two times, t0 and t1
        # If cm = True, move all vertices to the center of mass
        if self.cm:
            self.cm_coords = {}
            self.cm_coords[0] = np.around([np.mean([v.x for v in t0['vertices'].values()]), 
                                np.mean([v.y for v in t0['vertices'].values()])], 3)
            self.cm_coords[1] = np.around([np.mean([v.x for v in t1['vertices'].values()]), 
                                np.mean([v.y for v in t1['vertices'].values()])], 3)

            for _, v in t0['vertices'].items():
                v.x -= self.cm_coords[0][0]
                v.y -= self.cm_coords[0][1]
            for _, v in t1['vertices'].items():
                v.x -= self.cm_coords[1][0]
                v.y -= self.cm_coords[1][1]    

        # Only use the real vertices
        realVertices0_ids = np.array([[x[0]]+[x[-1]] for x in t0['earr']])
        realVertices1_ids = np.array([[x[0]]+[x[-1]] for x in t1['earr']])

        externalVertices0_ids = np.array(t0['external'])
        externalVertices1_ids = np.array(t1['external'])

        # take the correspondind dictionary
        rvertices0 = {key: value for key, value in t0['vertices'].items() if key in realVertices0_ids.flatten()}
        rvertices1 = {key: value for key, value in t1['vertices'].items() if key in realVertices1_ids.flatten()}
        evertices0 = {key: value for key, value in t0['vertices'].items() if key in externalVertices0_ids.flatten()}
        evertices1 = {key: value for key, value in t1['vertices'].items() if key in externalVertices1_ids.flatten()}

        # should we use all coordinates ?
        xcoords0 = [v.x for v in evertices0.values()]
        ycoords0 = [v.y for v in evertices0.values()]
        xcoords1 = [v.x for v in evertices1.values()]
        ycoords1 = [v.y for v in evertices1.values()]
        # self.maxcoord = max(xcoords0+ycoords0+xcoords1+ycoords1)
        minx = min(xcoords0+xcoords1)
        maxx = max(xcoords0+xcoords1)
        miny = min(ycoords0+ycoords1)
        maxy = max(ycoords0+ycoords1)
        self.maxcoord = max(maxx - minx, maxy - miny)

        maxDifference = 0.10
        xshape0 = max(xcoords0) - min(xcoords0)
        yshape0 = max(ycoords0) - min(ycoords0)
        xshape1 = max(xcoords1) - min(xcoords1)
        yshape1 = max(ycoords1) - min(ycoords1)

        displacement = np.sqrt((xshape1 - xshape0)**2 + (yshape1 - yshape0)**2)

        if displacement > maxDifference * self.maxcoord or \
            len(t0['cells']) > len(t0['cells']) + 1 or \
            len(t1['cells']) > len(t1['cells']) + 1:           
            raise DifferentTissueException()

        mapping = {}
        # Add the initial guess
        # py3.5+ compatible
        mapping = {**mapping, **initial_guess}

        # find mapping between real vertices
        for v0 in rvertices0.values():
            if v0.id not in mapping.keys():
                try:
                    mapping[v0.id] = self.find_best(v0, rvertices1, mapping.values()).id
                except AttributeError:
                    mapping[v0.id] = None

        # Map vertices in-between 
        edges0 = t0['earr']
        edges1 = t1['earr']
        
        edgeMap = {}
        for edge in edges0:
            # first and last should be mapped already.
            # find the edge in the new map with the first and last
            vfirst = mapping[edge[0]]
            vlast = mapping[edge[-1]]

            # print("Real at t+1", realVertices1_ids)
            mask = [True if (e[0] == vfirst and e[-1] == vlast) or (e[0] == vlast and e[-1] == vfirst) 
                        else False for e in realVertices1_ids]
            if not np.any(mask):
                # probably a T1 happened if we didn't lose a vertex, so two vertices
                # should be interchanged
                # print("########################")
                # print("Possible T1, no mask ...")
                # print("Edge:", edge)
                # print("First and last (t=0):", edge[0], edge[-1])
                # print("First and last (t=1):", vfirst, vlast)
                maskT11 = [True if (e[0] == vfirst or e[-1] == vlast) or
                                    (e[0] == vlast or e[-1] == vfirst)
                    else False for e in realVertices1_ids]

                # print("There are ", maskT11.count(True), " trues")
                # print("New mask: ", maskT11)
                
                # if there is more than one possible, pick the closest edge(to the 
                # average edge center)
                # todo: use least squares with the "line" of the edge? 
                distancesT1 = {}
                # for jj in range(0, len(edges1)):
                #     if maskT11[jj]:
                #         print("Possible edge: ", edges1[jj])
                # edge center at t=0
                xcoords = [t0['vertices'][e].x for e in edge]
                ycoords = [t0['vertices'][e].y for e in edge]
                center0 = [np.mean(xcoords), np.mean(ycoords)]
                # print("At t=0, edge center is: ", center0)

                for jj in range(0, len(edges1)):
                    if maskT11[jj]:
                        #print("Using: ", edge[1], " and ", edges1[jj][1] )
                        # edge center of this edge
                        xcoords1 = [t1['vertices'][e].x for e in edges1[jj]]
                        ycoords1 = [t1['vertices'][e].y for e in edges1[jj]] 
                        center1 = [np.mean(xcoords1), np.mean(ycoords1)]
                        distancesT1[jj] = np.sqrt(
                                                (center0[0] - center1[0])**2 +
                                                (center0[1] - center1[1])**2)
                        # distancesT1[jj] = distance(
                        #     mesh[0]['vertices'][edge[1]], 
                        #     mesh[1]['vertices'][edges1[jj][1]])
                # print(distancesT1)
                
                try:
                    selectedEdge = edges1[min(distancesT1, key=distancesT1.get)]
                except ValueError:
                    raise DifferentTissueException()
                    
                # print("Selected edge: ", selectedEdge)
                # print("########################")
                # exit()
                # replace the original with the new last
                mask = [True if (e[0] == selectedEdge[0] and e[-1] == selectedEdge[-1]) or 
                                (e[0] == selectedEdge[-1] and e[-1] == selectedEdge[0]) 
                        else False for e in realVertices1_ids]

            edgeMap[edges0.index(edge)] = mask.index(True)
            # edgeMap[edges0.index(edge)] = np.where(realVertices1_ids == [vfirst, vlast])
            # transform vertex in between
            newEdge = edges1[edgeMap[edges0.index(edge)]]
            # print(edge, " maps to ", newEdge)
            # if edge == [1820, 1821, 1822, 1826, 1830, 1339]:
            #     exit()
            # if I have less vertices map what we can, if it's 2 nothing to do
            diff = len(newEdge) - len(edge)
            absDiff = abs(diff)
            if len(newEdge) == 2:
                # Join everyone to None, except the sides
                for ii in range(1, len(edge) - 1):
                    mapping[edge[ii]] = None
            else:           
                # define ranges
                # print("DIff: ", diff)
                if diff < 0:
                    if absDiff % 2 == 0:
                        firsts = range(1, int(absDiff/2) + 1)
                        middle = range(int(absDiff/2)+1, (len(edge) - 1) - int(absDiff/2))
                    else:
                        firsts = range(1, int(absDiff/2) + 2)
                        middle = range(int(absDiff/2)+2, (len(edge) - 1) - int(absDiff/2))
                        
                    last = range((len(edge) - 1) - int(absDiff/2), len(edge)-1)
                    # print("newEdge: ", newEdge)
                    # print("Edge: ", edge)
                    # print("len(newEdge) < len(edge)")

                    # print("Joining: ", list(firsts), "with None")
                    # print("Joining: ", list(middle), "with the rest")
                    # print("Joining: ", list(last), "with None")
                    
                    for ii in firsts:
                        mapping[edge[ii]] = None

                    for ii in last:
                        mapping[edge[ii]] = None

                    for ii in middle:
                        mapping[edge[ii]] = newEdge[ii-int(absDiff/2)]
                elif diff > 0:
                    # print("len(newEdge) > len(edge)")
                    firsts = range(1, int(absDiff/2))
                    middle = range(int(absDiff/2), len(edge) - 1 - int(absDiff/2))
                    lasts = range(len(edge) - 1 - int(absDiff/2), len(edge) - 1)
                    # ojo!!!
                    for ii in firsts:
                        mapping[edge[ii]] = None

                    for ii in lasts:
                        mapping[edge[ii]] = None


                    # print("Mapping in the range: ", middle)

                    for ii in middle:
                        mapping[edge[ii]] = newEdge[ii]

                else:
                    for ii in range(1, len(edge) - 1):
                        mapping[edge[ii]] = newEdge[ii]
                

        return mapping
    
    def find_best(self, v0, pool, found):
        # cutoff = 0.05
        candidatesObverse = []
        candidatesInverse = []
        spread = 0.01
        while len(candidatesObverse) <= 1 and spread < self.cutoff:
            maxspread = spread * self.maxcoord
            for v1 in pool.values():
                # make the map inyective
                if v1.id not in found:
                    xcoord = (v1.x - v0.x)**2
                    ycoord = (v1.y - v0.y)**2
                    # if v1.x > minx  and v1.x < v0.x + maxspread and v1.y > v0.y - maxspread and v1.y < v0.y + maxspread:
                    # if v0.id == 1758 and v1.id== 1904:
                    #     print("1758 position", v0.x, v0.y)
                    #     print("1904 position", v1.x, v1.y)
                    #     print("v1.x - v0.x", v1.x - v0.x)
                    #     print("v1.y - v0.y", v1.y - v0.y)
                    #     print(xcoord + ycoord, maxspread**2)
                    if xcoord + ycoord < maxspread**2:
                        candidatesObverse.append(v1)
            spread += 0.01
        # print("Inverse ....")
        spread = 0.01  
        while len(candidatesInverse) <= 1 and spread < self.cutoff:
            for v1 in reversed(pool.values()):
                # make the map inyective
                if v1.id not in found:
                    xcoord = (v1.x - v0.x)**2
                    ycoord = (v1.y - v0.y)**2
                    # if v1.x > minx  and v1.x < v0.x + maxspread and v1.y > v0.y - maxspread and v1.y < v0.y + maxspread:
                    if xcoord + ycoord < maxspread**2:
                        candidatesInverse.append(v1)
            spread += 0.01
        # if v0.id == 1958:
        #     print("Obverse spread", candidatesObverse)
        #     print("Inverse spread", candidatesInverse)

        candidates = np.concatenate((candidatesInverse, candidatesObverse))
        # candidates = candidatesObverse
        if len(candidates) == 0:
            best = None
            # print("One solitary!!!!: ", v0.id)
        # from candidates pick the closest
        elif len(candidates) == 1:
            best = candidates[0]
        else:
            # distances = [np.sqrt((vc.x - v0.x)**2 + (vc.y - v0.y)**2) for vc in candidates]
            distances = [self.distance(v0, vc) for vc in candidates]
            best = candidates[distances.index(min(distances))]
        return best
    
    @staticmethod
    def distance(v0: object, v1: object) -> float:
        return np.sqrt((v1.x - v0.x)**2 + (v1.y - v0.y)**2)

    def calculate_velocity(self, point: int, initial_time: str) -> list:
        # Before mapping, check if they are too different
        # mapping = self.create_mapping(between)
        # times = between.split("-")
        v0 = self.mesh[initial_time]['vertices'][point]
        if self.mapping[initial_time] == None:
            raise DifferentTissueException()
        if self.mapping[initial_time][point] == None:
            # fictious vertex to give velocity zero
            v1 = fvertex.Vertex(-1, v0.x, v0.y)
        else:
            v1 = self.mesh[int(initial_time+1)]['vertices'][self.mapping[initial_time][point]]
        try:
            ti = self.mesh[int(initial_time)]['time']
            tf = self.mesh[int(initial_time+1)]['time']
        except KeyError:
            tf = 1
            ti = 0

        return (np.array([v1.x, v1.y]) - np.array([v0.x, v0.y])) / (tf - ti)

    def whole_tissue_acceleration(self, initial_time: int) -> dict:
        t0 = self.mesh[initial_time]
        for ev in t0['earr']:
            # print("##################")
            # print("For:" , ev)
            acc0 = self.calculate_acceleration(ev[0], initial_time)
            acc1 = self.calculate_acceleration(ev[-1], initial_time)
            # print("Acc0: ", acc0, np.linalg.norm(acc0))
            # print("Acc1: ", acc1, np.linalg.norm(acc1))
            self.accelerations[t0['earr'].index(ev)] = np.linalg.norm(acc0) + np.linalg.norm(acc1)

        #     print("Norm value: ", self.accelerations[t0['earr'].index(ev)])
        # print("##################")
        
        return self.accelerations

    def whole_tissue_velocity(self, initial_time: int) -> dict:
        t0 = self.mesh[initial_time]
        for ev in t0['earr']:
            vel0 = self.calculate_velocity(ev[0], initial_time)
            vel1 = self.calculate_velocity(ev[-1], initial_time)
            self.velocities[t0['earr'].index(ev)] = np.linalg.norm(vel0) + np.linalg.norm(vel1)
        return self.velocities


    def calculate_acceleration(self, point: int, initial_time: int) -> list:
        # This requires three timepoints, except at t0
        # What if the timepoints are not equally distributed ? (if using finite differences)
        if self.mapping[initial_time] == None or self.mapping[initial_time + 1] == None:
            # print("Not enough timepoints in the mesh to calculate acceleration")
            raise DifferentTissueException()

        t0 = self.mesh[initial_time]
        try:
            t1 = self.mesh[initial_time + 1]
            t2 = self.mesh[initial_time + 2]
        except KeyError:
            # print("Not enough timepoints in the mesh to calculate acceleration")
            exit()

        v0 = t0['vertices'][point]
        try:
            v1 = t1['vertices'][self.mapping[initial_time][point]]
            v2 = t2['vertices'][self.get_point_id_by_map(point, initial_time, initial_time + 2)]
            # v2 = t2['vertices'][self.mapping[initial_time + 1][self.mapping[initial_time][point]]]
                
            f0 = np.array([v0.x, v0.y])
            fh = np.array([v1.x, v1.y])
            f2h = np.array([v2.x, v2.y])

            # print("Coordinates: ", [v0.x, v0.y], [v1.x, v1.y], [v2.x, v2.y])
            

            # TODO: Generalize to non consecutive times ? 
            acceleration = f2h - 2 * fh + f0
            # print("Accelerations: ", f0, fh, f2h, "Total: ", acceleration)
        except KeyError:
            # that vertex has no counterpart at a later time
            acceleration = np.nan

        return acceleration

    def get_point_id_by_map(self, point, initial_time, final_time):
        timerange = np.arange(initial_time, final_time, 1)
        for ii in timerange:
            # between = str(ii)+"-"+str(ii+1)
            # print("Checking ", ii, "-",ii+1, " at ", point)
            if point == None or self.mapping[ii] == None:
                break
            # if point == 108:
            #     print(ii, self.mapping, self.mapping[ii])
            #     print(self.mapping[ii][point])
            point = self.mapping[ii][point]
        return point
        
    def acceleration_per_edge(self, earrid, initial_time, final_time):
        """ Return the acceleration of the earrid in the given timeframe"""
        t0 = self.mesh[initial_time]

        earr = t0['earr'][earrid]
        timerange = np.arange(initial_time, final_time, 1)
        # accelerationMatrix = np.zeros((len(timerange), len(earr)))
        row = []
        for ii in timerange:
            # print(earr)
            id0 = self.get_point_id_by_map(earr[0], initial_time, ii)
            id1 = self.get_point_id_by_map(earr[-1], initial_time, ii)
            if id0 == None or id1 == None:
                acc = np.nan
            else:
                try:
                    acc0 = self.calculate_acceleration(id0, ii)
                    acc1 = self.calculate_acceleration(id1, ii)
                    acc = np.linalg.norm(acc0) + np.linalg.norm(acc1)
                except DifferentTissueException:
                    acc = np.nan

            row.append(acc)
        # if np.all(np.isnan(row[-1])):
        #     row = row[:-1]
        return row

    def velocity_per_edge(self, earrid, initial_time, final_time):
        """ Return the acceleration of the earrid in the given timeframe"""
        t0 = self.mesh[initial_time]

        earr = t0['earr'][earrid]
        timerange = np.arange(initial_time, final_time, 1)
        # accelerationMatrix = np.zeros((len(timerange), len(earr)))
        row = []
        for ii in timerange:
            # print(earr)
            id0 = self.get_point_id_by_map(earr[0], initial_time, ii)
            id1 = self.get_point_id_by_map(earr[-1], initial_time, ii)
            if id0 == None or id1 == None:
                vel = np.nan
            else:
                try:
                    vel0 = self.calculate_velocity(id0, ii)
                    vel1 = self.calculate_velocity(id1, ii)
                    vel = np.linalg.norm(vel0) + np.linalg.norm(vel1)
                except DifferentTissueException:
                    vel = np.nan

            row.append(vel)
        return row


