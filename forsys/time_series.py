from dataclasses import dataclass, field
import numpy as np
import json

from  forsys.exceptions import DifferentTissueException

import forsys.vertex as fvertex

### create a mapping between all meshes in a time series
@dataclass
class TimeSeries():
    time_series: dict
    cm: bool=True
    initial_guess: list = field(default_factory=list)

    def __post_init__(self):
        self.cutoff = 0.05
        self.accelerations = {}
        self.velocities = {}
        if self.initial_guess == []:
            # self.initial_guess = [{} for _ in range(0, len(self.mesh))]
            self.initial_guess = {k: {} for k in self.time_series.keys()}

        # self.cm_coords = {}            
        # for ii in self.mesh.keys():
        #     self.cm_coords[ii] = self.get_cm_coords(self.mesh[ii].vertices)

        self.mapping = {}
        # for ii in range(0, len(self.mesh) - 1) :
        for key in self.time_series.keys():
            # key = str(ii)+"-"+str(ii+1)
            print("meshing ", key)
            t0 = self.time_series[key]
            # print("t0: ", t0)
            try:
                t1 = self.time_series[key+1]
            except KeyError:
                break
            # print("t1: ", t1)
            try:
                self.mapping[key] = self.create_mapping(t0, t1, self.initial_guess[key])
            except DifferentTissueException:
                print("Tissues between ", key, " and ", key+1, " too different, skipping...")
                self.mapping[key] = None


    def export_mapping(self, where='.'):
        with open(where, 'w') as f:
            json.dump(self.mapping, f)

    def get_cm_coords(self, vertices):
        return np.around([np.mean([v.x for v in vertices.values()]), 
                        np.mean([v.y for v in vertices.values()])], 3)

    def create_mapping(self, t0, t1, initial_guess):
        # Mesh should have time two times, t0 and t1
        # If cm = True, move all vertices to the center of mass
        if self.cm:
            self.cm_coords = {}
            self.cm_coords[0] = np.around([np.mean([v.x for v in t0.vertices.values()]), 
                                np.mean([v.y for v in t0.vertices.values()])], 3)
            self.cm_coords[1] = np.around([np.mean([v.x for v in t1.vertices.values()]), 
                                np.mean([v.y for v in t1.vertices.values()])], 3)
            
            for _, v in t0.vertices.items():
                v.x -= self.cm_coords[0][0]
                v.y -= self.cm_coords[0][1]
            for _, v in t1.vertices.items():
                v.x -= self.cm_coords[1][0]
                v.y -= self.cm_coords[1][1]    

        # Only use the real vertices
        realVertices0_ids = np.array([[x[0]]+[x[-1]] for x in t0.earr])
        realVertices1_ids = np.array([[x[0]]+[x[-1]] for x in t1.earr])

        externalVertices0_ids = np.array(t0.border_vertices)
        externalVertices1_ids = np.array(t1.border_vertices)

        # take the correspondind dictionary
        rvertices0 = {key: value for key, value in t0.vertices.items() if key in realVertices0_ids.flatten()}
        rvertices1 = {key: value for key, value in t1.vertices.items() if key in realVertices1_ids.flatten()}
        evertices0 = {key: value for key, value in t0.vertices.items() if key in externalVertices0_ids.flatten()}
        evertices1 = {key: value for key, value in t1.vertices.items() if key in externalVertices1_ids.flatten()}

        # should we use all coordinates ?
        xcoords0 = [v.x for v in evertices0.values()]
        ycoords0 = [v.y for v in evertices0.values()]
        xcoords1 = [v.x for v in evertices1.values()]
        ycoords1 = [v.y for v in evertices1.values()]
        # self.maxcoord = max(xcoords0+ycoords0+xcoords1+ycoords1)
        if len(xcoords0) > 0:
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
                len(t0.cells) > len(t0.cells) + 1 or \
                len(t1.cells) > len(t1.cells) + 1:           
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

        # find mapping between externals
        for v0 in evertices0.values():
            if v0.id not in mapping.keys():
                try:
                    mapping[v0.id] = self.find_best(v0, evertices1, mapping.values()).id
                except AttributeError:
                    mapping[v0.id] = None
        return mapping
    
    def find_best(self, v0, pool, found):
        candidatesObverse = []
        candidatesInverse = []
        spread = 0.005
        while len(candidatesObverse) <= 1 and spread < self.cutoff:
            maxspread = spread * self.maxcoord
            for v1 in pool.values():
                # make the map inyective
                if v1.id not in found:
                    xcoord = (v1.x - v0.x)**2
                    ycoord = (v1.y - v0.y)**2
                    if xcoord + ycoord < maxspread**2:
                        candidatesObverse.append(v1)
            spread += spread
        while len(candidatesInverse) <= 1 and spread < self.cutoff:
            for v1 in reversed(pool.values()):
                # make the map inyective
                if v1.id not in found:
                    xcoord = (v1.x - v0.x)**2
                    ycoord = (v1.y - v0.y)**2
                    if xcoord + ycoord < maxspread**2:
                        candidatesInverse.append(v1)
            spread += spread

        candidates = np.concatenate((candidatesInverse, candidatesObverse))
        if len(candidates) == 0:
            best = None
        # from candidates pick the closest
        elif len(candidates) == 1:
            best = candidates[0]
        else:
            distances = [self.distance(v0, vc) for vc in candidates]
            best = candidates[distances.index(min(distances))]
        return best
    
    @staticmethod
    def distance(v0: object, v1: object) -> float:
        return np.sqrt((v1.x - v0.x)**2 + (v1.y - v0.y)**2)

    def calculate_velocity(self, point: int, initial_time: str) -> list:
        v0 = self.time_series[initial_time].vertices[point]
        if initial_time == len(self.time_series) - 1:
            # last time, use backward
            if self.mapping[initial_time - 1] == None:
                raise DifferentTissueException()
            tt1 = initial_time - 1       
        else:
            tt1 = initial_time + 1

            if self.mapping[initial_time] == None:
                raise DifferentTissueException()


        try:
            v1 = self.time_series[tt1].vertices[self.get_point_id_by_map(point, initial_time, tt1)]
        except KeyError:
            # fictious vertex to give velocity zero
            v1 = fvertex.Vertex(-1, v0.x, v0.y)
        

        try:
            ti = self.time_series[int(initial_time)].time
            tf = self.time_series[tt1].time
        except KeyError:
            tf = 1
            ti = 0

        return (np.array([v1.x, v1.y]) - np.array([v0.x, v0.y])) / (tf - ti)

    def whole_tissue_acceleration(self, initial_time: int) -> dict:
        t0 = self.time_series[initial_time]
        for ev in t0.earr:
            acc0 = self.calculate_acceleration(ev[0], initial_time)
            acc1 = self.calculate_acceleration(ev[-1], initial_time)
            self.accelerations[t0.earr.index(ev)] = np.linalg.norm(acc0) + np.linalg.norm(acc1)
        return self.accelerations

    def whole_tissue_velocity(self, initial_time: int) -> dict:
        t0 = self.time_series[initial_time]
        for ev in t0.earr:
            vel0 = self.calculate_velocity(ev[0], initial_time)
            vel1 = self.calculate_velocity(ev[-1], initial_time)
            self.velocities[t0.earr.index(ev)] = np.linalg.norm(vel0) + np.linalg.norm(vel1)
        return self.velocities


    def calculate_acceleration(self, point: int, initial_time: int) -> list:
        # This requires three timepoints
        # TODO What if the timepoints are not equally distributed ? (if using finite differences)
        t0 = self.time_series[initial_time]        
        if initial_time == len(self.time_series) - 1:
            # last time, use backward
            tt1 = initial_time - 1
            tt2 = initial_time - 2            
        elif initial_time == 0:
            # first time, use forward
            tt1 = initial_time + 1
            tt2 = initial_time + 2
        else:
            # Middle, use central
            tt1 = initial_time + 1
            tt2 = initial_time - 1
        
        t1 = self.time_series[tt1]
        t2 = self.time_series[tt2]
        v0 = t0.vertices[point]

        try:
            v1 = t1.vertices[self.get_point_id_by_map(point, initial_time, tt1)]
            v2 = t2.vertices[self.get_point_id_by_map(point, initial_time, tt2)]
            
            f0 = np.array([v0.x, v0.y])
            f1 = np.array([v1.x, v1.y])
            f2 = np.array([v2.x, v2.y])

            # TODO: Generalize to non consecutive times ?
            if initial_time == len(self.time_series) - 1:
                acceleration = f0 - 2 * f1 + f2
            elif initial_time == 0:
                acceleration = f2 - 2 * f1 + f0
            else:
                acceleration = f1 - 2 * f0 + f2


            # print("Accelerations: ", f0, fh, f2h, "Total: ", acceleration)
        except KeyError:
            # that vertex has no counterpart at a later time
            acceleration = np.nan

        try:
            ti = t0.time
            tf = t2.time
        except KeyError:
            tf = 1
            ti = 0
        return acceleration

    def get_point_id_by_map(self, point, initial_time, final_time):
        """"
        Get the ID of a point between initial and final time
        """
        if initial_time < final_time:
            timerange = np.arange(initial_time, final_time, 1)
        else:
            timerange = np.arange(final_time, initial_time, 1)[::-1]
        
        for ii in timerange:
            if initial_time < final_time:
                tempMapping = self.mapping[ii]
            else:
                tempMapping = {v: k for k, v in self.mapping[ii].items()}
            if point == None or tempMapping == None:
                break
            point = tempMapping[point]            
        
        return point
        
    def acceleration_per_edge(self, earrid, initial_time, final_time):
        """ Return the acceleration of the earrid in the given timeframe"""
        t0 = self.time_series[initial_time]

        earr = t0.earr[earrid]
        timerange = np.arange(initial_time, final_time, 1)
        row = []
        for ii in timerange:
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
        return row

    def velocity_per_edge(self, earrid, initial_time, final_time):
        """ Return the acceleration of the earrid in the given timeframe"""
        t0 = self.time_series[initial_time]

        earr = t0.earr[earrid]
        timerange = np.arange(initial_time, final_time, 1)
        row = []
        for ii in timerange:
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

    def times_to_use(self, last_frame=False):
        final_times = [-1]
        if not last_frame:
            t_range = range(0, len(self.time_series) - 1)
        else:
            t_range = range(0, last_frame-1)
            
        for t in t_range:
            if self.mapping[t] == None:
                final_times.append(t)
        final_times.append(t+1)
        return final_times

    def get_vertex_position(self, v0, t0, tmax):
        """"
        Get the position of vertex from time t0 to tmax
        """
        positions_x = []
        positions_y = []
        if tmax == -1:
            range_values = self.time_series.keys()
        else:
            range_values = range(t0, tmax)
        for t in range_values:
            print(f"Vertex {v0} is now {self.get_point_id_by_map(v0, t0, t)}")
            v_now = self.time_series[t].vertices[self.get_point_id_by_map(v0, t0, t)]
            positions_x.append(v_now.x)
            positions_y.append(v_now.y)

        return positions_x, positions_y
