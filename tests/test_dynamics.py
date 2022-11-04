import unittest

import numpy as np
import forsys as fs


class TestDynamics(unittest.TestCase):
        
    def setUp(self):
        frames = {}
        vertices = {}
        edges = {}
        cells = {}

        vertices[0] = fs.vertex.Vertex(0, 0, 1)
        vertices[1] = fs.vertex.Vertex(1, 0, 1)
        edges[0] = fs.edge.Edge(0, vertices[0], vertices[1])
        cells[0] = fs.cell.Cell(0, vertices.values(), {})
        frames[0] = fs.frames.Frame(vertices.copy(), edges, cells, time=0)

        vertices[0] = fs.vertex.Vertex(0, 4, 4)
        vertices[1] = fs.vertex.Vertex(1, 0, 1)
        edges[0] = fs.edge.Edge(0, vertices[0], vertices[1])
        cells[0] = fs.cell.Cell(0, vertices.values(), {})
        frames[1] = fs.frames.Frame(vertices.copy(), edges, cells, time=1)

        vertices[0] = fs.vertex.Vertex(0, 12, 11)
        vertices[1] = fs.vertex.Vertex(1, 0, 1)
        edges[0] = fs.edge.Edge(0, vertices[0], vertices[1])
        cells[0] =  fs.cell.Cell(0, vertices.values(), {})
        frames[2] = fs.frames.Frame(vertices.copy(), edges, cells, time=2)
        
        vertices[0] = fs.vertex.Vertex(0, 18, 22)
        vertices[1] = fs.vertex.Vertex(1, 0, 1)
        edges[0] = fs.edge.Edge(0, vertices[0], vertices[1])
        cells[0] = fs.cell.Cell(0, vertices.values(), {})
        frames[3] = fs.frames.Frame(vertices.copy(), edges, cells, time=3)

        self.forsys = fs.ForSys(frames, cm=False)

        self.forsys.mesh.mapping = {0: {0: 0, 1:1}, 2: {0: 0, 1:1}, 1: {0: 0, 1:1}}

    def test_acceleration(self):
        self.assertTrue(np.all(self.forsys.mesh.calculate_acceleration(0, 0) == [4., 4.]))
        self.assertTrue(np.all(self.forsys.mesh.calculate_acceleration(0, 1) == [4., 4.]))
        self.assertTrue(np.all(self.forsys.mesh.calculate_acceleration(0, 2) == [-2., 4.]))
        self.assertTrue(np.all(self.forsys.mesh.calculate_acceleration(0, 3) == [-2., 4.]))

    def test_velocity(self):
        self.assertTrue(np.all(self.forsys.mesh.calculate_velocity(0, 0) == [4., 3.]))
        self.assertTrue(np.all(self.forsys.mesh.calculate_velocity(0, 1) == [8., 7.]))
        self.assertTrue(np.all(self.forsys.mesh.calculate_velocity(0, 2) == [6., 11.]))
        self.assertTrue(np.all(self.forsys.mesh.calculate_velocity(0, 3) == [6., 11.]))