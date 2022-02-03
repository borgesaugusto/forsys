from dataclasses import dataclass


from dataclasses import dataclass, field
import numpy as np

class BigEdge():
    
    # vertices: list = field(default_factory=list)
    # edges: list = field(default_factory=list)

    vertices: list
    edges: list

    curve: list = field(default_factory=list)

    def __post_init__(self):
        # construct array of (x, y) points in a [[x0, y0], [x1, y1], ...] fashion
        for v in self.vertices:
            self.curve.append([v.x, v.y])


    def calculate_curvature(self):
            dxdt = np.gradient(self.curve[:, 0])
            dydt = np.gradient(self.curve[:, 1])
            # velocity = zip(dxdt, dydt)
            # dsdt = np.sqrt(dxdt**2 + dydt**2)

            # tangent = np.array(((1/dsdt) * 2)).transpose() * velocity

            # print("Sanity check: ", np.sqrt(tangent[:, 0]**2 + tangent[:, 1]**2))

            # dtandx = np.gradient(tangent[:, 0])
            # dtandy = np.gradient(tangent[:, 1])

            d2xdt2 = np.gradient(dxdt)
            d2ydt2 = np.gradient(dydt)
            curvature = np.abs(d2xdt2 * dydt - dxdt * d2ydt2) / (dxdt**2 + dydt**2)**1.5

            return curvature

