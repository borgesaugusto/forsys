from dataclasses import dataclass, field


@dataclass
class Vertex:
    """
    Class representation of the vertices objects.

    :param id: Unique identifier of the vertex
    :type id: int
    :param x: x-position of the vertex
    :type x: float
    :param y: y-position of the vertex
    :type y: float
    :param ownEdges: List of edges that connect with this vertex
    :type ownEdges: list
    :param ownCells: List of cells that connect with this vertex
    :type ownCells: list
    :param own_big_edges: Unique identifier of the vertex
    :type own_big_edges: int
    """

    id: int
    x: float
    y: float
    ownEdges: list = field(default_factory=list)
    ownCells: list = field(default_factory=list)
    own_big_edges: list = field(default_factory=list)

    def get_coords(self) -> list:
        """
        Return an array with the [x, y] coordinates of the vertex

        :return: [x, y] position of the vertex
        :rtype: list
        """
        return [self.x, self.y]

    def add_cell(self, cid: int) -> bool:
        """
        Add a new cell to the list of cells that include this vertex

        :param cid: ID of the cell to be added
        :type cid: int
        :return: False if the provided ID is already in the list
        :rtype: bool
        """

        if cid in self.ownCells:
            return False
        else:
            self.ownCells.append(cid)
            return True

    def remove_cell(self, cid: int) -> list:
        """
        Remove a given cell from the list of cells that include this vertex

        :param cid: ID of the cell to be removed
        :type cid: int
        :return: List of cells after removal
        :rtype: list
        """

        self.ownCells.remove(cid)
        return self.ownCells

    def add_edge(self, eid: int) -> bool:
        """
        Add a given id to the list of edges that include this vertex

        :param eid: ID of the edge to be added
        :type eid: int
        :return: False if the provided ID is already in the list
        :rtype: bool
        """

        if eid in self.ownEdges:
            # print(eid, self.ownEdges)
            # print("edge already in vertex")
            return False
        else:
            self.ownEdges.append(eid)
            return True

    def remove_edge(self, eid: int) -> list:
        """
        Remove a given edge from the list of edges that include this vertex

        :param eid: ID of the edge to be removed
        :type cid: int
        :return: List of edges after removal
        :rtype: list
        """
        self.ownEdges.remove(eid)
        return self.ownEdges

    def add_big_edge(self, beid: int) -> bool:
        """
        Add a given big edge id to the list of big edges that include this vertex

        :param beid: ID of the big edge to be added
        :type beid: int
        :return: False if the provided ID is already in the list
        :rtype: bool
        """
        if beid in self.own_big_edges:
            print(beid, self.own_big_edges)
            print("edge already in vertex")
            return False
        else:
            self.own_big_edges.append(beid)
            return True

    def remove_big_edge(self, beid: int) -> list:
        """
        Remove a given big edge from the list of big edges that include this vertex

        :param beid: ID of the big edge to be removed
        :type cid: int
        :return: List of big edges after removal
        :rtype: list
        """
        self.own_big_edges.remove(beid)
        return self.own_big_edges
