

import numpy as np


class Graph(object):
    """Basic directed graph."""

    def __init__(self):

        # Store edges as dictionary: {start: {end: edge, ...}, ...}
        self.graph = {}

    def add_vertex(self, vertex):

        # Create empty edge dictionary
        if vertex not in self.graph:
            self.graph[vertex] = {}

    def add_edge(self, start, end, edge):

        # Add start and end vertices
        # Note: Adding the end vertex is required, otherwise the search methods
        # will fail if there are any dead-end vertices.
        self.add_vertex(start)
        self.add_vertex(end)

        # Add edge
        self.graph[start][end] = edge

    def find_vertex_paths(self, start, test, path=[]):

        # Add start to path (keyword used in recursion)
        path = path + [start]

        # Check test
        if test(start):
            # Pass: done
            paths = [path]
        else:
            # Fail: iterate over connected nodes
            paths = []
            for node in self.graph[start]:
                # Only examine unvisited nodes
                if node not in path:
                    # Recurse
                    subpaths = self.find_vertex_paths(node, test, path)
                    for sp in subpaths:
                        paths.append(sp)

        return paths

    def find_edge_paths(self, start, test):

        # Get vertex paths
        vertex_paths = self.find_vertex_paths(start, test)

        # Iterate over vertex paths
        paths = []
        for vp in vertex_paths:
            # Loop through vertices
            path = []
            n_edges = len(vp) - 1
            for i in xrange(n_edges):
                # Get edge from graph
                current = vp[i]
                next = vp[i+1]
                path.append(self.graph[current][next])
            paths.append(path)

        return paths

    def find_shortest_path(self, start, test, weight=len):

        # Get edge paths
        paths = self.find_edge_paths(start, test)

        # Check if any paths exist
        # Note: this is to distinguish between no possible paths (paths = [])
        # and the trivial/empty path (paths = [[]]), i.e. where start satisfies test.
        if len(paths) == 0:
            raise ValueError("No paths found.")

        # Check weights (counts edges by default)
        cost = [weight(p) for p in paths]

        # Return shortest
        shortest = np.argmin(cost)

        return paths[shortest]

