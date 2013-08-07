

from .graph import Graph


class Layout(object):
    pass

A = Layout()
A.local = [False, False, True]

B = Layout()
B.local = [False, True, False]

C = Layout()
C.local = [True, False, False]


def local(i):
    test = lambda layout: layout.local[i]
    return test


graph = Graph()
graph.add_vertex(A)
graph.add_vertex(B)
graph.add_vertex(C)

graph.add_edge(A, B, 'A2B')
graph.add_edge(B, A, 'B2A')

graph.add_edge(B, C, 'B2C')
graph.add_edge(C, B, 'C2B')

graph.add_edge(A, C, 'A2C')


print graph.find_edge_paths(A, local(0))

