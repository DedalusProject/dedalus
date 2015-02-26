

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from ..core.future import Future


class Node:

    def __init__(self, label, level):

        # Plotting label
        self.label = label

        # Position information
        self.level = level
        self.position = 0.

    def __repr__(self):

        return self.label


class Leaf(Node):

    count = 0.

    def __init__(self, *args):

        Node.__init__(self, *args)

        # Set position based on leaf count
        self.position = Leaf.count + 1.
        Leaf.count += 1.


class Tree:

    def __init__(self, operator):

        # Define node and branch containers
        self.nodes = []
        self.branches = defaultdict(list)

        # Build tree
        root = self.build(operator, 0)

        # Set positions
        self.set_position(root)

    def build(self, arg, level):

        # Recursively construct nodes and add branches
        if isinstance(arg, Future):
            node = Node(arg.name, level)
            for a in arg.args:
                self.branches[node].append(self.build(a, level+1))
        else:
            node = Leaf(str(arg), level)

        # Add node
        self.nodes.append(node)

        return node

    def set_position(self, node):

        # Set node positions to mean of sub node positions
        if not isinstance(node, Leaf):
            sub_pos = [self.set_position(sub) for sub in self.branches[node]]
            node.position = np.mean(sub_pos)

        return node.position


def plot_operator(operator, fontsize=8, figsize=8, saveas=None):

    # Create tree
    tree = Tree(operator)

    # Create figure
    fig = plt.figure(1, figsize=(figsize, figsize))
    fig.clear()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Store node positions in lists
    x = []
    y = []

    for node in tree.nodes:

        # Add node positions
        x.append(node.position)
        y.append(-node.level)
        s = node.label

        # Plot branches to sub nodes
        for sub in tree.branches[node]:
            sx = sub.position
            sy = -sub.level
            plt.plot([x[-1], sx], [y[-1], sy], '-k', alpha=0.5, zorder=0)

        # Plot circle at node
        if isinstance(node, Leaf):
            fc = '#9CBA7F'
        else:
            fc = '#B4CDCD'
        c = plt.Circle((x[-1], y[-1]), radius=0.2, fc=fc, ec='k', zorder=1)
        fig.gca().add_artist(c)

        # Plot node label
        plt.text(x[-1], y[-1], s, fontsize=fontsize, zorder=2,
            verticalalignment='center', horizontalalignment='center')

    # Set limits
    plt.axis(pad(*plt.axis(), pad=0.5, square=True))
    plt.axis('off')

    # Save
    if saveas:
        plt.savefig(saveas, dpi=200)


def pad(xmin, xmax, ymin, ymax, pad=0., square=False):

    xcenter = (xmin + xmax) / 2.
    ycenter = (ymin + ymax) / 2.
    xradius = (xmax - xmin) / 2.
    yradius = (ymax - ymin) / 2.

    if square:
        xradius = yradius = max(xradius, yradius)

    xradius *= (1. + pad)
    yradius *= (1. + pad)

    xmin = xcenter - xradius
    ymin = ycenter - yradius
    xmax = xcenter + xradius
    ymax = ycenter + yradius

    return [xmin, xmax, ymin, ymax]

