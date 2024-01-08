{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Tutorial 1: Coordinates, Distributors, and Bases"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Overview:** This tutorial covers the basics of setting up and interacting with coordinate, distributor, and basis objects in Dedalus.\n",
    "Dedalus uses spectral discretizations to represent fields and solve PDEs.\n",
    "These discretizations are specified by selecting spectral bases for the spatial coordinates of a problem.\n",
    "The set of coordinates in a problem are used to construct a distributor object that controls how fields and problems are distributed in parallel."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import dedalus.public as d3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "%config InlineBackend.figure_format = 'retina'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1.1: Coordinates"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The spatial coordinates in a PDE are represented by coordinate objects in Dedalus.\n",
    "Individual coordinates can be defined using the `Coordinate` class, which is primarily used for 1D problems.\n",
    "In multiple dimensions, multiple coordinates can be constructed together as a `CoordinateSystem`.\n",
    "The currently included coordinate systems are:\n",
    "\n",
    "* `CartesianCoordinates` of any dimension.\n",
    "* `PolarCoordinates` with azimuth and radius.\n",
    "* `S2Coordinates` with azimuth and colatitude.\n",
    "* `SphericalCoordinates` with azimuth, colatitude, and radius.\n",
    "\n",
    "When instantiating a `CoordinateSystem`, we provide the names we'd like to use for each coordinate, ordered as described in the list above.\n",
    "Let's follow an example of setting up a problem in 3D Cartesian coordinates."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "coords = d3.CartesianCoordinates('x', 'y', 'z')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1.2: Distributors"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "`Distributor` objects direct the parallel decomposition of fields and problems, and are needed for all problems, even when running in serial.\n",
    "To build a distributor, we pass the coordinate system of our PDE, specify the datatype of the fields we'll be using, and optionally specify a process mesh for parallelization."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "dist = d3.Distributor(coords, dtype=np.float64) # No mesh for serial / automatic parallelization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Parallelization & process meshes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "When running in an MPI environment, Dedalus uses block-distributed domain decompositions to parallelize computations.\n",
    "By default, problems are distributed across a 1-dimensional mesh of all the available MPI processes (i.e. a \"slab\" decomposition).\n",
    "However, arbitrary process meshes (i.e. \"pencil\" decompositions) can be used by specifying a mesh shape with the `mesh` keyword when instantiating a domain.\n",
    "The current MPI environment must have the same number of processes as the product of the mesh shape.\n",
    "\n",
    "The mesh dimension must be less than or equal to the number of separable coordinates in the linear part of our PDE.\n",
    "Typically this means problems can be parallelized over any periodic and angular coordinates.\n",
    "However, for fully separable problems (like fully periodic boxes or simulations on the sphere), the mesh dimension must be less than the total dimension."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Layouts"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The distributor object builds the machinery necessary for the parallelized allocation and transformation of fields.  This includes an ordered set of `Layout` objects describing the necessary transform/distribution states of the data between coefficient space and grid space.\n",
    "Subsequent layouts are connected by spectral transforms, which must be performed locally, and global transposes (rearrangements of the data distribution across the process mesh) to achieve the required locality.\n",
    "\n",
    "The general algorithm starts from coefficient space (layout 0), with the last axis local (non-distributed).\n",
    "It proceeds towards grid space by transforming the last axis into grid space (layout 1).\n",
    "Then it globally transposes the data (if necessary) such that the penultimate axis is local, transforms that axis into grid space, etc., until all axes have been transformed to grid space (the last layout).\n",
    "\n",
    "Let's examine the layouts for the domain we just constructed.\n",
    "For serial computations such as this, no global transposes are necessary (all axes are always local), and the paths between layouts consist of coefficient-to-grid transforms, backwards from the last axis:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Layout 0:  Grid space: [False False False]  Local: [ True  True  True]\n",
      "Layout 1:  Grid space: [False False  True]  Local: [ True  True  True]\n",
      "Layout 2:  Grid space: [False  True  True]  Local: [ True  True  True]\n",
      "Layout 3:  Grid space: [ True  True  True]  Local: [ True  True  True]\n"
     ]
    }
   ],
   "source": [
    "for layout in dist.layouts:\n",
    "    print('Layout {}:  Grid space: {}  Local: {}'.format(layout.index, layout.grid_space, layout.local))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To see how things work for a distributed simulation, we'll change the specified process mesh shape and rebuild the layout objects, circumventing the internal checks on the number of available processes, etc.\n",
    "\n",
    "**Note this is for demonstration only... messing with these attributes will generally break things.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Don't do this. For illustration only.\n",
    "dist.mesh = np.array([4, 2])\n",
    "dist.comm_coords = np.array([0, 0])\n",
    "dist._build_layouts(dry_run=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Layout 0:  Grid space: [False False False]  Local: [False False  True]\n",
      "Layout 1:  Grid space: [False False  True]  Local: [False False  True]\n",
      "Layout 2:  Grid space: [False False  True]  Local: [False  True False]\n",
      "Layout 3:  Grid space: [False  True  True]  Local: [False  True False]\n",
      "Layout 4:  Grid space: [False  True  True]  Local: [ True False False]\n",
      "Layout 5:  Grid space: [ True  True  True]  Local: [ True False False]\n"
     ]
    }
   ],
   "source": [
    "for layout in dist.layouts:\n",
    "    print('Layout {}:  Grid space: {}  Local: {}'.format(layout.index, layout.grid_space, layout.local))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can see that there are now two additional layouts, corresponding to the transposed states of the mixed-transform layouts.\n",
    "Two global transposes are necessary here in order for the $y$ and $x$ axes to be stored locally, which is required to perform the respective spectral transforms.\n",
    "Here's a sketch of the data distribution in the different layouts:\n",
    "\n",
    "<img src=\"fig_layouts_fold.png\" width=\"800px\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Interacting with the layout objects directly is typically not necessary, but being aware of this system for controlling the distribution and tranformation state of data is important for interacting with field objects, as we'll see in future notebooks."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1.3: Bases"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Creating a basis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Each type of basis in Dedalus is represented by a separate class.\n",
    "These classes define the corresponding spectral operators as well as transforms between the \"grid space\" and \"coefficient space\" representations of functions in that basis.\n",
    "The most commonly used bases are:\n",
    "\n",
    "* `RealFourier` for real periodic functions on an interval using cosine & sine modes.\n",
    "* `ComplexFourier` for complex periodic functions on an interval using complex exponentials.\n",
    "* `Chebyshev` for functions on an interval.\n",
    "* `Jacobi` for functions on an interval under a more general inner product (usually `Chebyshev` is best for performance).\n",
    "* `DiskBasis` for functions on a full disk in polar coordinates.\n",
    "* `AnnulusBasis` for functions on an annulus in polar coordinates.\n",
    "* `SphereBasis` for functions on the 2-sphere in S2 or spherical coordinates.\n",
    "* `BallBasis` for functions on a full ball in spherical coordinates.\n",
    "* `ShellBasis` for functions on a spherical shell in spherical coordinates.\n",
    "\n",
    "The one-dimensional / Cartesian bases are instantiated with:\n",
    "\n",
    "* the corresponding coordinate object,\n",
    "* the number of modes for the basis,\n",
    "* the coordinate bounds of the basis interval.\n",
    "\n",
    "The multidimensional / curvilinear bases are instantiated with:\n",
    "\n",
    "* the corresponding coordinate system,\n",
    "* the multidimensional mode shape for the basis, \n",
    "* the radial extent of the basis, \n",
    "* the problem dtype.\n",
    "\n",
    "Optionally, for all bases you can specify dealiasing scale factors for each basis axis, indicating how much to pad the included modes when transforming to grid space.\n",
    "To properly dealias quadratic nonlinearities, for instance, you would need a scaling factor of 3/2."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "xbasis = d3.RealFourier(coords['x'], size=32, bounds=(0,1), dealias=3/2)\n",
    "ybasis = d3.RealFourier(coords['y'], size=32, bounds=(0,1), dealias=3/2)\n",
    "zbasis = d3.Chebyshev(coords['z'], size=32, bounds=(0,1), dealias=3/2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Some bases have other arguments that can be used to modify their exact internal behavior.\n",
    "See the [basis.py API documentation](../autoapi/dedalus/core/basis/index.html) for more information."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Basis grids and scale factors"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Each basis has a corresponding coordinate/collocation grid (or grids for multidimensional bases) that can be used for tasks like initializing and plotting fields.\n",
    "The global (non-distributed) grids of a basis can be accessed using the basis object's `global_grid` method (or `global_grids` for multidimensional bases).\n",
    "These methods optionally take scale factors determining the number of points in the grid relative to the number of basis modes.\n",
    "Let's look at the Chebyshev grids with scaling factors of 1 and 3/2."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABJwAAAEaCAYAAABU/02+AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAB7CAAAewgFu0HU+AABVfElEQVR4nO3dd3xU1br/8W9Ip4QihJZQlSqI0kRAQm+ClAsIqAF7PVc9ih4LIBw96kVFvEc9ehUOWLALAqJ0BESKiiAdBUIIEGLohLT9+yO/2WcmdSbZM7Nn8nm/XnkxYdZez7NmZs/MfrL22iGGYRgCAAAAAAAALFLB3wkAAAAAAAAguFBwAgAAAAAAgKUoOAEAAAAAAMBSFJwAAAAAAABgKQpOAAAAAAAAsBQFJwAAAAAAAFiKghMAAAAAAAAsRcEJAAAAAAAAlqLgBAAAAAAAAEtRcAIAAAAAAIClKDgBAAAAAADAUhScAAAAAAAAYCkKTgAAAAAAALAUBScAAAAAAABYioITAAAAAAAALEXBCQAAAAAAAJai4AQAAAAAAABLUXACAAAAAACApSg4AQAAAAAAwFIUnAAEpISEBIWEhCgkJESrV6/2dzqFOnjwoJljo0aN/J2OVwXC84E83niupk6davY5depUS/r0FW/lPmfOHLPfCRMmWNYv8vD4orRWr15tvnYSEhL8nQ4ABLUwfycAoHw5ffq0vvnmGy1btkxbt25VamqqTp48qYiICFWvXl3NmzdXx44dNWTIEHXp0sXf6QIA4Hd//PGHtmzZos2bN2vz5s3aunWrzp49a95vGIYfsyvfsrKy9Ouvv5rPz86dO83vNufPn1dMTIwaNWqkTp06aezYserevXup4vz555+KjY1VTk6OBg8erEWLFrncv2vXLi1btkzr1q3Tjh07lJycrAsXLqhq1apq1KiRunbtqgkTJujqq6+2YtgA4BYKTgB84sKFC5o5c6ZmzJih9PT0AvdnZmbq3LlzSkpK0vLly/WPf/xDzZo109SpU3XTTTcpJCTED1kDQN6MiJ49e0qSevTowSw++ExKSoratGmjtLQ0f6eCIixYsECjRo0q8v60tDSlpaVp69atevPNN9W7d2+99957atCggUdxlixZopycHEnSkCFDzP9ftmyZHn74Yf32228lxp81a5ZGjRqlt956SzVq1PAoPgCUBgUnAF53+PBhDRkyRL/++qvL/zdo0EBt27ZVrVq1lJOTo2PHjmnbtm06fvy4JGnv3r0aN26ckpKSNGnSJH+kDgCA31y6dIliUwCJiopS69at1bhxY1WvXl2GYSglJUU//vijTp48KUlasWKFunXrpvXr1ys+Pt7tvhcuXChJCgkJ0Q033GD+/9atW12KTSEhIWrbtq2aNWum6tWrKzU1VevWrVNqaqok6dNPP9XOnTu1Zs0aXXbZZVYMGwCKRMEJgFcdPHhQXbp00bFjxyTlfREaO3asnnzySbVu3bpAe8MwtGXLFr3++uv64IMPlJubqwsXLvg6bSBoMTvH1dSpUwNu3SmUP5UrV9Y111yjjh07qlOnTsrIyFBiYqK/04KkunXr6plnntGgQYPUvn17hYeHF2iTnZ2t2bNn6y9/+YsyMjKUlJSk++67T19//bVbMbKysvTtt99Kkq655hrVr1+/QJt27drpzjvv1JgxYwoUkjIzM/XKK6/o6aefVk5Ojn777Tfdd999+vjjj0sxYgBwHwUnAF6TmZmpUaNGmcWmqKgoffTRRxo2bFiR24SEhKhjx46aO3euJk2apLFjx/ooWwAA7KV27drasWOHWrZsqQoV/nOtHwrH9tG1a1d17dq12DZhYWG68847FRYWpttuu02StHjxYh05ckRxcXElxli9erXOnDkjyfV0Oklq1qyZvvzyy2K/W0VEROiJJ55QZGSkHnnkEUnSJ598omeffVYtWrQoMT4AlBZXqQPgNS+99JK2bNli/v7vf/+72C9E+V155ZXauHGj+vbt64XsAACwt+joaLVu3dql2ITANX78eEVFRUnKm9H9yy+/uLWd80yooUOHutw3YsQIt79b/eUvf1G9evXM35csWeLWdgBQWnx6AfCKixcvatasWebvI0aM0OjRoz3up1KlSiX+5dDhzz//1IsvvqiOHTuqZs2aio6OVpMmTXT77bdrx44dHsU1DENffvmlEhMT1axZM1WtWlVRUVGKj4/XsGHD9O9//1vZ2dkej0eSVq1apXHjxqlp06aKjo5WzZo11a1bN82aNUsZGRlFbjd06FDzUs4vvPCC2/GefPJJc7vx48cX2mb37t2aNGmSrr32WtWsWVMRERGqWrWqLr/8cl177bW677779NVXX5l/YXWHXZ6PrKws1axZ03wMfvjhB7fj9urVy9zu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5XQ6fxoyZIjWr1+vI0eO6P3339fdd9+tVq1amfcbhqEvvvhC7du31969e122dX7vKetp0e+8845ZALr66qv166+/atKkSbruuutUt25dRUdHuxQfSjuryZnzfteuXbtSvYdMnTq1QL8hISFKTEzUtm3btH//fr377rtKTExUkyZNzDbZ2dl699131alTJ5cZOIHGH8+bp+zwfuUpx2e3JK1fv77QNoZhaNGiRZLyCn2tW7f2KEZycrJ69+5tzvRr2rSpli1b5nKKIgD4AgUnAF7hvO7NoUOHCiy+ajfOp1Xt27fP8v7T09Pd+jJ+5swZl7/MF3e1nTvvvNO87bx4uPPpdMXNbsovPDxc3bp105NPPqklS5bo5MmTWrp0qXr06GG2ef3118t8BS13ePP5GD58uHmQsmnTJpf+33//ffO2FbObJNfn0N11jpKTky2JXRrOpwI51rIqSf61bqxQ2NXqHKcvdevWzeWUkMLaUnD6j/r162v8+PF666239NtvvykpKUnTp083izJpaWl65JFHXLZx3geTkpKKncFZEse6R5L09NNPlzjDorAZV55yzv/3338v9Uyp4jRt2lS33Xab5syZowMHDmjv3r169NFHzdMPDxw44LJQc6Dxx/PmKbu8X3nC+QIPjtlH+f3000/m54Cnp9OdOHFCvXv31h9//CEp71Tu5cuX+/W0YgDlFwUnAF4xatQolwPt4i59bAfOC+J+++23lvdvGIZbM71+/PFHc2HPatWqFVi/ydnQoUPNK/wsW7ZMhw8f1r59+/T9999LypuhMHr06FLnHB4erv79++u7775zWdzVeZq/t3jz+ahYsaKGDRtm/u6Y5XT69GktXrxYUt4shnHjxlkSr127duZtxyWpi3Px4kXt2LHDktil4Zzvxo0bS2zv7mvbU/lPq9uzZ49SUlIK3Cflzb5wrAezevVql7WemjVrprp165YpF3+cEudNcXFxevrpp11Ok/3uu+9c1ulp166doqKiJOXNHCnLc+y8nlJJMzVycnKKnPXhiXbt2ikyMlJSXiHf+UIE3nLFFVfof/7nfzRt2jTz/xYuXOj1uN7ij+fNU87vV7t373brVEp33te8yfE+JsnlVHZnzq8bT06nS0tLU58+fbRnzx5JeaeRL1++PKgXtQdgbxScAHhFdHS0y+lXn3/+uT7//HOP+zl//rxP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OlqnT5/WgQMHtH37dmVkZEhSobPoLl68qHfeeUfvvPOOqlSponbt2qlBgwaqVKmSTp48qd27d2vnzp1m+1q1ahV6pTtPTZ48ucBaUPln6TivZeQwbdo0jxecdubP580TtWrV0jvvvKNRo0YpJydHP//8s1q0aKEePXqoSZMmOnPmjFatWqXU1FSFh4fr73//uyZNmmRZ/PT0dE2fPl3Tp09XvXr1zM/RihUr6ty5c9q/f79++uknl+JvnTp1tHDhwkL/uOAoOIWFhWngwIElxn/mmWe0bNky8/eWLVvqk08+0SeffFLitldccYX++7//251hAkDpGADgA2fPnjWmTZtmVKtWzZDk1s9VV11lfPHFF4X216NHD7PdqlWrSow/ZcoUs/2UKVOKbXvw4EGjd+/ebudZu3ZtY+nSpQX6+eOPP8w2DRs2NDIzM40777yz2L4uv/xyY/v27e48pKZLly4ZtWrVMvuIjIw00tLSStzu008/NUJCQtwaY4UKFYz77rvPyMzMLLSvQHg+CtOyZcsC2xf1miurixcvGiNHjiw29y5duhjJycnGuHHjSswnMTHRbDN79my383D3ucrNzTX++te/FvsaueKKK4zffvvNo+fTEydOnCgQf/PmzUW2Hzp0qEvbu+++u8QY7uZ++PBho169ekU+Fg0bNnRpP3v2bPO+xMTEEvPI/35RFldeeaXb+0vjxo2N9evXF9vfhQsXjDvvvNMIDQ0tsb+oqCjjzJkzBfrYs2eP0aRJk2K37dq1q3HkyBG3nhNPHt/s7GzjmWeeMSpWrOjWYxIeHm7cf//9BfqpXLmy24/rVVddZezatavYvNzlvK978uPJ+0JRrH7ePH2vWLVqldm+R48exbb98MMPi32OqlSpYnz++ece9VmS06dPG2FhYR49LyNGjDCOHDlSaH/nzp0zIiMjPcqttK8PK8YPACVhhhMAn6hcubKeeeYZ/eUvf9HixYu1bNkybd26Vampqfrzzz8VERGhGjVqqEWLFurcubOGDRvmsv6TLzVs2FDLly/XDz/8oE8//VRr165VUlKS0tPTFRYWpssuu0xXXHGFOnTooH79+ikhIaHQU3zyCw8P19tvv61Ro0bp3Xff1aZNm5SSkqKKFSuqRYsWGjVqlO655x6XBYrdERERoSFDhpgLhg8fPrzIK984+6//+i/zL//r16/X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",
      "text/plain": [
       "<Figure size 600x150 with 1 Axes>"
      ]
     },
     "metadata": {
      "image/png": {
       "height": 141,
       "width": 590
      }
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "grid_normal = zbasis.global_grid(dist, scale=1).ravel()\n",
    "grid_dealias = zbasis.global_grid(dist, scale=3/2).ravel()\n",
    "\n",
    "plt.figure(figsize=(6, 1.5), dpi=100)\n",
    "plt.plot(grid_normal, 0*grid_normal+1, 'o', markersize=5)\n",
    "plt.plot(grid_dealias, 0*grid_dealias-1, 'o', markersize=5)\n",
    "plt.xlabel('z')\n",
    "plt.title('Chebyshev grid with scales 1 and 3/2')\n",
    "plt.ylim([-2, 2])\n",
    "plt.gca().yaxis.set_ticks([]);\n",
    "plt.tight_layout()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Note that the Chebyshev grids are non-equispaced: the grid points cluster quadratically towards the ends of the interval, which can be very useful for resolving sharp features like boundary layers."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Distributed grid and element arrays"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To help with creating field data, the distributor provides the local portions of the coordinate grids and mode numbers (wavenumbers or polynomial degrees).\n",
    "The local grids (distributed according to the last layout, or full \"grid space\") are accessed using the `dist.local_grid` (or `dist.local_grids` for multidimensional bases) method and specifying the basis and scale factors (1 by default):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Local x shape: (32, 1, 1)\n",
      "Local y shape: (1, 8, 1)\n",
      "Local z shape: (1, 1, 16)\n"
     ]
    }
   ],
   "source": [
    "local_x = dist.local_grid(xbasis)\n",
    "local_y = dist.local_grid(ybasis)\n",
    "local_z = dist.local_grid(zbasis)\n",
    "print('Local x shape:', local_x.shape)\n",
    "print('Local y shape:', local_y.shape)\n",
    "print('Local z shape:', local_z.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The local x grid is the full Fourier grid for the x-basis, and will be the same on all processes, since the first axis is local in grid space.\n",
    "The local y and local z grids will generally differ across processes, since they contain just the local portions of the y and z basis grids, distributed across the specified process mesh (4 blocks in y and 2 blocks in z).\n",
    "\n",
    "The local modes (distributed according to layout 0, or full \"coeff space\") are accessed using the `dist.local_modes` method and specifying the basis:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Local kx shape: (8, 1, 1)\n",
      "Local ky shape: (1, 16, 1)\n",
      "Local nz shape: (1, 1, 32)\n"
     ]
    }
   ],
   "source": [
    "local_kx = dist.local_modes(xbasis)\n",
    "local_ky = dist.local_modes(ybasis)\n",
    "local_nz = dist.local_modes(zbasis)\n",
    "print('Local kx shape:', local_kx.shape)\n",
    "print('Local ky shape:', local_ky.shape)\n",
    "print('Local nz shape:', local_nz.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The local kx and local ky elements will now differ across processes, since they contain just the local portions of the x and y wavenumbers, which are distributed in coefficient space.\n",
    "The local nz elements are the full set of Chebyshev modes, which are always local in coefficient space.\n",
    "\n",
    "These local arrays can be used to form parallel-safe initial conditions for fields, in grid or coefficient space, as we'll see in the next notebook."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "d3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
