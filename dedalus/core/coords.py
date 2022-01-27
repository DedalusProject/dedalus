"""Coordinates and coordinate sytems."""

import numpy as np
from ..libraries.dedalus_sphere import jacobi
from ..libraries import dedalus_sphere

from ..tools.array import nkron

# Public interface
__all__ = ['Coordinate',
           'CartesianCoordinates',
           'S2Coordinates',
           'PolarCoordinates',
           'SphericalCoordinates']


class CoordinateSystem:

    def __eq__(self, other):
        if type(self) is type(other):
            if len(self.coords) == len(other.coords):
                for i, coord in enumerate(self.coords):
                    if coord != other.coords[i]:
                        return False
                    return True
        return False

    def __hash__(self):
        return id(self)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.coords[self.names.index(key)]
        else:
            return self.coords[key]

    def set_distributor(self, distributor):
        self.dist = distributor
        for coord in self.coords:
            coord.dist = distributor

    def check_bounds(self, coord, bounds):
        pass

    @property
    def first_axis(self):
        return self.dist.coords.index(self.coords[0])

    @property
    def axis(self):
        return self.dist.coords.index(self.coords[0])


class Coordinate:
    dim = 1
    default_nonconst_groups = (1,)

    def __init__(self, name, cs=None):
        self.name = name
        self.coords = (self,)
        self.cs = cs

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if self.name == other.name: return True
        else: return False

    def __hash__(self):
        return id(self)

    @property
    def axis(self):
        return self.dist.coords.index(self)

    def check_bounds(self, bounds):
        if self.cs == None: return
        else: self.cs.check_bounds(self, bounds)

    def set_distributor(self, distributor):
        self.dist = distributor
        if self.cs:
            self.cs.dist = distributor


class CartesianCoordinates(CoordinateSystem):

    def __init__(self, *names, right_handed=True):
        self.names = names
        self.dim = len(names)
        self.coords = tuple(Coordinate(name, cs=self) for name in names)
        self.right_handed = right_handed
        self.default_nonconst_groups = (1,) * self.dim

    def __str__(self):
        return '{' + ','.join([c.name for c in self.coords]) + '}'

    def forward_intertwiner(self, axis, order, group):
        return np.identity(self.dim**order)

    def backward_intertwiner(self, axis, order, group):
        return np.identity(self.dim**order)


class AzimuthalCoordinate(Coordinate):
    pass


class S2Coordinates(CoordinateSystem):
    """
    S2 coordinate system: (azimuth, colatitude)
    Coord component ordering: (azimuth, colatitude)
    Spin component ordering: (-, +)
    """

    spin_ordering = (-1, +1)
    dim = 2
    default_nonconst_groups = (0, 1) # ell=0 has different validity

    def __init__(self, azimuth, colatitude):
        self.names = (azimuth, colatitude)
        self.azimuth = AzimuthalCoordinate(azimuth, cs=self)
        self.colatitude = Coordinate(colatitude, cs=self)
        self.coords = (self.azimuth, self.colatitude)

    @classmethod
    def _U_forward(cls, order):
        """Unitary transfrom from coord to spin components."""
        # u[+-] = (u[θ] +- 1j*u[φ]) / sqrt(2)
        Ui = {+1: np.array([+1j, 1]) / np.sqrt(2),
              -1: np.array([-1j, 1]) / np.sqrt(2)}
        U = np.array([Ui[spin] for spin in cls.spin_ordering])
        return nkron(U, order)

    @classmethod
    def _U_backward(cls, order):
        """Unitary transform from spin to coord components."""
        return cls._U_forward(order).T.conj()

    @property
    def axis(self):
        return self.azimuth.axis

    def forward_intertwiner(self, axis, order, group):
        subaxis = axis - self.axis
        if subaxis == 0:
            # Azimuth intertwiner is identity, independent of group
            return np.identity(self.dim**order)
        elif subaxis == 1:
            # Colatitude intertwiner is spin-U, independent of group
            return self._U_forward(order)
        else:
            raise ValueError("Invalid axis")

    def backward_intertwiner(self, axis, order, group):
        subaxis = axis - self.axis
        if subaxis == 0:
            # Azimuth intertwiner is identity, independent of group
            return np.identity(self.dim**order)
        elif subaxis == 1:
            # Colatitude intertwiner is spin-U, independent of group
            return self._U_backward(order)
        else:
            raise ValueError("Invalid axis")


class PolarCoordinates(CoordinateSystem):
    """
    Polar coordinate system: (azimuth, radius)
    Coord component ordering: (azimuth, radius)
    Spin component ordering: (-, +)
    """

    spin_ordering = (-1, +1)
    dim = 2
    default_nonconst_groups = (0, 0)

    def __init__(self, azimuth, radius):
        self.names = (azimuth, radius)
        self.azimuth = AzimuthalCoordinate(azimuth, cs=self)
        self.radius = Coordinate(radius, cs=self)
        self.coords = (self.azimuth, self.radius)

    @classmethod
    def _U_forward(cls, order):
        """Unitary transfrom from coord to spin components."""
        # u[+-] = (u[radius] +- 1j*u[φ]) / sqrt(2)
        Ui = {+1: np.array([+1j, 1]) / np.sqrt(2),
              -1: np.array([-1j, 1]) / np.sqrt(2)}
        U = np.array([Ui[spin] for spin in cls.spin_ordering])
        return nkron(U, order)

    @classmethod
    def _U_backward(cls, order):
        """Unitary transform from spin to coord components."""
        return cls._U_forward(order).T.conj()

    @property
    def axis(self):
        return self.azimuth.axis

    def forward_intertwiner(self, axis, order, group):
        subaxis = axis - self.axis
        if subaxis == 0:
            # Azimuth intertwiner is identity, independent of group
            return np.identity(self.dim**order)
        elif subaxis == 1:
            # Radial intertwiner is spin-U, independent of group
            return self._U_forward(order)
        else:
            raise ValueError("Invalid axis")

    def backward_intertwiner(self, axis, order, group):
        subaxis = axis - self.axis
        if subaxis == 0:
            # Azimuth intertwiner is identity, independent of group
            return np.identity(self.dim**order)
        elif subaxis == 1:
            # Radial intertwiner is spin-U, independent of group
            return self._U_backward(order)
        else:
            raise ValueError("Invalid axis")

    @staticmethod
    def cartesian(phi, r):
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y


class SphericalCoordinates(CoordinateSystem):
    """
    Spherical coordinate system: (azimuth, colatitude, radius)
    Coord component ordering: (azimuth, colatitude, radius)
    Spin component ordering: (-, +, 0)
    Regularity component ordering: (-, +, 0)
    """

    spin_ordering = (-1, +1, 0)
    reg_ordering = (-1, +1, 0)
    dim = 3
    right_handed = False
    default_nonconst_groups = (0, 1, 0) # ell=0 has different validity

    def __init__(self, azimuth, colatitude, radius):
        self.names = (azimuth, colatitude, radius)
        self.azimuth = AzimuthalCoordinate(azimuth, cs=self)
        self.colatitude = Coordinate(colatitude, cs=self)
        self.radius = Coordinate(radius, cs=self)
        self.S2coordsys = S2Coordinates(azimuth, colatitude)
        self.coords = (self.azimuth, self.colatitude, self.radius)

    @classmethod
    def _U_forward(cls, order):
        """Unitary transfrom from coord to spin components."""
        # u[+-] = (u[θ] +- 1j*u[φ]) / sqrt(2)
        # u[0] = u[r]
        Ui = {+1: np.array([+1j, 1, 0]) / np.sqrt(2),
              -1: np.array([-1j, 1, 0]) / np.sqrt(2),
               0: np.array([  0, 0, 1])}
        U = np.array([Ui[spin] for spin in cls.spin_ordering])
        return nkron(U, order)

    @classmethod
    def _U_backward(cls, order):
        """Unitary transform from spin to coord components."""
        return cls._U_forward(order).T.conj()

    @classmethod
    def _Q_forward(cls, ell, order):
        """Orthogonal transform from spin to regularity components."""
        return cls._Q_backward(ell, order).T

    @classmethod
    def _Q_backward(cls, ell, order):
        """Orthogonal transform from regularity to spin components."""
        # This may not rebust to having spin and reg orderings be different?
        return dedalus_sphere.spin_operators.Intertwiner(ell, indexing=cls.reg_ordering)(order)

    @property
    def axis(self):
        return self.azimuth.axis

    def check_bounds(self, coord, bounds):
        if coord == self.radius:
            if min(bounds) < 0:
                raise ValueError("bounds for radial coordinate must not be negative")

    def sub_cs(self, other):
        if type(other) is Coordinate:
            if (other == self.radius) or (other == self.colatitude) or (other == self.azimuth):
                return True
            else:
                return False
        elif type(other) is S2Coordinates:
            if other == self.S2coordsys: return True
            else: return False
        return False

    def set_distributor(self, distributor):
        self.dist = distributor
        super().set_distributor(distributor)
        self.S2coordsys.set_distributor(distributor)

    @staticmethod
    def cartesian(phi, theta, r):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    def forward_intertwiner(self, axis, order, group):
        subaxis = axis - self.axis
        if subaxis == 0:
            # Azimuth intertwiner is identity, independent of group
            return np.identity(self.dim**order)
        elif subaxis == 1:
            # Colatitude intertwiner is spin-U, independent of group
            return self._U_forward(order)
        elif subaxis == 2:
            # Radius intertwiner is reg-Q, dependent on ell
            ell = group[axis-1]
            return self._Q_forward(ell, order)
        else:
            raise ValueError("Invalid axis")

    def backward_intertwiner(self, axis, order, group):
        subaxis = axis - self.axis
        if subaxis == 0:
            # Azimuth intertwiner is identity, independent of group
            return np.identity(self.dim**order)
        elif subaxis == 1:
            # Colatitude intertwiner is spin-U, independent of group
            return self._U_backward(order)
        elif subaxis == 2:
            # Radius intertwiner is reg-Q, dependent on ell
            ell = group[axis-1]
            return self._Q_backward(ell, order)
        else:
            raise ValueError("Invalid axis")

