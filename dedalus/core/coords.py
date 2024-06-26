"""Coordinates and coordinate sytems."""

import numpy as np
from ..libraries.dedalus_sphere import jacobi
from ..libraries import dedalus_sphere

from ..tools.array import nkron, sparse_block_diag
from ..tools.cache import CachedMethod, CachedAttribute

# Public interface
__all__ = ['Coordinate',
           'DirectProduct',
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

    def check_bounds(self, coord, bounds):
        pass

    def forward_intertwiner(self, subaxis, order, group):
        raise NotImplementedError("Subclasses must implement.")

    def backward_intertwiner(self, subaxis, order, group):
        raise NotImplementedError("Subclasses must implement.")


class SeparableIntertwiners:

    def forward_vector_intertwiner(self, subaxis, group):
        raise NotImplementedError("Subclasses must implement.")

    def backward_vector_intertwiner(self, subaxis, group):
        raise NotImplementedError("Subclasses must implement.")

    def forward_intertwiner(self, subaxis, order, group):
        vector = self.forward_vector_intertwiner(subaxis, group)
        return nkron(vector, order)

    def backward_intertwiner(self, subaxis, order, group):
        vector = self.backward_vector_intertwiner(subaxis, group)
        return nkron(vector, order)


class Coordinate(SeparableIntertwiners):
    dim = 1
    default_nonconst_groups = (1,)
    curvilinear = False

    def __init__(self, name, cs=None):
        self.name = name
        self.coords = (self,)
        self.cs = cs

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if type(self) is type(other):
            if self.name == other.name:
                return True
        return False

    def __hash__(self):
        return id(self)

    def check_bounds(self, bounds):
        if self.cs == None: return
        else: self.cs.check_bounds(self, bounds)

    def forward_vector_intertwiner(self, subaxis, group):
        return np.array([[1]])

    def backward_vector_intertwiner(self, subaxis, group):
        return np.array([[1]])


class DirectProduct(SeparableIntertwiners, CoordinateSystem):

    def __init__(self, *coordsystems, right_handed=None):
        for cs in coordsystems:
            if not isinstance(cs, SeparableIntertwiners):
                raise NotImplementedError("Direct products only implemented for separable intertwiners.")
        self.coordsystems = coordsystems
        self.coords = sum((cs.coords for cs in coordsystems), ())
        if len(set(self.coords)) < len(self.coords):
            raise ValueError("Cannot repeat coordinates in DirectProduct.")
        self.dim = sum(cs.dim for cs in coordsystems)
        if self.dim == 3:
            if self.curvilinear:
                if right_handed is None:
                    right_handed = False
            else:
                if right_handed is None:
                    right_handed = True
            self.right_handed = right_handed

    @CachedAttribute
    def subaxis_by_cs(self):
        subaxis_dict = {}
        subaxis = 0
        for cs in self.coordsystems:
            subaxis_dict[cs] = subaxis
            subaxis += cs.dim
        return subaxis_dict

    @CachedAttribute
    def curvilinear(self):
        return any(cs.curvilinear for cs in self.coordsystems)

    def forward_vector_intertwiner(self, subaxis, group):
        factors = []
        start_axis = 0
        for cs in self.coordsystems:
            if start_axis <= subaxis < start_axis + cs.dim:
                factors.append(cs.forward_vector_intertwiner(subaxis-start_axis, group))
            else:
                factors.append(np.identity(cs.dim))
            start_axis += cs.dim
        return sparse_block_diag(factors).toarray()

    def backward_vector_intertwiner(self, subaxis, group):
        factors = []
        start_axis = 0
        for cs in self.coordsystems:
            if start_axis <= subaxis < start_axis + cs.dim:
                factors.append(cs.backward_vector_intertwiner(subaxis-start_axis, group))
            else:
                factors.append(np.identity(cs.dim))
            start_axis += cs.dim
        return sparse_block_diag(factors).toarray()

    @CachedAttribute
    def default_nonconst_groups(self):
        return sum((cs.default_nonconst_groups for cs in self.coordsystems), ())


class CartesianCoordinates(SeparableIntertwiners, CoordinateSystem):

    curvilinear = False

    def __init__(self, *names, right_handed=True):
        if len(set(names)) < len(names):
            raise ValueError("Must specify unique names.")
        self.names = names
        self.dim = len(names)
        self.coords = tuple(Coordinate(name, cs=self) for name in names)
        if self.dim == 3:
            self.right_handed = right_handed
        self.default_nonconst_groups = (1,) * self.dim

    def __str__(self):
        return '{' + ','.join([c.name for c in self.coords]) + '}'

    def forward_vector_intertwiner(self, subaxis, group):
        return np.identity(self.dim)

    def backward_vector_intertwiner(self, subaxis, group):
        return np.identity(self.dim)

    @CachedMethod
    def unit_vector_fields(self, dist):
        fields = []
        for i, c in enumerate(self.coords):
            ec = dist.VectorField(self, name=f"e{c.name}")
            ec['g'][i] = 1
            fields.append(ec)
        return tuple(fields)


class AzimuthalCoordinate(Coordinate):
    pass


class CurvilinearCoordinateSystem(CoordinateSystem):

    curvilinear = True


class S2Coordinates(SeparableIntertwiners, CurvilinearCoordinateSystem):
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
        if order > 1:
            U = nkron(U, order)
        return U

    @classmethod
    def _U_backward(cls, order):
        """Unitary transform from spin to coord components."""
        return cls._U_forward(order).T.conj()

    def forward_vector_intertwiner(self, subaxis, group):
        if subaxis == 0:
            # Azimuth intertwiner is identity, independent of group
            return np.identity(self.dim)
        elif subaxis == 1:
            # Colatitude intertwiner is spin-U, independent of group
            return self._U_forward(1)
        else:
            raise ValueError("Invalid axis")

    def backward_vector_intertwiner(self, subaxis, group):
        if subaxis == 0:
            # Azimuth intertwiner is identity, independent of group
            return np.identity(self.dim)
        elif subaxis == 1:
            # Colatitude intertwiner is spin-U, independent of group
            return self._U_backward(1)
        else:
            raise ValueError("Invalid axis")


class PolarCoordinates(SeparableIntertwiners, CurvilinearCoordinateSystem):
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
        if order > 1:
            U = nkron(U, order)
        return U

    @classmethod
    def _U_backward(cls, order):
        """Unitary transform from spin to coord components."""
        return cls._U_forward(order).T.conj()

    def forward_vector_intertwiner(self, subaxis, group):
        if subaxis == 0:
            # Azimuth intertwiner is identity, independent of group
            return np.identity(self.dim)
        elif subaxis == 1:
            # Radial intertwiner is spin-U, independent of group
            return self._U_forward(1)
        else:
            raise ValueError("Invalid axis")

    def backward_vector_intertwiner(self, subaxis, group):
        if subaxis == 0:
            # Azimuth intertwiner is identity, independent of group
            return np.identity(self.dim)
        elif subaxis == 1:
            # Radial intertwiner is spin-U, independent of group
            return self._U_backward(1)
        else:
            raise ValueError("Invalid axis")

    @staticmethod
    def cartesian(phi, r):
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y


class SphericalCoordinates(CurvilinearCoordinateSystem):
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

    @staticmethod
    def cartesian(phi, theta, r):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    def forward_intertwiner(self, subaxis, order, group):
        if subaxis == 0:
            # Azimuth intertwiner is identity, independent of group
            return np.identity(self.dim**order)
        elif subaxis == 1:
            # Colatitude intertwiner is spin-U, independent of group
            return self._U_forward(order)
        elif subaxis == 2:
            # Radius intertwiner is reg-Q, dependent on ell
            ell = group[subaxis-1]
            return self._Q_forward(ell, order)
        else:
            raise ValueError("Invalid axis")

    def backward_intertwiner(self, subaxis, order, group):
        if subaxis == 0:
            # Azimuth intertwiner is identity, independent of group
            return np.identity(self.dim**order)
        elif subaxis == 1:
            # Colatitude intertwiner is spin-U, independent of group
            return self._U_backward(order)
        elif subaxis == 2:
            # Radius intertwiner is reg-Q, dependent on ell
            ell = group[subaxis-1]
            return self._Q_backward(ell, order)
        else:
            raise ValueError("Invalid axis")
