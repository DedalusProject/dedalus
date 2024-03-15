"""
Classes for systems of coefficients/fields.

"""

import numpy as np
from math import prod

from ..tools.cache import CachedMethod
from ..tools.general import unify


class CoeffSystem:
    """
    Representation of a collection of fields that don't need to be transformed,
    and are therefore stored as a contigous set of coefficient data for
    efficient pencil and group manipulation.

    Parameters
    ----------
    nfields : int
        Number of fields to represent
    domain : domain object
        Problem domain

    Attributes
    ----------
    data : ndarray
        Contiguous buffer for field coefficients

    """

    """
    var buffer


    """

    def __init__(self, subproblems, dtype):
        # Build buffer
        total_size = sum(sp.LHS.shape[1]*len(sp.subsystems) for sp in subproblems)
        self.data = np.zeros(total_size, dtype=dtype)
        # Build views
        i0 = i1 = 0
        self.views = views = {}
        for sp in subproblems:
            views[sp] = views_sp = {}
            i00 = i0
            for ss in sp.subsystems:
                i1 += sp.LHS.shape[1]
                views_sp[ss] = self.data[i0:i1]
                i0 = i1
            i11 = i1
            if i11 - i00 > 0:
                views_sp[None] = self.data[i00:i11].reshape((sp.LHS.shape[1], -1))
            else:
                views_sp[None] = self.data[i00:i11].reshape((0, 0))

    def get_subdata(self, sp, ss=None):
        return self.views[sp][ss]


class FieldSystem(CoeffSystem):
    """
    Collection of fields alongside a CoeffSystem buffer for efficient pencil
    and group manipulation.

    Parameters
    ----------
    field_names : list of strings
        Names of fields to build
    domain : domain object
        Problem domain

    Attributes
    ----------
    data : ndarray
        Contiguous buffer for field coefficients
    fields : list
        Field objects
    nfields : int
        Number of fields in system
    field_dict : dict
        Dictionary of fields

    """

    def __init__(self, fields, subproblems):

        domain = unify_attributes(fields, 'domain')
        array_shapes = [coeff_layout.local_array_shape(f.subdomain) for f in fields]
        array_sizes = [prod(shape) for shape in array_shapes]
        buffer_size = sum(array_sizes)
        starts = np.cumsum(array_sizes) - array_sizes
        buffer = np.zeros(buffer_size, dtype=domain.dtype)

        flat_views = [buffer[start:start+size] for start, size in zip(starts, array_sizes)]
        self.array_views = [flat.reshape(shape) for flat, shape in zip(flat_views, array_shapes)]

        blocks = [sp.group_to_modes(var.bases)*sp.local_to_group(var.subdomain) for var in vars for sp in subproblems]
        self.perm = sparse_block_diag(blocks, format='csr')

        self.buffer = buffer
        self.group_buffer = buffer[:self.perm.shape[0]]
        self.field_buffer = buffer[:self.perm.shape[1]]

        # # Build fields
        # fields = [domain.new_field(name=fn) for fn in field_names]
        # nfields = len(fields)

        # # Allocate data for joined coefficients
        # super().__init__(nfields, domain)

        # # Attributes
        # self.domain = domain
        # self.field_names = field_names
        # self.fields = fields
        # self.nfields = nfields
        # self.field_dict = dict(zip(field_names, fields))

    # def __getitem__(self, name):
    #     """Return field corresponding to specified name."""
    #     return self.field_dict[name]

    # @classmethod
    # def from_fields(cls, fields):
    #     names = [field.name for field in fields]
    #     domain = unify(field.domain for field in fields)
    #     sys = FieldSystem(names, domain)
    #     sys.fields = fields
    #     sys.field_dict = dict(zip(names, fields))
    #     return sys

    def gather(self):
        """Copy fields into system buffer."""
        for field, view in zip(self.fields, self.array_views):
            view[:] = field['c']
        self.group_buffer[:] = self.perm * self.field_buffer

        # stride = self.nfields
        # for start, field in enumerate(self.fields):
        #     field.require_coeff_space()
        #     np.copyto(self.data[..., start::stride], field.data)

    def scatter(self):
        """Extract fields from system buffer."""
        self.field_buffer[:] = self.perm.T * self.group_buffer
        for field, view in zip(self.fields, self.array_views):
            field['c'] = view

        # stride = self.nfields
        # coeff_layout = self.domain.dist.coeff_layout
        # for start, field in enumerate(self.fields):
        #     field.preset_layout(coeff_layout)
        #     np.copyto(field.data, self.data[..., start::stride])

