"""
Classes for manipulating pencils.

"""

from functools import partial, reduce
from itertools import product
from collections import defaultdict
import numpy as np
from scipy import sparse
from mpi4py import MPI
import uuid
from math import prod

from .domain import Domain
from ..tools.array import zeros_with_pattern, expand_pattern, sparse_block_diag, copyto, perm_matrix, drop_empty_rows, apply_sparse, assert_sparse_pinv
from ..tools.cache import CachedAttribute, CachedMethod
from ..tools.general import replace, OrderedSet
from ..tools.progress import log_progress

import logging
logger = logging.getLogger(__name__.split('.')[-1])


# for axis in range(domain.dim):
#     if separable[axis]:
#         if len(domain.bases[axis]) > 1:
#             raise ValueError("Cannot have multiple bases along separable dimension")
#         else:
#             basis = domain.bases[axis][0]
#             trans_shape.append(basis.)
#     else:

def build_subsystems(solver):
    """Build local subsystem objects."""
    # Collect local groupsets for each variable and equation
    matrix_coupling = tuple(solver.matrix_coupling)
    coeff_layout = solver.dist.coeff_layout
    all_local_groupsets = []
    for var in solver.problem.variables:
        all_local_groupsets.append(coeff_layout.local_groupsets(matrix_coupling, var.domain, scales=1))
    for eqn in solver.problem.equations:
        all_local_groupsets.append(coeff_layout.local_groupsets(matrix_coupling, eqn['domain'], scales=1))
    # Combine and check that groupsets are nested
    # This prevents incompatible distributions of e.g. different sized transverse bases
    local_groupsets = OrderedSet()
    for i, lgs1 in enumerate(all_local_groupsets):
        for lgs2 in all_local_groupsets[i+1:]:
            if not ((set(lgs1) <= set(lgs2)) or (set(lgs1) >= set(lgs2))):
                raise ValueError("Incompatible group distributions. Are distributed dimensions the same size?")
        local_groupsets.update(lgs1)
    # Build subsystem for each local groupset
    return tuple(Subsystem(solver, groupset) for groupset in local_groupsets)

def build_subproblems(solver, subsystems, build_matrices=None):
    """Build subproblem matrices with progress logger."""
    # Arrange subsysytems by matrix groups
    subsystems_by_group = defaultdict(list)
    for subsystem in subsystems:
        subsystems_by_group[subsystem.matrix_group].append(subsystem)
    # Build subproblem objects for each matrix group
    subproblems = []
    for matrix_group, subsystems in subsystems_by_group.items():
        subproblem = Subproblem(solver, subsystems, matrix_group)
        subproblems.append(subproblem)
    subproblems = tuple(subproblems)
    # Build matrices if requested
    if build_matrices is not None:
        build_subproblem_matrices(solver, subproblems, build_matrices)
    return subproblems

def build_subproblem_matrices(solver, subproblems, matrices):
    # Setup NCCs
    for eq in solver.problem.eqs:
        for matrix in matrices:
            expr = eq[matrix]
            if expr:
                expr.gather_ncc_coeffs()
    # Build matrices
    for subproblem in log_progress(subproblems, logger, 'info', desc='Building subproblem matrices', iter=np.inf, frac=0.1, dt=10):
         subproblem.build_matrices(matrices)

# def rebuild_subproblem_matrices(problem, subsystems, matrices):
    # """Rebuild subproblem matrices with progress logger."""
    # # Setup NCCs
    # for eq in problem.eqs:
    #     for matrix in matrices:
    #         expr = eq[matrix]
    #         if expr:
    #             expr.gather_ncc_coeffs()
    #             # separability = ~problem.matrix_coupling
    #             # expr.build_ncc_matrices(separability, problem.variables)
    # # Get matrix groups
    # subproblem_map = defaultdict(list)
    # for subsystem in subsystems:
    #     subproblem_map[subsystem.matrix_group].append(subsystem)
    # # Build subproblems
    # subproblems = []
    # for matrix_group in log_progress(subproblem_map, logger, 'info', desc='Building subproblem matrices', iter=np.inf, frac=0.1, dt=10):
    #     subsystems = tuple(subproblem_map[matrix_group])
    #     subproblem = Subproblem(problem, subsystems, matrix_group)
    #     subproblem.build_matrices(matrices)
    #     subproblems.append(subproblem)
    # return tuple(subproblems)


class Subsystem:
    """
    Class representing a subset of the global coefficient space.
    I.e. the multidimensional generalization of a pencil.

    Each subsystem is described by a "group" tuple containing a
    group index (for each separable axis) or None (for each coupled
    axis).
    """

    def __init__(self, solver, group):
        self.solver = solver
        self.problem = problem = solver.problem
        self.dist = solver.dist
        self.dtype = problem.dtype
        self.group = group
        # Determine matrix group using solver matrix dependence
        # Map non-dependent groups to group 1, since group 0 may have different truncation
        matrix_dependence = solver.matrix_dependence | solver.matrix_coupling
        change_group = (~matrix_dependence) & (np.array(group) != 0)
        matrix_group = np.array(group)
        matrix_group[change_group] = np.array(self.dist.default_nonconst_groups)[change_group]
        self.matrix_group = tuple(matrix_group)

    def coeff_slices(self, domain):
        slices = self.dist.coeff_layout.local_groupset_slices(self.group, domain, scales=1)
        if len(slices) == 0:
            return (slice(0,0),) * self.dist.dim
        elif len(slices) > 1:
            raise ValueError("Subsystem data not contiguous.")
        else:
            return slices[0]

    def coeff_shape(self, domain):
        shape = []
        # Extract shape from slices
        coeff_slices = self.coeff_slices(domain)
        for ax_slice, ax_size in zip(coeff_slices, domain.coeff_shape):
            indices = ax_slice.indices(ax_size)
            shape.append(len(range(*indices)))
        return tuple(shape)

    def coeff_size(self, domain):
        return prod(self.coeff_shape(domain))

    @CachedMethod
    def field_slices(self, field):
        comp_slices = (slice(None),) * len(field.tensorsig)
        coeff_slices = self.coeff_slices(field.domain)
        return comp_slices + coeff_slices

    @CachedMethod
    def field_shape(self, field):
        comp_shape = tuple(cs.dim for cs in field.tensorsig)
        coeff_shape = self.coeff_shape(field.domain)
        return comp_shape + coeff_shape

    @CachedMethod
    def field_size(self, field):
        return prod(self.field_shape(field))

    # @CachedMethod
    # def valid_field_slices(self, field):
    #     # Enumerate component indices
    #     comp_shape = tuple(cs.dim for cs in field.tensorsig)
    #     enum_components = list(enumerate(np.ndindex(*comp_shape)))
    #     # Filter through all bases
    #     for basis in field.domain.bases:
    #         enum_components = basis.valid_components(self.group, field.tensorsig, enum_components)
    #     # Keep remaining components
    #     comp_selection = np.array([i for i, comp in enum_components])
    #     coeff_slices = self.coeff_slices(field.domain)
    #     return (comp_selection,) + coeff_slices

    # @CachedMethod
    # def valid_field_shape(self, field):
    #     slices = self.valid_field_slices(field)
    #     comp_selection = slices[0]
    #     coeff_shape = self.coeff_shape(field.domain)
    #     return (comp_selection.size,) + coeff_shape

    # @CachedMethod
    # def valid_field_size(self, field):
    #     return prod(self.valid_field_shape(field))

    @CachedMethod
    def _gather_scatter_setup(self, fields):
        # Allocate vector
        fsizes = tuple(self.field_size(f) for f in fields)
        fslices = tuple(self.field_slices(f) for f in fields)
        fshapes = tuple(self.field_shape(f) for f in fields)
        data = np.empty(sum(fsizes), dtype=self.dtype)
        # Make views into data
        fviews = []
        i0 = 0
        for fsize, fshape in zip(fsizes, fshapes):
            if fsize:
                i1 = i0 + fsize
                fview = data[i0:i1].reshape(fshape)
                fviews.append(fview)
                i0 = i1
            else:
                fviews.append(None)
        fviews = tuple(fviews)
        return data, fsizes, fviews, fslices

    def gather(self, fields):
        """Gather and concatenate subsystem data in from multiple fields."""
        data, fsizes, fviews, fslices = self._gather_scatter_setup(tuple(fields))
        # Gather from fields
        for fsize, fview, fslice, field in zip(fsizes, fviews, fslices, fields):
            if fsize:
                copyto(fview, field.data[fslice])
        return data

    def scatter(self, data_input, fields):
        """Scatter concatenated subsystem data out to multiple fields."""
        data, fsizes, fviews, fslices = self._gather_scatter_setup(tuple(fields))
        # Copy to preallocated data with views
        # TODO: optimize by making sure the input data is already written to this buffer
        copyto(data, data_input)
        # Scatter to fields
        for fsize, fview, fslice, field in zip(fsizes, fviews, fslices, fields):
            if fsize:
                copyto(field.data[fslice], fview)


class Subproblem:
    """
    Object representing one coupled subsystem of a problem.

    Subproblems are identified by their group multi-index, which identifies
    the corresponding group of each separable dimension of the problem.

    This is the generalization of 'pencils' from a problem with exactly one
    coupled dimension.
    """

    def __init__(self, solver, subsystems, group):
        self.solver = solver
        self.problem = problem = solver.problem
        self.subsystems = subsystems
        self.group = group
        self.dist = problem.dist
        self.domain = problem.variables[0].domain  # HACK
        self.dtype = problem.dtype
        # Cross reference from subsystems
        for subsystem in subsystems:
            subsystem.subproblem = self
        # Build group dictionary
        self.group_dict = {}
        for axis, ax_group in enumerate(group):
            if ax_group is not None:
                ax_coord = self.dist.coords[axis]
                self.group_dict['n'+ax_coord.name] = ax_group
        # Build input and output buffers and views, except for EVPs
        from .problems import EigenvalueProblem
        if not isinstance(problem, EigenvalueProblem):
            self._build_buffers()

    def _build_buffers(self):
        self._input_buffer, self._input_views = self._build_buffer_views(self.problem.LHS_variables)
        if 'F' in self.problem.equations[0]:
            self._output_buffer, self._output_views = self._build_buffer_views([eqn['F'] for eqn in self.problem.equations])

    @CachedAttribute
    def shape(self):
        return (self.pre_left.shape[0], len(self.subsystems))

    @CachedAttribute
    def size(self):
        return self.shape[0] * self.shape[1]

    @CachedAttribute
    def _compressed_buffer(self):
        return np.zeros(self.shape, dtype=self.dtype)

    def coeff_slices(self, domain):
        return self.subsystems[0].coeff_slices(domain)

    def coeff_shape(self, domain):
        return self.subsystems[0].coeff_shape(domain)

    def coeff_size(self, domain):
        return self.subsystems[0].coeff_size(domain)

    def field_slices(self, field):
        return self.subsystems[0].field_slices(field)

    def field_shape(self, field):
        return self.subsystems[0].field_shape(field)

    def field_size(self, field):
        return self.subsystems[0].field_size(field)

    def _build_buffer_views(self, fields):
        # Allocate buffer
        fsizes = tuple(self.field_size(f) for f in fields)
        buffer = np.zeros((sum(fsizes), len(self.subsystems)), dtype=self.dtype)
        # Make views into buffer
        views = []
        i0 = 0
        for fsize, field in zip(fsizes, fields):
            field_views = []
            if fsize:
                fshape = self.field_shape(field)
                i1 = i0 + fsize
                for j, ss in enumerate(self.subsystems):
                    ss_view = buffer[i0:i1, j].reshape(fshape)
                    ss_slices = ss.field_slices(field)
                    field_views.append((ss_view, ss_slices))
                i0 = i1
            views.append(field_views)
        return buffer, views

    @CachedMethod
    def _input_field_views(self, fields):
        views = []
        for field, buffer_data in zip(fields, self._input_views):
            for buffer_view, field_slices in buffer_data:
                field_view = field.data[field_slices]
                views.append((buffer_view, field_view))
        return tuple(views)

    @CachedMethod
    def _output_field_views(self, fields):
        views = []
        for field, buffer_data in zip(fields, self._output_views):
            for buffer_view, field_slices in buffer_data:
                field_view = field.data[field_slices]
                views.append((buffer_view, field_view))
        return tuple(views)

    def gather_inputs(self, fields, out=None):
        """Gather and precondition subproblem data from input-like field list."""
        # Gather from fields
        views = self._input_field_views(tuple(fields))
        for buffer_view, field_view in views:
            np.copyto(buffer_view, field_view)
        # Apply right preconditioner inverse to compress inputs
        if out is None:
            out = self._compressed_buffer
        apply_sparse(self.pre_right_pinv, self._input_buffer, axis=0, out=out)
        return out

    def gather_outputs(self, fields, out=None):
        """Gather and precondition subproblem data from output-like field list."""
        # Gather from fields
        views = self._output_field_views(tuple(fields))
        for buffer_view, field_view in views:
            np.copyto(buffer_view, field_view)
        # Apply left preconditioner to compress outputs
        if out is None:
            out = self._compressed_buffer
        apply_sparse(self.pre_left, self._output_buffer, axis=0, out=out)
        return out

    def scatter_inputs(self, data, fields):
        """Precondition and scatter subproblem data out to input-like field list."""
        # Undo right preconditioner inverse to expand inputs
        apply_sparse(self.pre_right, data, axis=0, out=self._input_buffer)
        # Scatter to fields
        views = self._input_field_views(tuple(fields))
        for buffer_view, field_view in views:
            np.copyto(field_view, buffer_view)

    def scatter_outputs(self, data, fields):
        """Precondition and scatter subproblem data out to output-like field list."""
        # Undo left preconditioner to expand outputs
        apply_sparse(self.pre_left_pinv, data, axis=0, out=self._output_buffer)
        # Scatter to fields
        views = self._output_field_views(tuple(fields))
        for buffer_view, field_view in views:
            np.copyto(field_view, buffer_view)

    def inclusion_matrices(self, bases):
        """List of inclusion matrices."""
        matrices = []
        domain = domain.from_bases(self.domain, bases)
        # if any(bases):
        #     domain = domain.from_bases(bases)
        # else:
        #     domain = domain.from_domain(self.domain)
        global_slices = self.global_slices(domain)
        for gs, basis in zip(global_slices, bases):
            if basis is None:
                matrices.append(np.array([[1]])[gs, gs])
            else:
                matrices.append(basis.inclusion_matrix[gs, gs])
        return matrices

    # @CachedMethod
    # def expansion_matrix(self, input_domain, output_domain):
    #     matrices = []
    #     dtype = self.domain.dtype
    #     layout = self.domain.dist.coeff_layout
    #     arg_shape = layout.global_array_shape(input_domain, scales=1)
    #     out_shape = layout.global_array_shape(output_domain, scales=1)
    #     arg_elements = self.global_slices(input_domain)
    #     out_elements = self.global_slices(output_domain)
    #     for axis, (I, J, i, j) in enumerate(zip(out_shape, arg_shape, out_elements, arg_elements)):
    #         matrix = sparse.eye(I, J, dtype=dtype, format='csr')
    #         matrices.append(matrix[i, j])
    #     return reduce(sparse.kron, matrices, 1)

    # def expansion_matrix(self, arg, layout):
    #     matrices = []
    #     dtype = self.domain.dtype
    #     arg_shape = layout.global_array_shape(arg.domain, arg.scales)
    #     out_shape = layout.global_array_shape(self.domain, self.scales)
    #     arg_elements = layout.local_elements(arg.domain, arg.scales)
    #     out_elements = layout.local_elements(self.domain, self.scales)
    #     for axis, (I, J, i, j) in enumerate(zip(out_shape, arg_shape, out_elements, arg_elements)):
    #         if (arg.bases[axis] is None) and layout.grid_space[axis]:
    #             matrix = sparse.csr_matrix(np.ones((I, J), dtype=dtype))
    #         else:
    #             matrix = sparse.eye(I, J, dtype=dtype, format='csr')
    #         # Avoid bug on (1,1) advanced indexing of sparse matrices
    #         if i.size == 1:
    #             i = [[i[0]]]
    #         else:
    #             i = i[:,None]
    #         if j.size == 1:
    #             j = [[j[0]]]
    #         else:
    #             j = j[None,:]
    #         matrices.append(matrix[i, j])
    #     return reduce(sparse.kron, matrices, 1)

    # def expansion_matrix(self, inbases, outbases):
    #     axmats = self.inclusion_matrices(outbases)
    #     for axis, (inbasis, outbasis) in enumerate(zip(inbases, outbases)):
    #         if (inbasis is None) and (outbasis is not None):
    #             axmats[axis] = axmats[axis][:, 0:1]
    #     return reduce(sparse.kron, axmats, 1).tocsr()

    # def local_to_group(self, domain):
    #     """Matrix restricting local data to group data."""
    #     shape = self.domain.dist.coeff_layout.local_array_shape(domain, scales=1)
    #     slices = self.local_slices(domain)
    #     matrices = []
    #     for axis in range(self.domain.dim):
    #         matrix = sparse.identity(shape[axis], format='csr')[slices[axis], :]
    #         matrices.append(matrix)
    #     return reduce(sparse.kron, matrices, 1)

    # def group_to_modes(self, field):
    #     """Matrix restricting group data to nonzero modes."""
    #     # matrices = []
    #     # for group, basis in zip(self.group, bases):
    #     #     if basis is None:
    #     #         matrices.append(np.array([[1]]))
    #     #     else:
    #     #         matrices.append(basis.mode_map(group))
    #     # return reduce(sparse.kron, matrices, 1).tocsr()
    #     fsize = self.field_size(field)
    #     if fsize:
    #         vfshape = self.valid_field_shape(field)
    #         indices = np.arange(fsize).reshape((-1,) + vfshape[1:])
    #         # Avoid issue when there are no valid slices
    #         if self.valid_field_slices(field)[0].size:
    #             indices = indices[self.valid_field_slices(field)[0]].ravel()
    #         else:
    #             indices = slice(0, 0)
    #     else:
    #         indices = (slice(None), slice(None))
    #     matrix = sparse.identity(fsize, format='csr')[indices]
    #     return matrix.tocsr()

    def valid_modes(self, field, valid_modes):
        sp_slices = self.field_slices(field)
        return valid_modes[sp_slices]

    # def mode_map(self, basis_sets):
    #     """Restrict group data to nonzero modes."""
    #     var_mats = []
    #     for basis_set in basis_sets:
    #         ax_mats = []
    #         for group, basis in zip(self.group, basis_set):
    #             if basis is None:
    #                 ax_mats.append(np.array([[1]]))
    #             else:
    #                 ax_mats.append(basis.mode_map(group))
    #         var_mat = reduce(sparse.kron, ax_mats, 1).tocsr()
    #         var_mats.append(reduce(sparse.kron, ax_mats, 1).tocsr())
    #     return sparse_block_diag(var_mats).tocsr()

    def check_condition(self, eqn):
        return eval(eqn['condition'], self.group_dict)

    def build_matrices(self, names):
        """Build problem matrices."""

        solver = self.solver
        eqns = self.problem.equations
        vars = self.problem.LHS_variables
        eqn_conditions = [self.check_condition(eqn) for eqn in eqns]  # HACK
        eqn_sizes = [self.field_size(eqn['eqn']) for eqn in eqns]
        var_sizes = [self.field_size(var) for var in vars]
        I = sum(eqn_sizes)
        J = sum(var_sizes)
        dtype = self.dtype

        # Construct subsystem matrices
        # Include all equations and group entries
        matrices = {}
        for name in names:
            # Collect entries
            data, rows, cols = [], [], []
            i0 = 0
            for eqn, eqn_size, eqn_cond in zip(eqns, eqn_sizes, eqn_conditions):
                if eqn_size and eqn_cond and (eqn[name] != 0):
                    # Build matrix and append data
                    eqn_blocks = eqn[name].expression_matrices(subproblem=self, vars=vars, ncc_cutoff=solver.ncc_cutoff, max_ncc_terms=solver.max_ncc_terms)
                    j0 = 0
                    for var, var_size in zip(vars, var_sizes):
                        if var_size and (var in eqn_blocks):
                            block = sparse.coo_matrix(eqn_blocks[var])
                            data.append(block.data)
                            rows.append(i0 + block.row)
                            cols.append(j0 + block.col)
                        j0 += var_size
                i0 += eqn_size
            # Build sparse matrix
            if data:
                data = np.concatenate(data)
                rows = np.concatenate(rows)
                cols = np.concatenate(cols)
                # Filter entries
                data[np.abs(data) < solver.entry_cutoff] = 0
            matrices[name] = sparse.coo_matrix((data, (rows, cols)), shape=(I, J), dtype=dtype).tocsr()

        # Valid modes
        valid_eqn = [self.valid_modes(eqn['eqn'], eqn['valid_modes']) for eqn in eqns]
        valid_var = [self.valid_modes(var, var.valid_modes) for var in vars]
        # Invalidate equations that fail condition test
        for n, eqn_cond in enumerate(eqn_conditions):
            if not eqn_cond:
                valid_eqn[n] = False * valid_eqn[n] # Force copy instead of changing in place
        # Convert to filter matrices
        valid_eqn = sparse.diags(np.concatenate([v.ravel() for v in valid_eqn]), dtype=int).tocsr()
        valid_var = sparse.diags(np.concatenate([v.ravel() for v in valid_var]), dtype=int).tocsr()

        # Check squareness of restricted system
        if valid_eqn.nnz != valid_var.nnz:
            raise ValueError("Non-square system: group={}, I={}, J={}".format(self.group, valid_eqn.nnz, valid_var.nnz))

        # Permutations
        left_perm = left_permutation(self, eqns, bc_top=solver.bc_top, interleave_components=solver.interleave_components).tocsr()
        right_perm = right_permutation(self, vars, tau_left=solver.tau_left, interleave_components=solver.interleave_components).tocsr()

        # Preconditioners
        # TODO: remove astype casting, requires dealing with used types in apply_sparse
        self.pre_left = drop_empty_rows(left_perm @ valid_eqn).tocsr().astype(dtype)
        self.pre_left_pinv = self.pre_left.T.tocsr().astype(dtype)
        self.pre_right_pinv = drop_empty_rows(right_perm @ valid_var).tocsr().astype(dtype)
        self.pre_right = self.pre_right_pinv.T.tocsr().astype(dtype)

        # Check preconditioner pseudoinverses
        assert_sparse_pinv(self.pre_left, self.pre_left_pinv)
        assert_sparse_pinv(self.pre_right, self.pre_right_pinv)

        # Precondition matrices
        for name in matrices:
            matrices[name] = self.pre_left @ matrices[name] @ self.pre_right

        # Store minimal CSR matrices for fast dot products
        for name, matrix in matrices.items():
            setattr(self, '{:}_min'.format(name), matrix.tocsr())

        # Store expanded CSR matrices for fast recombination
        if len(matrices) > 1:
            if solver.store_expanded_matrices:
                # Build expanded LHS matrix to store matrix combinations
                self.LHS = zeros_with_pattern(*matrices.values()).tocsr()
                # Store expanded matrices for fast combination
                for name, matrix in matrices.items():
                    expanded = expand_pattern(matrix, self.LHS)
                    setattr(self, '{:}_exp'.format(name), expanded.tocsr())
            else:
                # Placeholder for accessing shape
                self.LHS = sparse.csr_matrix((valid_eqn.nnz, valid_var.nnz), dtype=dtype)

        # Update rank for Woodbury
        eqn_dofs_by_dim = defaultdict(int)
        eqn_pass_cond = [eqn for eqn, cond in zip(eqns, eqn_conditions) if cond]
        for eqn in eqn_pass_cond:
            eqn_dofs_by_dim[eqn['domain'].dim] += self.field_size(eqn['eqn'])
        self.update_rank = sum(eqn_dofs_by_dim.values()) - eqn_dofs_by_dim[max(eqn_dofs_by_dim.keys())]

        # Store RHS conversion matrix
        # F_conv = [self.expansion_matrix(eq['F'].domain, eq['domain']) for eq in eqns]
        # for n, eqn in enumerate(eqns):
        #     if not eval(eqn['raw_condition'], self.group_dict):
        #         F_conv[n] = F_conv[n][0:0, :]
        # self.rhs_map = drop_eqn * sparse_block_diag(F_conv).tocsr()

    def expand_matrices(self, matrices):
        matrices = {matrix: getattr(self, '%s_min' %matrix) for matrix in matrices}
        # Build expanded LHS matrix to store matrix combinations
        self.LHS = zeros_with_pattern(*matrices.values()).tocsr()
         # Store expanded matrices for fast combination
        for name, matrix in matrices.items():
            expanded = expand_pattern(matrix, self.LHS)
            setattr(self, '{:}_exp'.format(name), expanded.tocsr())


def left_permutation(subproblem, equations, bc_top, interleave_components):
    """
    Left permutation acting on equations.
    bc_top determines if lower-dimensional equations are placed at the top or bottom of the matrix.

    Input ordering:
        Equations > Components > Modes
    Output ordering with interleave_components=True:
        Modes > Components > Equations
    Output ordering with interleave_components=False:
        Modes > Equations > Components
    """
    # Compute hierarchy or input equation indices
    i = 0
    L0 = []
    for eqn in equations:
        L1 = []
        vfshape = subproblem.field_shape(eqn['eqn'])
        rank = len(eqn['tensorsig'])
        vfshape = (prod(vfshape[:rank]),) + vfshape[rank:]
        if vfshape[0] == 0:
            L1.append([])
            L0.append(L1)
            continue
        for comp in range(vfshape[0]):
            L2 = []
            for coeff in range(prod(vfshape[1:])):
                L2.append(i)
                i += 1
            L1.append(L2)
        L0.append(L1)
    # Reverse list hierarchy, grouping by dimension
    indices = defaultdict(list)  # dict over dimension
    n1max = len(L0)
    n2max = max(len(L1) for L1 in L0)
    n3max = max(len(L2) for L1 in L0 for L2 in L1)
    if interleave_components:
        for n3 in range(n3max):
            for n2 in range(n2max):
                for n1 in range(n1max):
                    dim = equations[n1]['domain'].dim
                    try:
                        indices[dim].append(L0[n1][n2][n3])
                    except IndexError:
                        continue
    else:
        for n3 in range(n3max):
            for n1 in range(n1max):
                dim = equations[n1]['domain'].dim
                for n2 in range(n2max):
                    try:
                        indices[dim].append(L0[n1][n2][n3])
                    except IndexError:
                        continue
    # Combine indices by dimension
    dims = sorted(list(indices.keys()))
    if bc_top:
        indices = [indices[dim] for dim in dims]
    else:
        indices = [indices[dim] for dim in dims[::-1]]
    indices = sum(indices, [])
    return perm_matrix(indices, source_index=True, sparse=True).tocsr()


def right_permutation(subproblem, variables, tau_left, interleave_components):
    """
    Right permutation acting on variables.
    tau_left determines if lower-dimensional variables are placed at the left or right of the matrix.

    Input ordering:
        Variables > Components > Modes
    Output ordering with interleave_components=True:
        Modes > Components > Variables
    Output ordering with interleave_components=False:
        Modes > Variables > Components
    """
    # Compute hierarchy or input variable indices
    i = 0
    L0 = []
    for var in variables:
        L1 = []
        vfshape = subproblem.field_shape(var)
        rank = len(var.tensorsig)
        vfshape = (prod(vfshape[:rank]),) + vfshape[rank:]
        if vfshape[0] == 0:
            L1.append([])
            L0.append(L1)
            continue
        for comp in range(vfshape[0]):
            L2 = []
            for coeff in range(prod(vfshape[1:])):
                L2.append(i)
                i += 1
            L1.append(L2)
        L0.append(L1)
    # Reverse list hierarchy, grouping by dimension
    indices = defaultdict(list)  # dict over dimension
    L1max = len(L0)
    L2max = max(len(L1) for L1 in L0)
    L3max = max(len(L2) for L1 in L0 for L2 in L1)
    if interleave_components:
        for n3 in range(L3max):
            for n2 in range(L2max):
                for n1 in range(L1max):
                    dim = variables[n1].domain.dim
                    try:
                        indices[dim].append(L0[n1][n2][n3])
                    except IndexError:
                        continue
    else:
        for n3 in range(L3max):
            for n1 in range(L1max):
                dim = variables[n1].domain.dim
                for n2 in range(L2max):
                    try:
                        indices[dim].append(L0[n1][n2][n3])
                    except IndexError:
                        continue
    # Combine indices by dimension
    dims = sorted(list(indices.keys()))
    if tau_left:
        indices = [indices[dim] for dim in dims]
    else:
        indices = [indices[dim] for dim in dims[::-1]]
    indices = sum(indices, [])
    return perm_matrix(indices, source_index=True, sparse=True).tocsr()
