"""
Classes for representing systems of equations.

"""

import numpy as np
import sympy as sy

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class ParsedProblem:
    """
    PDE definitions using string representations.

    Equations are assumed to take the form ('LHS = RHS'), where the left-hand
    side contains terms that are linear in the dependent variables (and will be
    represented in coefficient matrices), and the right-hand side contains terms
    that are non-linear (and will be represented by operator trees).  For
    simplicity, the last axis is referred to as the "z" direction.

    Parameters
    ----------
    axis_names : list of strs
        Names of coordinate axes
    field_names : list of strs
        Names of required field variables
    param_names : list of strs, optional
        Names of other parameters

    Attributes
    ----------
    parameters : dict
        Parameters used to construct equations and boundary conditions

    Notes
    -----
    The linear terms are separated into several matrices depending on the
    presence of temporal derivatives and spatial derivatives along the z axis,
    i.e. the last/pencil/implict axis.  Other axes must be represented by
    TransverseBasis objects, and hence their differential operators reduce to
    multiplicative constants for each pencil.

    When adding equations and boundary conditions, the provided axis, field,
    and parameter names will be recognized.  Derivative operators of the form
    'd_' are recognized, where '_' is 't' or an axis name for temporal and
    spatial derivatives, respectively.

    The parser currently cannot expand temporal and z derivatives, i.e. such
    terms must be written as some function of 'dt(f)', 'dt(dz(f))', or 'dz(f)',
    where 'f' is one of the provided field names.

    Currently, nonconstant coefficients must be written as functions of 'z', or
    be arrays of the shape of zbasis.grid, i.e. defined on the global z grid on
    each process.

    """

    def __init__(self, axis_names, field_names, param_names=[]):

        # Create symbols
        axis_syms = [sy.Symbol(an) for an in axis_names]
        field_syms = [sy.Symbol(fn) for fn in field_names]
        param_syms = [sy.Symbol(pn) for pn in param_names]

        # Create differentiation symbols and functions
        diff_names = ['d'+an for an in axis_names]
        trans_names = diff_names[:-1]
        trans_syms = [sy.Symbol(tn) for tn in trans_names]
        trans_ops = [(lambda ts: (lambda A: ts*A))(ts) for ts in trans_syms]
        dz = sy.Function(diff_names[-1])
        dt = sy.Function('dt')

        # Check for name conflicts
        names = axis_names + field_names + param_names + trans_names + [str(dz), str(dt)]
        for name in names:
            if names.count(name) > 1:
                raise ValueError("Name conflict detected: multiple uses of '%s'" %name)

        # Attributes
        self.dim = len(axis_names)
        self.nfields = len(field_names)
        self.parameters = dict()
        self.equations = []
        self.boundary_conditions = []

        # References
        self.axis_names = axis_names
        self.axis_syms = axis_syms
        self.axis_dict = dict(zip(axis_names, axis_syms))
        self.field_names = field_names
        self.field_syms = field_syms
        self.field_dict = dict(zip(field_names, field_syms))
        self.param_names = param_names
        self.param_syms = param_syms
        self.param_dict = dict(zip(param_names, param_syms))
        self.diff_names = diff_names
        self.trans_names = trans_names
        self.trans_syms = trans_syms
        self.trans_ops = trans_ops
        self.trans_dict = dict(zip(trans_names, trans_ops))
        self.dz = dz
        self.dt = dt

    def add_equation(self, eqn_str, condition="True"):
        """Add equation to problem."""

        # Parse equation string into LHS rows and RHS string
        LHS_str, RHS_str = eqn_str.split("=")
        LHS_rows = self._build_rows(LHS_str)

        # Build equation dictionary
        eqn = dict()
        eqn['LHS'] = LHS_rows
        eqn['RHS'] = RHS_str
        eqn['condition'] = condition

        self.equations.append(eqn)

    def add_bc(self, bc_str, functional, condition="True"):
        """Add boundary condition to problem."""

        # Check functional
        if functional not in ['left', 'right', 'int']:
            raise ValueError("Invalid functional: %s" %str(functional))

        # Parse bc string into LHS rows and RHS string
        LHS_str, RHS_str = bc_str.split("=")
        LHS_rows = self._build_rows(LHS_str)

        # Build boundary condition dictionary
        bc = dict()
        bc['LHS'] = LHS_rows
        bc['RHS'] = RHS_str
        bc['functional'] = functional
        bc['condition'] = condition

        self.boundary_conditions.append(bc)

    def add_left_bc(self, bc_str, condition="True"):
        """Add left boundary condition to problem."""

        self.add_bc(bc_str, 'left', condition=condition)

    def add_right_bc(self, bc_str, condition="True"):
        """Add right boundary condition to problem."""

        self.add_bc(bc_str, 'right', condition=condition)

    def add_int_bc(self, bc_str, condition="True"):
        """Add integral boundary condition to problem."""

        self.add_bc(bc_str, 'int', condition=condition)

    def _build_rows(self, exp_str):
        """Parse expression string into rows."""

        # Evaluate string to expression
        dz = self.dz
        dt = self.dt
        namespace = {str(dz):dz, str(dt):dt}
        namespace.update(self.axis_dict)
        namespace.update(self.field_dict)
        namespace.update(self.param_dict)
        namespace.update(self.trans_dict)
        exp = eval(exp_str, namespace).simplify().expand()

        # Allocate rows
        M0 = sy.zeros(1, self.nfields)
        M1 = sy.zeros(1, self.nfields)
        L0 = sy.zeros(1, self.nfields)
        L1 = sy.zeros(1, self.nfields)

        # Extract coefficients from expression
        for i, f in enumerate(self.field_syms):
            M0[i] = exp.coeff(dt(f))
            M1[i] = exp.coeff(dt(dz(f)))
            L0[i] = exp.coeff(f)
            L1[i] = exp.coeff(dz(f))
            exp -= M0[i] * dt(f)
            exp -= M1[i] * dt(dz(f))
            exp -= L0[i] * f
            exp -= L1[i] * dz(f)
            exp = exp.simplify().expand()

        # Make sure all terms were extracted
        if exp:
            raise ValueError("Cannot parse LHS terms: %s" %str(exp))

        # Check for nonlinear terms and transverse NCCs
        for row in [M0, M1, L0, L1]:
            for term in row:
                if term.has(*self.field_syms):
                    raise ValueError("Cannot parse nonlinear LHS coefficient: %s" %str(term))
                if term.has(*self.axis_syms[:-1]):
                    raise ValueError("Cannot parse transverse NCC: %s" %str(term))

        return (M0, M1, L0, L1)

    def expand(self, domain, order=1):
        """
        Expand equations and BCs into z-basis coefficient matrices.

        Parameters
        ----------
        domain : domain object
            Problem domain
        order : int, optional
            Number of terms to retain in the spectral expansion of LHS
            nonconstant coefficients (default: 1)

        """

        # Check dimension compatibility
        if domain.dim != self.dim:
            raise ValueError("Dimension mismatch between problem and domain")

        # References
        eqns = self.equations
        bcs = self.boundary_conditions
        zbasis = domain.bases[-1]

        # Add attributes
        self.neqns = len(eqns)
        self.nbc = len(bcs)
        self.order = order

        # Separate constant and nonconstant parameters
        c_params = dict()
        nc_params = dict()
        for (pn, v) in self.parameters.items():
            if np.isscalar(v):
                # Use symbol as key for sympy subs
                psym = self.param_dict[pn]
                c_params[psym] = v
            elif isinstance(v, np.ndarray):
                if v.shape != zbasis.grid.shape:
                    raise ValueError("Nonconstant coefficients must have same shape as last basis grid: %s" %pn)
                # Use name as key for python eval
                nc_params[pn] = v
            else:
                logger.warning("Non-scalar and non-array parameters not currently implemented for LHS.")

        # Add z grid to nonconstant parameters
        nc_params[self.axis_names[-1]] = zbasis.grid

        # Merge and expand equations, boundary conditions
        self.eqn_set = self._expand_expressions(eqns, c_params, nc_params, zbasis, order)
        self.bc_set = self._expand_expressions(bcs, c_params, nc_params, zbasis, order)

    def _expand_expressions(self, expressions, c_params, nc_params, zbasis, order):
        """Merge and expand symbolic expressions."""

        # Collect rows from each expression
        if expressions:
            M0, M1, L0, L1 = zip(*(exp['LHS'] for exp in expressions))
        else:
            M0, M1, L0, L1 = [], [], [], []

        # Cast as matrices and substitute constant parameters
        M0 = sy.Matrix(M0).subs(c_params)
        M1 = sy.Matrix(M1).subs(c_params)
        L0 = sy.Matrix(L0).subs(c_params)
        L1 = sy.Matrix(L1).subs(c_params)

        # Expand nonconstant coefficients
        M0 = self._expand_matrix(M0, nc_params, zbasis, order)
        M1 = self._expand_matrix(M1, nc_params, zbasis, order)
        L0 = self._expand_matrix(L0, nc_params, zbasis, order)
        L1 = self._expand_matrix(L1, nc_params, zbasis, order)

        # Lambdify for fast evaluation
        # (Should only depend on trans diff consts)
        nsubs = len(zbasis.subbases)
        exp_set = dict()
        exp_set['M0'] = [[sy.lambdify(self.trans_syms, Cp) for Cp in M0[isub]] for isub in range(nsubs)]
        exp_set['M1'] = [[sy.lambdify(self.trans_syms, Cp) for Cp in M1[isub]] for isub in range(nsubs)]
        exp_set['L0'] = [[sy.lambdify(self.trans_syms, Cp) for Cp in L0[isub]] for isub in range(nsubs)]
        exp_set['L1'] = [[sy.lambdify(self.trans_syms, Cp) for Cp in L1[isub]] for isub in range(nsubs)]
        exp_set['F'] = [exp['RHS'] for exp in expressions]

        return exp_set

    def _expand_matrix(self, C, nc_params, zbasis, order):
        """Expand symbolic coefficient matrix."""

        # Allocate matrices for expansion
        nsubs = len(zbasis.subbases)
        shape = C.shape
        C_exp = [[sy.zeros(shape) for p in range(order)] for i in range(nsubs)]

        # Expand term by term and substitute
        for i in range(shape[0]):
            for j in range(shape[1]):
                if C[i,j]:
                    Cij_exp = self._expand_entry(C[i,j], nc_params, zbasis, order)
                    for p in range(order):
                        for isub in range(nsubs):
                            C_exp[isub][p][i,j] = Cij_exp[isub][p]

        return C_exp

    def _expand_entry(self, Cij, nc_params, zbasis, order):
        """Expand symbolic matrix entry."""

        # Allocate data to use in transforms
        nsubs = len(zbasis.subbases)
        pdata = np.zeros(zbasis.coeff_embed, dtype=zbasis.coeff_dtype)
        cdata = np.zeros(zbasis.coeff_size, dtype=zbasis.coeff_dtype)
        Cij_exp = [sy.zeros(1, zbasis.subbases[i].coeff_size) for i in range(nsubs)]

        Cij = Cij.expand()
        terms = sy.Add.make_args(Cij)
        for term in terms:
            if term.has(*nc_params):
                # Separate constant and nonconstant components
                # This allows us to retain {dx} terms and still transform NCCs
                const, nonconst = term.as_independent(*nc_params)
                gdata = eval(str(nonconst), nc_params).astype(zbasis.grid_dtype)
                zbasis.forward(gdata, pdata, 0)
                zbasis.unpad_coeff(pdata, cdata, 0)
                for isub in range(nsubs):
                    start = zbasis.coeff_start[isub]
                    end = zbasis.coeff_start[isub+1]
                    Cij_exp[isub] += const*cdata[start:end]
            else:
                for isub in range(nsubs):
                    Cij_exp[isub][0] += term

        return Cij_exp

    def build_matrices(self, trans):
        """Build problem matrices for a set of trans diff consts."""

        # References
        nfields = self.nfields
        neqns = self.neqns
        nbc = self.nbc
        eqn_set = self.eqn_set
        bc_set = self.bc_set
        nsubs = len(eqn_set['M0'])

        # Evaluate matrices using trans diff consts
        eqn_M0 = np.array([[Cp(*trans) for Cp in eqn_set['M0'][isub]] for isub in range(nsubs)])
        eqn_M1 = np.array([[Cp(*trans) for Cp in eqn_set['M1'][isub]] for isub in range(nsubs)])
        eqn_L0 = np.array([[Cp(*trans) for Cp in eqn_set['L0'][isub]] for isub in range(nsubs)])
        eqn_L1 = np.array([[Cp(*trans) for Cp in eqn_set['L1'][isub]] for isub in range(nsubs)])
        bc_M0 = np.array([[Cp(*trans) for Cp in bc_set['M0'][isub]] for isub in range(nsubs)])
        bc_M1 = np.array([[Cp(*trans) for Cp in bc_set['M1'][isub]] for isub in range(nsubs)])
        bc_L0 = np.array([[Cp(*trans) for Cp in bc_set['L0'][isub]] for isub in range(nsubs)])
        bc_L1 = np.array([[Cp(*trans) for Cp in bc_set['L1'][isub]] for isub in range(nsubs)])

        # Check there are as many selected equations as fields
        trans_dict = dict(zip(self.trans_names, trans))
        eqn_select = []
        for j, eqn in enumerate(self.equations):
            if eval(eqn['condition'], trans_dict):
                eqn_select.append(j)
        if len(eqn_select) < nfields:
            raise ValueError("Too few equations for trans = %s" %str(trans))
        elif len(eqn_select) > nfields:
            raise ValueError("Too many equations for trans = %s" %str(trans))

        # Track selected equations in selection matrix
        # Track differential equations
        Se = np.matrix(np.zeros((nfields, neqns), dtype=int))
        diff_eqn = []
        for i, j in enumerate(eqn_select):
            Se[i,j] = 1
            if (eqn_M1[:,:,j].any() or eqn_L1[:,:,j].any()):
                diff_eqn.append(i)

        # Check that there aren't more selected bcs than diff eqns
        bc_select = []
        for j, bc in enumerate(self.boundary_conditions):
            if eval(bc['condition'], trans_dict):
                bc_select.append((j, bc['functional']))
        if len(bc_select) > len(diff_eqn):
            raise ValueError("Too many boundary conditions for trans = %s" %str(trans))

        # Track selected boundary conditions selection matrices
        Sl = np.matrix(np.zeros((nfields, nbc), dtype=int))
        Sr = np.matrix(np.zeros((nfields, nbc), dtype=int))
        Si = np.matrix(np.zeros((nfields, nbc), dtype=int))
        for k, (j, f) in enumerate(bc_select):
            i = diff_eqn[k]
            if f == 'left':
                Sl[i,j] = 1
            elif f == 'right':
                Sr[i,j] = 1
            elif f == 'int':
                Si[i,j] = 1

        return ((eqn_M0, eqn_M1, eqn_L0, eqn_L1),
                (bc_M0, bc_M1, bc_L0, bc_L1),
                (Se, Sl, Sr, Si))


class MatrixProblem:
    """
    PDE definitions using matrices and operator trees.

    Equations are assumed to take the form (LHS = RHS), where the left-hand side
    contains terms that are linear in the dependent variables (and will be
    represented in coefficient matrices), and the right-hand side contains terms
    that are non-linear (and will be represented by operator trees).  For
    simplicity, the last axis is referred to as the "z" direction.

    Parameters
    ----------
    axis_names : list of strs
        Names of coordinate axes
    field_names : list of strs
        Names of required field variables
    order : int, optional
        Number of terms to retain in the spectral expansion of LHS
        nonconstant coefficients (default: 1)

    Attributes
    ----------
    nfields : int
        Number of fields / equations
    parameters : dict
        Parameters used in constructing RHS operator trees
    eqn_set : dict
        Dictionary with the following keys:
            'M0' : list of functions returning ndarrays
                Coefficient matrix for (d/dt) terms
            'M1' : list of functions returning ndarrays
                Coefficient matrix for (d/dt d/dz) terms
            'L0' : list of functions returning ndarrays
                Coefficient matrix for terms with no derivatives
            'L1' : list of functions returning ndarrays
                Coefficient matrix for (d/dz) terms
            'F' : list of strs
                String representation of RHS operator trees
    bc_set : dict
        Dictionary with the above keys, and:
            'functional' : list of {None, 'left', 'right', 'int'}
                Indication of which functional is to be applied to the
                corresponding row of the provided boundary conditions

    Notes
    -----
    The (i,j)-th element of a coefficient matrix encodes the dependence of the
    i-th equation/constraint on the j-th field named in the `field_names` list.

    The linear terms are separated into several matrices depending on the
    presence of temporal derivatives and spatial derivatives along the z axis,
    i.e. the last/pencil/implict axis.  Other axes must be represented by
    TransverseBasis objects, and hence their differential operators reduce to
    multiplicative constants for each pencil.

    The entries defining the problem matrices are lists of functions, one for
    each term in the spectral expansion of the nonconstant coefficients, that
    take as an argument the list of multiplicative constants for transverse
    differentiation for a given pencil, and return the resulting problem
    coefficients.

    """

    def __init__(self, axis_names, field_names, order=1):

        # Attributes
        self.axis_names = axis_names
        self.diff_names = ['d'+an for an in axis_names]
        self.field_names = field_names
        self.nfields = problem.neqns = problen.nbcs = len(field_names)
        self.order = order
        self.parameters = dict()

        # Check for name conflicts
        names = axis_names + diff_names + field_names
        for name in names:
            if names.count(name) > 1:
                raise ValueError("Name conflict detected: multiple uses of '%s'" %name)

        # Default problem matrices
        size = self.nfields
        self.eqn_set = dict()
        self.eqn_set['M0'] = [lambda trans: np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.eqn_set['M1'] = [lambda trans: np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.eqn_set['L0'] = [lambda trans: np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.eqn_set['L1'] = [lambda trans: np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.eqn_set['F'] = ['0'] * self.nfields
        self.bc_set = dict()
        self.bc_set['M0'] = [lambda trans: np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.bc_set['M1'] = [lambda trans: np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.bc_set['L0'] = [lambda trans: np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.bc_set['L1'] = [lambda trans: np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.bc_set['F'] = ['0'] * self.nfields
        self.bc_set['functional'] = [None] * self.nfields

    def build_matrices(self, trans):
        """Build problem matrices for a set of trans diff consts."""

        # References
        nfields = self.nfields
        eqn_set = self.eqn_set
        bc_set = self.bc_set

        # Evaluate matrices using trans diff consts
        eqn_M0 = np.array([Cp(trans) for Cp in eqn_set['M0']])
        eqn_M1 = np.array([Cp(trans) for Cp in eqn_set['M1']])
        eqn_L0 = np.array([Cp(trans) for Cp in eqn_set['L0']])
        eqn_L1 = np.array([Cp(trans) for Cp in eqn_set['L1']])
        bc_M0 = np.array([Cp(trans) for Cp in bc_set['M0']])
        bc_M1 = np.array([Cp(trans) for Cp in bc_set['M1']])
        bc_L0 = np.array([Cp(trans) for Cp in bc_set['L0']])
        bc_L1 = np.array([Cp(trans) for Cp in bc_set['L1']])

        # Selection matrices
        Se = np.matrix(np.identity(nfields, dtype=int))
        Sl = np.matrix(np.zeros((nfields, nfields), dtype=int))
        Sr = np.matrix(np.zeros((nfields, nfields), dtype=int))
        Si = np.matrix(np.zeros((nfields, nfields), dtype=int))
        for i, f in enumerate(bc_set['functional']):
            if f == 'left':
                Sl[i,i] = 1
            elif f == 'right':
                Sr[i,i] = 1
            elif f == 'int':
                Si[i,i] = 1
            elif f:
                raise ValueError("Invalid functional: %s" %str(functional))

        return ((eqn_M0, eqn_M1, eqn_L0, eqn_L1),
                (bc_M0, bc_M1, bc_L0, bc_L1),
                (Se, Sl, Sr, Si))

