"""
Classes for representing systems of equations.

"""

from collections import defaultdict
from functools import reduce
import operator
import numpy as np
import sympy as sy
from sympy.simplify.simplify import bottom_up

from ..data import operators

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class SymbolicParsingError(Exception):
    pass


class UnsupportedEquationError(Exception):
    pass


class Problem:
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


    default_ops = ['L', 'R']
    op_roots = ['d', 'H']

    def __init__(self, axis_names, field_names, param_names=[]):

        # Attributes
        self.dim = len(axis_names)
        self.nfields = len(field_names)
        self.parameters = dict()
        self.equations = []
        self.boundary_conditions = []
        self.neqns = 0

        eqn_factory = lambda: sy.zeros(len(self.equations)+1, self.nfields)
        self.M_eqn = defaultdict(eqn_factory)
        self.L_eqn = defaultdict(eqn_factory)

        bc_factory = lambda: sy.zeros(len(self.boundary_conditions)+1, self.nfields)
        self.M_bc = defaultdict(bc_factory)
        self.L_bc = defaultdict(bc_factory)

        namespace = self.namespace = {}
        namespace['t'] = sy.Symbol('t')
        namespace['dt'] = sy.Symbol('dt')
        reserved_names = tuple(namespace.keys())

        def split_func(string):
            head, sep, tail = string.partition('(')
            if tail:
                args = tail.strip(')').split(',')
            else:
                args = []
            return head, args

        def checkname(name):
            if not name.isidentifier():
                raise SymbolicParsingError("Name '%s' is an invalid identifier." %name)
            if name in reserved_names:
                raise SymbolicParsingError("Name '%s' is reserved." %name)
            if name in namespace:
                raise SymbolicParsingError("Name '%s' is used multiple times." %name)

        const_params = []
        nonconst_params = []
        separable_ops = []
        coupled_ops = []

        self.axis_names = axis_names
        self.field_names = field_names
        self.const_params = [param for param in param_names if not bool(split_func(param)[1])]
        self.nonconst_params = [param for param in param_names if bool(split_func(param)[1])]
        self.separable_ops = [op+ax for op in self.op_roots for ax in axis_names[:-1]]
        self.coupled_ops = [op+axis_names[-1] for op in self.op_roots] + self.default_ops

        # Axes: noncommutative symbols
        for axis in axis_names:
            checkname(axis)
            namespace[axis] = sy.Symbol(axis, commutative=False)
        # Fields: noncommutative symbols
        for field in field_names:
            checkname(field)
            namespace[field] = sy.Symbol(field, commutative=False)
        # Constant parameters: commutative symbols
        for param in self.const_params:
            checkname(param)
            namespace[param] = sy.Symbol(param)
        # Nonconstant parameters: functions
        for param in self.nonconst_params:
            head, args = split_func(param)
            checkname(head)
            namespace[head] = sy.Function(head)(*[namespace[arg] for arg in args])
        # Operators: function symbols
        # Separable operators: commutative symbols
        for op in self.separable_ops:
            checkname(op)
            namespace[op] = sy.Symbol(op)
        # Coupled operators: noncommutative symbols
        for op in self.coupled_ops:
            checkname(op)
            namespace[op] = sy.Symbol(op, commutative=False)

        # NOTES
        # Replacing dt function with commutative symbol requires no time dependence outside of fields
        # Requires no time dependence outside of fields
        # (to replace dt function on fields with commutative symbol)
        # Requires no transverse dependence outside of fields
        # (to replace separable operators acting on fields with commutative symbols)
        # Replace coupled operators acting on fields with non-commutative symbols

    def add_equation(self, eqn_str, condition="True"):
        """Add equation to problem."""

        LHS_str, RHS_str = eqn_str.split("=")
        # Parse LHS to symbolic operators
        expr = self.parse_linear(LHS_str)
        # Build equation dictionary
        eqn = dict()
        eqn['LHS_str'] = LHS_str
        eqn['RHS_str'] = RHS_str
        eqn['condition'] = condition
        dz = self.namespace['d'+self.axis_names[-1]]
        eqn['differential'] = expr.has(dz)
        # Split operator terms
        # split_terms = self.split_operator_terms(expr)
        # Check equation conditions
        # self.check_equation_operators(split_terms)
        # Add equation to system
        split_terms = expr
        self.extract_coefficients(split_terms, self.M_eqn, self.L_eqn)
        self.equations.append(eqn)

    def add_bc(self, bc_str, condition="True"):
        """Add boundary condition to problem."""

        LHS_str, RHS_str = bc_str.split("=")
        # Parse LHS to symbolic operators
        expr = self.parse_linear(LHS_str)
        # Build boundary condition dictionary
        bc = dict()
        bc['LHS_str'] = LHS_str
        bc['RHS_str'] = RHS_str
        bc['condition'] = condition
        # Split operator terms
        # split_terms = self.split_operator_terms(expr)
        # Check bc conditions
        # self.check_bc_operators(split_terms)
        # Add bc to system
        split_terms = expr
        self.extract_coefficients(split_terms, self.M_bc, self.L_bc)
        self.boundary_conditions.append(bc)

    def parse_linear(self, expr):
        """Parse expression string into new LHS row."""

        def split_func(string):
            head, sep, tail = string.partition('(')
            if tail:
                args = tail.strip(')').split(',')
            else:
                args = []
            return head, args

        # References
        namespace = self.namespace
        axes = [namespace[axis] for axis in self.axis_names]
        fields = [namespace[field] for field in self.field_names]

        linear_op_names = ['dt'] + self.separable_ops + self.coupled_ops
        linear_operators = [sy.Function(op)for op in linear_op_names]
        coupled_operators = [namespace[op] for op in self.coupled_ops]
        ncc = [namespace[split_func(param)[0]] for param in self.nonconst_params]

        ## Parse string to symbolic expression
        expr = eval(expr, namespace).simplify().expand()
        # Temporarily swap axes with commutative symbols to float NCCs left in multiplications
        dummies = [sy.Dummy() for axis in axes]
        expr = expr.subs(zip(axes, dummies))
        expr = expr.subs(zip(dummies, axes))
        # Enforce unary input for linear operators
        def enforce_unary(subexpr):
            if type(subexpr) in linear_operators:
                if len(subexpr.args) > 1:
                    raise SymbolicParsingError("Linear operators only accept one argument: %s" %subexpr)
            return subexpr
        expr = bottom_up(expr, enforce_unary)
        # Distribute linear operators
        def distribute_subexpression(subexpr):
            if type(subexpr) in linear_operators:
                terms = sy.Add.make_args(subexpr.args[0])
                subexpr = sum(subexpr.func(term) for term in terms)
            return subexpr
        expr = bottom_up(expr, distribute_subexpression)

        logging.debug("Parsing result: %s" %expr)

        ## Convert to operator form
        # Require no transverse dependence in coefficients, so transverse modes
        # are separable and transverse operators are scalar/commutative
        if expr.has(*axes[:-1]):
            raise UnsupportedEquationError("LHS coefficients must be independent of transverse axes.")
        # Require no time dependence in coefficients, so dt is commutative
        if expr.has(namespace['t']):
            raise UnsupportedEquationError("LHS coefficients must be time-independent.")
        # Convert linear operators acting on fields to operator form
        def convert_subexpression(subexpr):
            if type(subexpr) in linear_operators:
                if subexpr.has(*fields):
                    symbol = namespace[subexpr.func.__name__]
                    subexpr = symbol * subexpr.args[0]
            return subexpr
        expr = bottom_up(expr, convert_subexpression)
        # Require linearity in fields
        for term in sy.Add.make_args(expr):
            factors = sy.Mul.make_args(term)
            if factors[-1] not in fields:
                raise UnsupportedEquationError("Last non-commutative term is not a field: %s" %term)
            if sy.Mul(*factors[:-1]).has(*fields):
                raise UnsupportedEquationError("Term is non-linear: %s" %term)

        logging.debug("Operator form: %s" %expr)
        return expr

    def check_equation_operators(self, split_terms):

        #dz =
        # Require no higher-than-first-order dz dependence
        for coeff, op, field in split_terms:
            powers = op.as_powers_dict()
            if powers[dz] > 1:
                raise UnsupportedEquationError("Equations must be first-order in vertical derivatives.")

    def check_bc_operators(self, split_terms):

        # First order dz?
        # Require functional form
        for coeff, op, field in split_terms:
            factors = sy.Mul.make_args(op)
            if factors[0] not in linear_functionals:
                raise SymbolicParsingError("Boundary conditions terms must have a linear functional as the final operator: %s" %term)

    def extract_coefficients(self, expr, M_coeffs, L_coeffs):

        namespace = self.namespace
        fields = [namespace[field] for field in self.field_names]
        matrices = [namespace[name] for name in self.coupled_ops+self.nonconst_params]

        ## Extract coefficients
        # Split into form dt*M + L, requiring no higher-order dt dependence
        dt = namespace['dt']
        L, M = expr.as_independent(dt, as_Add=True)
        M = (M/dt).simplify().expand()
        if M.has(dt):
            raise UnsupportedEquationError("Equations must be first-order in time derivatives.")
        # Add row to existing operator coefficient matrices
        # (New matrices will be expanded via factory closure)
        new_row = sy.zeros(1, self.nfields)
        for coeffs in (M_coeffs, L_coeffs):
            for op in coeffs:
                coeffs[op] = coeffs[op].col_join(new_row)
        # Add expressions to operator coefficient matrices
        for expr, coeffs in ((M, M_coeffs), (L, L_coeffs)):
            if expr != 0:
                for term in sy.Add.make_args(expr):
                    # Split into (scalar coeff)*(compound op)*(field)
                    field = sy.Mul.make_args(term)[-1]
                    coeff, compound_op = (term/field).as_independent(*matrices)
                    # Add coeff to operator coefficient matrix
                    f = fields.index(field)
                    coeffs[compound_op][-1, f] += coeff

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
        for i in range(domain.dim):
            domain.bases[i].name = self.axis_names[i]

        # Check parameters types
        for param in self.const_params:
            if not np.isscalar(self.parameters[param]):
                raise ValueError("%s parameter must be a scalar." %param)
        for param in self.nonconst_params:
            if not isinstance(self.parameters[param], Field):
                raise ValueError("%s parameter must be a field." %param)

        # Convert M_eqn, L_eqn, M_bc, L_bc, F_eqn, F_bc
        self.num_M_eqn = self.num_coeffs(self.M_eqn, domain, order)
        self.num_L_eqn = self.num_coeffs(self.L_eqn, domain, order)
        self.num_M_bc = self.num_coeffs(self.M_bc, domain, order)
        self.num_L_bc = self.num_coeffs(self.L_bc, domain, order)

    def num_coeffs(self, sym_coeffs, domain, order):

        coupled_ops = {}
        for op_root in self.op_roots:
            op_name = operators.root_dict[op_root].__name__
            op = getattr(domain.bases[-1], op_name, None)
            if op is not None:
                coupled_ops[op_root + domain.bases[-1].name] = op
        for op in self.default_ops:
            coupled_ops[op] = operators.op_dict[op]

        def op_matrix(op, domain):
            if op == 1:
                return 1
            if op.name in self.coupled_ops:
                # Lookup through domain
                dummy = domain.new_field()
                return coupled_ops[op.name](dummy).matrix_form()
            else:
                # Evaluate NCCs as operator trees
                print(op)
                operator = operators.Operator.from_string(op, self.parameters, domain)
                field = operator.evaluate()
                field.require_coeff_space()
                # Scatter const coeffs from root
                if domain.dist.rank == 0:
                    select = (0,) * (domain.dim - 1)
                    coeffs = field['c'][select]
                else:
                    coeffs = None
                coeffs = domain.dist.comm_cart.scatter(coeffs, root=0)
                return domain.bases[-1].build_mult(coeffs, order)

        def compound_op_matrix(compound_op, domain):
            factors = sy.Mul.make_args(compound_op)
            matrices = (op_matrix(factor, domain) for factor in factors)
            return reduce(operator.mul, matrices, 1)

        num_coeffs = []
        for sym_op, sym_coeff in sym_coeffs.items():
            # Fix parameters
            const_params = {pn: self.parameters[pn] for pn in self.const_params}
            # Build operator matrices using NCCs
            num_op = compound_op_matrix(sym_op, domain)
            # Build coefficient matrices using CCs (still depend on separable ops)
            expr = sym_coeff.subs(const_params)
            args = [self.namespace[op] for op in self.separable_ops]
            num_coeffs.append((num_op, sy.lambdify(args, expr, modules='math')))

        return num_coeffs

    def system_selection(self, args):
        """Build selection matrices for given pencil index."""

        # References
        nfields = self.nfields
        neqns = self.neqns
        nbcs = self.nbcs
        namespace = dict(zip(self.separable_ops, args))

        # Select equations
        eqn_select = []
        for e, eqn in enumerate(self.equations):
            if eval(eqn['condition'], namespace):
                eqn_select.append(e)
        # Count differential equations
        diff_eqn = []
        for se, e in enumerate(eqn_select):
            if self.equations[e]['differential']:
                diff_eqn.append(se)
        # Select boundary conditions
        bc_select = []
        for b, bc in enumerate(self.boundary_conditions):
            if eval(bc['condition'], namespace):
                bc_select.append(b)

        # Check selections
        # Need as many equations as fields
        if len(eqn_select) != nfields:
            raise ValueError("%i equations for %i fields for pencil index: %s" %index)
        # Need no more boundary conditions than differential eqns
        # For polynomial bases, should be the same, but this is a hack for fully Fourier
        if len(bc_select) > len(diff_eqn):
            raise ValueError("Too many boundary conditions for pencil index: %s" %index)

        # Build equation selection array
        Se = np.zeros((nfields, neqns), dtype=int)
        for se, e in enumerate(eqn_select):
            Se[se, e] = 1
        # Build differential and algebraic filter arrays
        D = np.zeros((nfields, nfields), dtype=int)
        for se in diff_eqn:
            D[se, se] = 1
        A = np.identity(nfields) - D
        # Build boundary condition selection array
        Sb = np.zeros((nfields, nbcs), dtype=int)
        for bs, b in enumerate(bc_select):
            se = diff_eqn[bs]
            sb[se, b] = 1

        # Cast arrays as matrices for easy linear algebra
        return np.matrix(Se), np.matrix(Sb), np.matrix(A), np.matrix(D)

