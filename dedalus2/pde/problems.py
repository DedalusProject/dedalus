"""
Classes for representing systems of equations.

"""

import re
from collections import OrderedDict
import numpy as np
import sympy as sy
from mpi4py import MPI

from ..data.metadata import MultiDict, Metadata
from ..data import field
from ..data import future
from ..data import operators
from ..tools import parsing
from ..tools.cache import CachedAttribute
from ..tools.cache import CachedMethod
from ..tools.exceptions import SymbolicParsingError
from ..tools.exceptions import UnsupportedEquationError

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class Namespace(OrderedDict):

    def allow_overwrites(self):
        self._allow_overwrites = True

    def disallow_overwrites(self):
        self._allow_overwrites = False

    def __setitem__(self, key, value):
        if key in self:
            if not self._allow_overwrites:
                raise SymbolicParsingError("Name '{}' is used multiple times.".format(key))
        else:
            if not key.isidentifier():
                raise SymbolicParsingError("Name '{}' is not a valid identifier.".format(key))
        super().__setitem__(key, value)

    def copy(self):
        copy = Namespace()
        copy.update(self)
        copy._allow_overwrites = self._allow_overwrites
        return copy

    def add_substitutions(self, substitutions):
        for call, result in substitutions.items():
            # Convert function calls to lambda expressions
            head, func = parsing.lambdify_functions(call, result)
            # Evaluate in current namespace
            self[head] = eval(func, self)


class ProblemBase:
    """
    PDE definitions using string representations.

    Equations are assumed to take the form ('LHS = RHS'), where the left-hand
    side contains terms that are linear in the dependent variables (and will be
    represented by coefficient matrices), and the right-hand side contains terms
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

    Currently, nonconstant coefficients must be written as functions of 'z', or
    be arrays of the shape of zbasis.grid, i.e. defined on the global z grid on
    each process.

    """


    def __init__(self, domain, variables):

        self.domain = domain
        self.variables = variables
        self.nvars = len(variables)

        self.meta = MultiDict({var: Metadata(domain) for var in variables})

        self.equations = self.eqs = []
        self.boundary_conditions = self.bcs = []

        self.parameters = OrderedDict()
        self.substitutions = OrderedDict()

        self.ncc_manager = NCCManager(self)

    @CachedAttribute
    def namespace(self):

        namespace = Namespace()
        namespace.disallow_overwrites()
        # Basis-specific items
        for axis, basis in enumerate(self.domain.bases):
            # Grids
            grid = field.Array(self.domain, name=basis.name)
            grid.from_global_vector(basis.grid(basis.dealias), axis)
            namespace[basis.name] = grid
            # Basis operators
            for op in basis.operators:
                namespace[op.name] = op
        # Fields
        for var in self.variables:
            namespace[var] = self.domain.new_field(name=var, allocate=False)
            namespace[var].meta = self.meta[var]
        # Parameters
        for name, param in self.parameters.items():
            # Cast parameters to operands
            casted_param = field.Operand.cast(param)
            casted_param.name = name
            namespace[name] = casted_param
        # Built-in functions
        namespace['Identity'] = 1
        namespace.update(operators.parseables)
        # Substitutions
        namespace.add_substitutions(self.substitutions)

        return namespace

    @CachedAttribute
    def coefficient_namespace(self):

        namespace = Namespace()
        namespace.disallow_overwrites()
        # Imaginary number
        namespace['I'] = 1j
        # NCC
        self.ncc_manager.build_coefficients()
        namespace['NCC'] = self.ncc_manager
        # Coupled basis operators
        for basis in self.domain.bases:
            if not basis.separable:
                for op in basis.operators:
                    namespace[op.name] = op
        # Parameters
        namespace.update(self.parameters)

        #namespace.allow_overwrites()
        return namespace

    def _build_basic_dictionary(self, equation, condition):
        LHS, RHS = parsing.split_equation(equation)
        dct = dict()
        dct['raw_equation'] = equation
        dct['raw_condition'] = condition
        dct['raw_LHS'] = LHS
        dct['raw_RHS'] = RHS
        logger.debug("  Condition: {}".format(condition))
        logger.debug("  LHS string form: {}".format(LHS))
        logger.debug("  RHS string form: {}".format(RHS))
        return dct

    def _build_object_forms(self, dct):
        """Parse raw LHS/RHS strings to object forms."""
        LHS = future.FutureField.parse(dct['raw_LHS'], self.namespace, self.domain)
        RHS = future.FutureField.parse(dct['raw_RHS'], self.namespace, self.domain)
        logger.debug("  LHS object form: {}".format(LHS))
        logger.debug("  RHS object form: {}".format(RHS))
        return LHS, RHS

    def _check_eqn_conditions(self, LHS, RHS):
        """Check object-form equation conditions."""
        self._check_meta_consistency(LHS, RHS)

    def _check_bc_conditions(self, LHS, RHS):
        """Check object-form BC conditions."""
        self._check_meta_consistency(LHS, RHS)
        self._check_boundary_form(LHS, RHS)

    def _check_meta_consistency(self, LHS, RHS):
        """Check LHS and RHS metadata for compatability."""
        default_meta = Metadata(self.domain)
        for axis in range(self.domain.dim):
            for key in default_meta[axis]:
                check = getattr(self, '_check_meta_%s' %key)
                check(LHS.meta[axis][key], RHS.meta[axis][key], axis)

    def _check_meta_scale(self, LHS_scale, RHS_scale, axis):
        # Solve occurs in coefficient space, so disregard scale
        pass

    def _check_meta_constant(self, LHS_constant, RHS_constant, axis):
        # RHS must be constant if LHS is constant
        if LHS_constant:
            if not RHS_constant:
                raise SymbolicParsingError("LHS is constant but RHS is nonconstant along {} axis.".format(self.domain.bases[axis].name))

    def _check_meta_parity(self, LHS_parity, RHS_parity, axis):
        if RHS_parity:
            # Parities must match
            if LHS_parity != RHS_parity:
                raise SymbolicParsingError("LHS and RHS parities along {} axis do not match.".format(self.domain.bases[axis].name))

    def _check_boundary_form(self, LHS, RHS):
        # Check that boundary expressions are constant along coupled axes
        for ax, basis in enumerate(self.domain.bases):
            if not basis.separable:
                if (not LHS.meta[ax]['constant']) or (not RHS.meta[ax]['constant']):
                    raise SymbolicParsingError("Boundary condition must be constant along '{}'.".format(basis.name))

    def _build_operator_form(self, LHS):
        """Convert operator tree to linear operator form on variables."""
        vars = [self.namespace[var] for var in self.variables]
        LHS = LHS.distribute_over(vars)
        op_form = LHS.as_symbolic_operator(vars)
        logger.debug("  LHS operator form: {}".format(op_form))
        if op_form is None:
            raise UnsupportedEquationError("LHS must contain terms that are linear in fields.")
        return op_form.simplify().expand()

    def _split_linear(self, expr, x):
        """Split expression into independent and linear parts in x."""
        indep, linear, nonlinear = self._separate_linear(expr, x)
        linear = (linear/x).simplify().expand()
        if nonlinear:
            raise UnsupportedEquationError("Equations must be first-order in '{}'.".format(x))
        return indep, linear

    @staticmethod
    def _separate_linear(expr, *syms):
        """Separate terms into independent, linear, and nonlinear parts."""
        indep, linear, nonlinear = 0, 0, 0
        for term in sy.Add.make_args(expr):
            powers = term.expand().as_powers_dict()
            total = sum(powers[sym] for sym in syms)
            if total == 0:
                indep += term
            elif total == 1:
                linear += term
            else:
                nonlinear += term

        return indep, linear, nonlinear

    def stringify_variable_coefficients(self, expr):
        # Extract symbolic variable coefficients
        vars = [sy.Symbol(var, commutative=False) for var in self.variables]
        coeffs = self._extract_left_coefficients(expr, vars)
        # Stringify coefficients
        stringforms = []
        for coeff in coeffs:
            terms = []
            coeff = sy.sympify(coeff)
            if coeff == 0:
                stringform = '0'
            elif coeff == 1:
                stringform = '1'
            else:
                for term in sy.Add.make_args(coeff):
                    term *= sy.numbers.One()
                    factors = sy.Mul.make_args(term)
                    factors = partition(self.categorize, factors)
                    factors = [self.category_to_string(*f) for f in factors]
                    stringform = '*'.join(factors)
                    # Multiply scalar terms by identity
                    if term == 1:
                        stringform = 'Identity'
                    elif term.is_commutative:
                        stringform += '*Identity'
                    terms.append(stringform)
                stringform = ' + '.join(terms)
            stringforms.append(stringform)

        return stringforms

    @staticmethod
    def _extract_left_coefficients(expr, symbols):
        """Extract left coefficients from a linear expression."""
        expr = sy.sympify(expr)
        # Extract coefficients
        coeffs = [0] * len(symbols)
        for i, sym in enumerate(symbols):
            expr = expr.simplify().expand()
            for term in sy.Add.make_args(expr):
                factors = sy.Mul.make_args(term)
                if factors[-1] is sym:
                    coeffs[i] += sy.Mul(*factors[:-1])
            expr = expr - coeffs[i]*sym
        # Check for completion
        if expr.simplify().expand() != 0:
            raise ValueError("Leftover: {}".format(remaining))

        return coeffs

    def _require_independent(self, operand, dep):
        """Require NCCs to be independent of some atom."""
        if isinstance(operand, field.Operand):
            if operand.has(dep):
                raise UnsupportedEquationError("'{}' is not independent of '{}'.".format(operand, dep))

    def _coupled_differential_order(self, expr):
        """Find coupled differential order of an expression, and require to be first order."""
        # Find coupled derivative operators
        first_order = [basis.Differentiate for basis in self.domain.bases if not basis.separable]
        first_order = [sy.Symbol(op.name, commutative=False) for op in first_order]
        # Separate terms based on dependence
        _, linear, nonlinear = self._separate_linear(expr, *first_order)
        if nonlinear:
            raise UnsupportedEquationError("Equations must be first-order in coupled derivatives.")
        return bool(linear)

    def categorize(self, factor):
        if factor.is_commutative:
            return 'scalar'
        else:
            expr = eval(str(factor), self.namespace)
            if isinstance(expr, type):
                return 'operator'
            else:
                return 'ncc'

    def category_to_string(self, category, itemlist):
        if category == 'scalar':
            out = [repr(item) for item in itemlist]
        elif category == 'operator':
            out = ['{}.matrix_form()'.format(item.name) for item in itemlist]
        elif category == 'ncc':
            # Multiply items into one NCC
            ncc_str = repr(sy.Mul(*itemlist))
            # Register NCC string
            self.ncc_manager.register(ncc_str)
            out = ["NCC('{}')".format(ncc_str)]
        return '*'.join(out)


class IVP(ProblemBase):

    def __init__(self, domain, variables, time='t', **kw):

        super().__init__(domain, variables, **kw)
        self.time = time

    @CachedAttribute
    def namespace(self):

        # Build time derivative operator
        class dt(operators.Separable):
            name = 'd' + self.time

            def as_symbolic_operator(self, vars):
                if not self.args[0].has(*vars):
                    raise ValueError("Cannot take time derivative of non-variable.")
                else:
                    return super().as_symbolic_operator(vars)
            def meta_constant(self, axis):
                # Preserves constancy
                return self.args[0].meta[axis]['constant']
            def meta_parity(self, axis):
                # Preserves parity
                return self.args[0].meta[axis]['parity']

        # Add time derivative operator and scalar to base namespace
        namespace = super().namespace
        namespace[self.time] = field.Scalar(name=self.time)
        namespace[dt.name] = dt

        return namespace

    def _check_eqn_conditions(self, LHS, RHS):
        super()._check_eqn_conditions(LHS, RHS)
        self._require_independent(LHS, self.namespace[self.time])

    def _check_bc_conditions(self, LHS, RHS):
        super()._check_bc_conditions(LHS, RHS)
        self._require_independent(LHS, self.namespace[self.time])

    def add_equation(self, equation, condition="True"):
        """Add equation to problem."""

        logger.debug("Parsing Eqn {}".format(len(self.eqs)))
        dct = self._build_basic_dictionary(equation, condition)

        LHS, RHS = self._build_object_forms(dct)
        self._check_eqn_conditions(LHS, RHS)

        LHS = self._build_operator_form(LHS)
        dct['differential'] = self._coupled_differential_order(LHS)

        L, M = self._split_linear(LHS, sy.Symbol('d'+self.time))
        dct['L'] = self.stringify_variable_coefficients(L)
        dct['M'] = self.stringify_variable_coefficients(M)

        self.equations.append(dct)

    def add_bc(self, equation, condition="True"):
        """Add equation to problem."""

        logger.debug("Parsing BC {}".format(len(self.bcs)))
        dct = self._build_basic_dictionary(equation, condition)

        LHS, RHS = self._build_object_forms(dct)
        self._check_bc_conditions(LHS, RHS)

        LHS = self._build_operator_form(LHS)
        dct['differential'] = self._coupled_differential_order(LHS)

        L, M = self._split_linear(LHS, sy.Symbol('d'+self.time))
        dct['L'] = self.stringify_variable_coefficients(L)
        dct['M'] = self.stringify_variable_coefficients(M)

        self.boundary_conditions.append(dct)


class NCCManager:

    def __init__(self, problem, cutoff=1e-10, max_terms=None):
        self.problem = problem
        self.domain = problem.domain
        self.basis = problem.domain.bases[-1]
        self.ncc_strings = set()
        self.ncc_coeffs = {}
        self.ncc_matrices = {}

        self.cutoff = cutoff
        if max_terms is None:
            max_terms = self.basis.coeff_size
        self.max_terms = max_terms

    def register(self, ncc_str):
        self.ncc_strings.add(ncc_str)

    def build_coefficients(self):
        namespace = self.problem.namespace
        domain = self.domain
        # Compute NCC coefficients in sorted order for proper parallel evaluation
        for ncc_str in sorted(self.ncc_strings):
            # Evaluate NCC as field
            ncc = future.FutureField.parse(ncc_str, namespace, domain)
            ncc = ncc.evaluate()
            ncc.require_coeff_space()
            # Scatter transverse-constant coefficients
            if domain.dist.rank == 0:
                select = (0,) * (domain.dim - 1)
                coeffs = ncc['c'][select]
            else:
                coeffs = np.zeros(domain.bases[-1].coeff_size, dtype=domain.bases[-1].coeff_dtype)
            domain.dist.comm_cart.Bcast(coeffs, root=0)
            self.ncc_coeffs[ncc_str] = coeffs.copy()
            # Build matrix
            n_terms, matrix = self.basis.NCC(coeffs, cutoff=self.cutoff, max_terms=self.max_terms)
            self.ncc_matrices[ncc_str] = matrix
            logger.info("Constructed NCC '{}' with {} terms.".format(ncc_str, n_terms))

    def __call__(self, ncc_str):
        return self.ncc_matrices[ncc_str]


def partition(categorize, items):

    current_category = None
    partitions = []

    for item in items:
        category = categorize(item)
        if category == current_category:
            current_list.append(item)
        else:
            current_category = category
            current_list = [item]
            partitions.append((current_category, current_list))

    return partitions

