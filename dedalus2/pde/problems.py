"""
Classes for representing systems of equations.

"""

import re
from collections import OrderedDict, defaultdict
import numpy as np
from mpi4py import MPI

from ..data.metadata import MultiDict, Metadata
from ..data import field
from ..data.field import Operand
from ..data import future
from ..data import operators
from . import solvers
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


    def __init__(self, domain, variables, **kw):
        self.domain = domain
        self.variables = variables
        self.nvars = len(variables)
        self.meta = MultiDict({var: Metadata(domain) for var in variables})
        self.equations = self.eqs = []
        self.boundary_conditions = self.bcs = []
        self.parameters = OrderedDict()
        self.substitutions = OrderedDict()
        self.kw = kw

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
        namespace.update(operators.parseables)
        # Substitutions
        namespace.add_substitutions(self.substitutions)

        return namespace

    def _build_basic_dictionary(self, temp, equation, condition):
        """Split and store equation and condition strings."""
        temp['raw_equation'] = equation
        temp['raw_condition'] = condition
        temp['raw_LHS'], temp['raw_RHS'] = parsing.split_equation(equation)
        logger.debug("  Condition: {}".format(condition))
        logger.debug("  LHS string form: {}".format(temp['raw_LHS']))
        logger.debug("  RHS string form: {}".format(temp['raw_RHS']))

    def _build_object_forms(self, temp):
        """Parse raw LHS/RHS strings to object forms."""
        temp['LHS'] = future.FutureField.parse(temp['raw_LHS'], self.namespace, self.domain)
        temp['RHS'] = future.FutureField.parse(temp['raw_RHS'], self.namespace, self.domain)
        logger.debug("  LHS object form: {}".format(temp['LHS']))
        logger.debug("  RHS object form: {}".format(temp['RHS']))

    def _check_eqn_conditions(self, temp):
        """Check object-form equation conditions."""
        self._check_conditions(temp)
        self._check_meta_consistency(temp['LHS'], temp['RHS'])

    def _check_bc_conditions(self, temp):
        """Check object-form BC conditions."""
        self._check_conditions(temp)
        self._check_meta_consistency(temp['LHS'], temp['RHS'])
        self._check_boundary_form(temp['LHS'], temp['RHS'])

    def _check_conditions(self, temp):
        self._check_differential_order(temp)

    def _check_differential_order(self, temp):
        """Find coupled differential order of an expression, and require to be first order."""
        coupled_diffs = [basis.Differentiate for basis in self.domain.bases if not basis.separable]
        order = self._require_first_order(temp['LHS'], coupled_diffs)
        temp['differential'] = bool(order)

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

    def _require_independent(self, expr, vars):
        """Require expression to be independent of some variables."""
        if expr.has(*vars):
            names = [var.name for var in vars]
            raise UnsupportedEquationError("LHS must be independent of {}.".format(names))

    def _require_first_order(self, expr, vars):
        """Require expression to be zeroth or first order in some variables."""
        order = expr.order(*vars)
        if order > 1:
            names = [var.name for var in vars]
            raise UnsupportedEquationError("LHS must be first-order in {}.".format(names))
        return order

    def _split_factors(self, temp, x):
        LHS = temp['LHS']
        # Split into independent and linear factors
        LHS = LHS.expand(x)
        factors = LHS.factor(x)
        try:
            x = x._scalar
        except:
            pass
        extra = set(factors).difference(set((1,x)))
        if extra:
            raise SymbolicParsingError('Other factors: {}'.format(','.join(map(str, extra))))

        self._check_linear_form(temp, factors[x], 'M')
        self._check_linear_form(temp, factors[1], 'L')

    def _check_linear_form(self, temp, expr, name):
        vars = [self.namespace[var] for var in self.variables]
        expr = Operand.cast(expr)
        expr = expr.expand(*vars)
        expr = expr.canonical_linear_form(*vars)
        logger.debug('  {} linear form: {}'.format(name, str(expr)))
        temp[name] = expr

    def build_solver(self, *args, **kw):
        return self.solver_class(self, *args, **kw)


class InitialValueProblem(ProblemBase):
    """Class for non-linear initial value problems."""

    solver_class = solvers.InitialValueSolver

    def __init__(self, domain, variables, time='t', **kw):
        super().__init__(domain, variables, **kw)
        self.time = time

    @CachedAttribute
    def namespace(self):
        """Add time and time derivative to default namespace."""
        class dt(operators.TimeDerivative):
            name = 'd' + self.time
            _scalar = field.Scalar(name=name)
        namespace = super().namespace
        namespace[self.time] = self._t = field.Scalar(name=self.time)
        namespace[dt.name] = self._dt = dt
        return namespace

    def _check_conditions(self, temp):
        """Check object-form conditions."""
        super()._check_conditions(temp)
        self._require_independent(temp['LHS'], [self._t])
        self._require_first_order(temp['LHS'], [self._dt])

    def add_equation(self, equation, condition="True"):
        """Add equation to problem."""
        logger.debug("Parsing Eqn {}".format(len(self.eqs)))
        temp = {}
        self._build_basic_dictionary(temp, equation, condition)
        self._build_object_forms(temp)
        self._check_eqn_conditions(temp)
        self._split_factors(temp, self._dt)
        self.eqs.append(temp)

    def add_bc(self, equation, condition="True"):
        """Add boundary condition to problem."""
        logger.debug("Parsing BC {}".format(len(self.bcs)))
        temp = {}
        self._build_basic_dictionary(temp, equation, condition)
        self._build_object_forms(temp)
        self._check_bc_conditions(temp)
        self._split_factors(temp, self._dt)
        self.bcs.append(temp)


class BoundaryValueProblem(ProblemBase):
    """Class for inhomogeneous, linear boundary value problems."""

    solver_class = solvers.BoundaryValueSolver

    def _check_conditions(self, temp):
        """Check object-form conditions."""
        super()._check_conditions(temp)
        vars = [self.namespace[var] for var in self.variables]
        self._require_independent(temp['RHS'], vars)

    def add_equation(self, equation, condition="True"):
        """Add equation to problem."""
        logger.debug("Parsing Eqn {}".format(len(self.eqs)))
        temp = {}
        self._build_basic_dictionary(temp, equation, condition)
        self._build_object_forms(temp)
        self._check_eqn_conditions(temp)
        self._check_linear_form(temp, temp['LHS'], 'L')
        self.eqs.append(temp)

    def add_bc(self, equation, condition="True"):
        """Add boundary condition to problem."""
        logger.debug("Parsing BC {}".format(len(self.bcs)))
        temp = {}
        self._build_basic_dictionary(temp, equation, condition)
        self._build_object_forms(temp)
        self._check_bc_conditions(temp)
        self._check_linear_form(temp, temp['LHS'], 'L')
        self.bcs.append(temp)


class EigenvalueProblem(ProblemBase):
    """Class for linear eigenvalue problems."""

    solver_class = solvers.EigenvalueSolver

    def __init__(self, domain, variables, eigenvalue, **kw):
        super().__init__(domain, variables, **kw)
        self.eigenvalue = eigenvalue

    @CachedAttribute
    def namespace(self):
        """Add eigenvalue to default namespace."""
        namespace = super().namespace
        namespace[self.eigenvalue] = self._ev = field.Scalar(name=self.eigenvalue)
        return namespace

    def _check_conditions(self, temp):
        """Check object-form conditions."""
        super()._check_conditions(temp)
        self._require_first_order(temp['LHS'], [self._ev])
        self._require_zero(temp['RHS'])

    def add_equation(self, equation, condition="True"):
        """Add equation to problem."""
        logger.debug("Parsing Eqn {}".format(len(self.eqs)))
        temp = {}
        self._build_basic_dictionary(temp, equation, condition)
        self._build_object_forms(temp)
        self._check_eqn_conditions(temp)
        self._split_factors(temp, self._ev)
        self.eqs.append(temp)

    def add_bc(self, equation, condition="True"):
        """Add boundary condition to problem."""
        logger.debug("Parsing BC {}".format(len(self.bcs)))
        temp = {}
        self._build_basic_dictionary(temp, equation, condition)
        self._build_object_forms(temp)
        self._check_bc_conditions(temp)
        self._split_factors(temp, self._ev)
        self.bcs.append(temp)


# Aliases
IVP = InitialValueProblem
BVP = BoundaryValueProblem
EVP = EigenvalueProblem

