"""
Classes for representing systems of equations.

"""

from collections import OrderedDict
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
from ..tools.exceptions import SymbolicParsingError
from ..tools.exceptions import UnsupportedEquationError

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class Namespace(OrderedDict):
    """
    Class ensuring a conflict-free namespace for parsing.  This class just wraps
    an OrderedDict, making sure that keys are valid python identifiers, and
    disallowing overwrites of previously-supplied keys.

    """

    def __init__(self):
        self.allow_overwrites = False

    def __setitem__(self, key, value):
        if key in self:
            if not self.allow_overwrites:
                raise SymbolicParsingError("Name '{}' is used multiple times.".format(key))
        else:
            if not key.isidentifier():
                raise SymbolicParsingError("Name '{}' is not a valid identifier.".format(key))
        super().__setitem__(key, value)

    def copy(self):
        """Copy entire namespace."""
        copy = Namespace()
        copy.update(self)
        copy.allow_overwrites = self.allow_overwrites
        return copy

    def add_substitutions(self, substitutions):
        """Parse substitutions in current namespace before adding to self."""
        for call, result in substitutions.items():
            # Convert function calls to lambda expressions
            head, func = parsing.lambdify_functions(call, result)
            # Evaluate in current namespace
            self[head] = eval(func, self)


class ProblemBase:
    """
    Base class for problems consisting of a system of PDEs, constraints, and
    boundary conditions.

    Parameters
    ----------
    domain : domain object
        Problem domain
    variables : list of str
        List of variable names, e.g. ['u', 'v', 'w']

    Attributes
    ----------
    parameters : OrderedDict
        External parameters used in the equations, and held constant during integration.
    substitutions : OrderedDict
        String-substitutions to be used in parsing.

    Notes
    -----
    Equations are entered as strings of the form 'LHS = RHS', where the
    left-hand side contains terms that are linear in the dependent variables
    (and will be parsed into a sparse matrix system), and the right-hand side
    contains terms that are non-linear (and will be parsed into operator trees).
    The LHS terms must be first-order in coupled derivatives.

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

    def add_equation(self, equation, condition="True"):
        """Add equation to problem."""
        logger.debug("Parsing Eqn {}".format(len(self.eqs)))
        temp = {}
        self._build_basic_dictionary(temp, equation, condition)
        self._build_object_forms(temp)
        self._check_eqn_conditions(temp)
        self._set_matrix_expressions(temp)
        self.eqs.append(temp)

    def add_bc(self, equation, condition="True"):
        """Add boundary condition to problem."""
        logger.debug("Parsing BC {}".format(len(self.bcs)))
        temp = {}
        self._build_basic_dictionary(temp, equation, condition)
        self._build_object_forms(temp)
        self._check_bc_conditions(temp)
        self._set_matrix_expressions(temp)
        self.bcs.append(temp)

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

    @CachedAttribute
    def namespace(self):
        """Build namespace for problem parsing."""
        namespace = Namespace()
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

    def _check_eqn_conditions(self, temp):
        """Check object-form equation conditions."""
        self._check_conditions(temp)
        self._check_differential_order(temp)
        self._check_meta_consistency(temp['LHS'], temp['RHS'])

    def _check_bc_conditions(self, temp):
        """Check object-form BC conditions."""
        self._check_conditions(temp)
        self._check_differential_order(temp)
        self._check_meta_consistency(temp['LHS'], temp['RHS'])
        self._check_boundary_form(temp['LHS'], temp['RHS'])

    def _check_differential_order(self, temp):
        """Find coupled differential order of an expression, and require to be first order."""
        coupled_diffs = [basis.Differentiate for basis in self.domain.bases if not basis.separable]
        order = self._require_first_order(temp, 'LHS', coupled_diffs)
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
        """Check that RHS is constant if LHS is consant."""
        if LHS_constant:
            if not RHS_constant:
                raise SymbolicParsingError("LHS is constant but RHS is nonconstant along {} axis.".format(self.domain.bases[axis].name))

    def _check_meta_parity(self, LHS_parity, RHS_parity, axis):
        """Check that LHS parity matches RHS parity, if RHS parity is nonzero."""
        if RHS_parity:
            if LHS_parity != RHS_parity:
                raise SymbolicParsingError("LHS and RHS parities along {} axis do not match.".format(self.domain.bases[axis].name))

    def _check_boundary_form(self, LHS, RHS):
        """Check that boundary expressions are constant along coupled axes."""
        for ax, basis in enumerate(self.domain.bases):
            if not basis.separable:
                if (not LHS.meta[ax]['constant']) or (not RHS.meta[ax]['constant']):
                    raise SymbolicParsingError("Boundary condition must be constant along '{}'.".format(basis.name))

    def _require_zero(self, temp, key):
        """Require expression to be equal to zero."""
        if temp[key] != 0:
            raise UnsupportedEquationError("{} must be zero.".format(key))

    def _require_independent(self, temp, key, vars):
        """Require expression to be independent of some variables."""
        if temp[key].has(*vars):
            names = [var.name for var in vars]
            raise UnsupportedEquationError("{} must be independent of {}.".format(key, names))

    def _require_first_order(self, temp, key, vars):
        """Require expression to be zeroth or first order in some variables."""
        order = temp[key].order(*vars)
        if order > 1:
            names = [var.name for var in vars]
            raise UnsupportedEquationError("{} must be first-order in {}.".format(key, names))
        return order

    def _factor_first_order(self, expr, x):
        """Factor an expression into independent and linear parts wrt some variable."""
        expr = expr.expand(x)
        factors = expr.factor(x)
        try:
            x = x._scalar
        except:
            pass
        extra = set(factors).difference(set((1,x)))
        if extra:
            raise SymbolicParsingError('Other factors: {}'.format(','.join(map(str, extra))))
        return factors[1], factors[x]

    def _set_linear_form(self, temp, expr, name):
        """Convert an expression into suitable form for LHS operator conversion."""
        vars = [self.namespace[var] for var in self.variables]
        expr = Operand.cast(expr)
        expr = expr.expand(*vars)
        expr = expr.canonical_linear_form(*vars)
        logger.debug('  {} linear form: {}'.format(name, str(expr)))
        temp[name] = expr

    def build_solver(self, *args, **kw):
        """Build corresponding solver class."""
        return self.solver_class(self, *args, **kw)


class InitialValueProblem(ProblemBase):
    """
    Class for non-linear initial value problems.

    Parameters
    ----------
    domain : domain object
        Problem domain
    variables : list of str
        List of variable names, e.g. ['u', 'v', 'w']
    time : str, optional
        Time label, default: 't'

    Notes
    -----
    This class supports non-linear initial value problems.  The LHS
    terms must be linear in the specified variables, first-order in time
    derivatives, and contain no explicit time dependence.

    """

    solver_class = solvers.InitialValueSolver

    def __init__(self, domain, variables, time='t', **kw):
        super().__init__(domain, variables, **kw)
        self.time = time

    @CachedAttribute
    def namespace(self):
        """Build namespace for problem parsing."""
        class dt(operators.TimeDerivative):
            name = 'd' + self.time
            _scalar = field.Scalar(name=name)
        namespace = super().namespace
        namespace[self.time] = self._t = field.Scalar(name=self.time)
        namespace[dt.name] = self._dt = dt
        return namespace

    def _check_conditions(self, temp):
        """Check object-form conditions."""
        self._require_independent(temp, 'LHS', [self._t])
        self._require_first_order(temp, 'LHS', [self._dt])

    def _set_matrix_expressions(self, temp):
        """Set expressions for building LHS matrices."""
        L, M = self._factor_first_order(temp['LHS'], self._dt)
        self._set_linear_form(temp, L, 'L')
        self._set_linear_form(temp, M, 'M')


class BoundaryValueProblem(ProblemBase):
    """
    Class for inhomogeneous, linear boundary value problems.

    Parameters
    ----------
    domain : domain object
        Problem domain
    variables : list of str
        List of variable names, e.g. ['u', 'v', 'w']

    Notes
    -----
    This class supports inhomogeneous, linear boundary value problems.  The LHS
    terms must be linear in the specified variables, and the RHS must be
    independent of the specified variables.

    """

    solver_class = solvers.BoundaryValueSolver

    def _check_conditions(self, temp):
        """Check object-form conditions."""
        vars = [self.namespace[var] for var in self.variables]
        self._require_independent(temp, 'RHS', vars)

    def _set_matrix_expressions(self, temp):
        """Set expressions for building LHS matrices."""
        self._set_linear_form(temp, temp['LHS'], 'L')


class EigenvalueProblem(ProblemBase):
    """
    Class for linear eigenvalue problems.

    Parameters
    ----------
    domain : domain object
        Problem domain
    variables : list of str
        List of variable names, e.g. ['u', 'v', 'w']
    eigenvalue : str
        Eigenvalue label, e.g. 'lambda'

    Notes
    -----
    This class supports linear eigenvalue problems.  The LHS terms must be
    linear in the specified variables, and linear or independent of the
    specified eigenvalue.  The RHS must be zero.

    """

    solver_class = solvers.EigenvalueSolver

    def __init__(self, domain, variables, eigenvalue, **kw):
        super().__init__(domain, variables, **kw)
        self.eigenvalue = eigenvalue

    @CachedAttribute
    def namespace(self):
        """Build namespace for problem parsing."""
        namespace = super().namespace
        namespace[self.eigenvalue] = self._ev = field.Scalar(name=self.eigenvalue)
        return namespace

    def _check_conditions(self, temp):
        """Check object-form conditions."""
        self._require_first_order(temp, 'LHS', [self._ev])
        self._require_zero(temp, 'RHS')

    def _set_matrix_expressions(self, temp):
        """Set expressions for building LHS matrices."""
        L, M = self._factor_first_order(temp['LHS'], self._ev)
        self._set_linear_form(temp, L, 'L')
        self._set_linear_form(temp, M, 'M')


# Aliases
IVP = InitialValueProblem
BVP = BoundaryValueProblem
EVP = EigenvalueProblem

