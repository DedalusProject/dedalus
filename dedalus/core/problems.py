"""
Classes for representing systems of equations.

"""

from collections import OrderedDict
import numpy as np
from mpi4py import MPI
from functools import reduce

from . import field
from .field import Operand, Field
from . import future
from . import operators
from . import solvers
from ..tools import parsing
from ..tools.cache import CachedAttribute
from ..tools.exceptions import SymbolicParsingError
from ..tools.exceptions import UnsupportedEquationError

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class Namespace(OrderedDict):
    """Class ensuring a conflict-free namespace for parsing."""

    __slots__ = 'allow_overwrites'

    def __init__(self):
        super().__init__()
        self.allow_overwrites = True

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
    ncc_cutoff : float, optional
        Mode amplitude cutoff for LHS NCC expansions (default: 1e-10)
    max_ncc_terms : int, optional
        Maximum terms to include in LHS NCC expansions (default: None (no limit))

    Attributes
    ----------
    parameters : OrderedDict
        External parameters used in the equations, and held constant during integration.
    substitutions : OrderedDict
        String-substitutions to be used in parsing.

    Notes
    -----
    Equations are entered as strings of the form "LHS = RHS", where the
    left-hand side contains terms that are linear in the dependent variables
    (and will be parsed into a sparse matrix system), and the right-hand side
    contains terms that are non-linear (and will be parsed into operator trees).

    The specified axes (from domain), variables, parameters, and substitutions
    are recognized by the parser, along with the built-in operators, which
    include spatial derivatives (of the form "dx()" for an axis named "x") and
    basic mathematical operators (trigonometric and logarithmic).

    The LHS terms must be linear in the specified variables and first-order in
    coupled derivatives.

    """

    def __init__(self, domain, ncc_cutoff=1e-10, max_ncc_terms=None):
        self.domain = domain
        self.variables = []
        self.equations = self.eqs = []
        self.parameters = OrderedDict()
        self.substitutions = OrderedDict()
        self.ncc_kw = {'cutoff': ncc_cutoff}

    def add_variable(self, var, bases=None):
        if isinstance(var, str):
            var = Field(name=var, bases=bases)
        self.variables.append(var)

    def add_equation(self, equation, condition="True"):
        """Add equation to problem."""
        logger.debug("Parsing Eqn {}".format(len(self.eqs)))
        temp = {}
        self._build_basic_dictionary(temp, equation, condition)
        self._build_object_forms(temp)
        self._check_eqn_conditions(temp)
        self._set_matrix_expressions(temp)
        self.eqs.append(temp)

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
        LHS = field.Operand.parse(temp['raw_LHS'], self.namespace, self.domain)
        RHS = field.Operand.parse(temp['raw_RHS'], self.namespace, self.domain)
        # Add together to require trigger proper conversions
        print(RHS, type(RHS))
        if RHS != 0:
            sum = LHS + RHS
            LHS, RHS = sum.args
        temp['LHS'] = LHS
        temp['RHS'] = RHS
        temp['bases'] = LHS.bases
        logger.debug("  LHS object form: {}".format(temp['LHS']))
        logger.debug("  RHS object form: {}".format(temp['RHS']))

    @CachedAttribute
    def namespace(self):
        """Build namespace for problem parsing."""
        namespace = Namespace()
        # Space-specific items
        for spaceset in self.domain.spaces:
            for space in spaceset:
                # Grid
                namespace[space.name] = space.grid_array(scales=None)
                # Operators
                namespace.update(space.operators)
        # Variables
        for var in self.variables:
            namespace[var.name] = var
        # Parameters
        # for name, param in self.parameters.items():
        #     # Cast parameters to operands
        #     casted_param = field.Operand.cast(param, self.domain)
        #     casted_param.name = name
        #     namespace[name] = casted_param
        namespace.update(self.parameters)
        # Built-in functions
        namespace.update(operators.parseables)
        # Additions from derived classes
        namespace.update(self.namespace_additions)
        # Substitutions
        namespace.add_substitutions(self.substitutions)

        return namespace

    def _check_eqn_conditions(self, temp):
        """Check object-form equation conditions."""
        self._check_conditions(temp)
        self._check_basis_consistency(temp['LHS'], temp['RHS'])

    def _check_basis_consistency(self, LHS, RHS):
        """Check LHS and RHS for basis consistency."""
        if not RHS.subdomain in LHS.subdomain:
            raise ValueError("RHS subdomain must be in LHS subdomain.")

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

    def _prep_linear_form(self, expr, vars, name=''):
        """Convert an expression into suitable form for LHS operator conversion."""
        if expr:
            expr = Operand.cast(expr, self.domain)
            expr = expr.expand(*vars)
            expr = expr.canonical_linear_form(*vars)
            logger.debug('  {} linear form: {}'.format(name, str(expr)))
        return (expr, vars)

    def build_solver(self, *args, **kw):
        """Build corresponding solver class."""
        return self.solver_class(self, *args, **kw)

    def separability(self):
        separabilities = [eq['separability'] for eq in self.eqs]
        return reduce(np.logical_and, separabilities)


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
    terms must be linear in the specified variables, first-order in coupled
    derivatives, first-order in time derivatives, and contain no explicit
    time dependence.

        M.dt(X) + L.X = F(X, t)

    """

    solver_class = solvers.InitialValueSolver

    def __init__(self, domain, time='t', **kw):
        super().__init__(domain, **kw)
        self.time = time

    @CachedAttribute
    def namespace_additions(self):
        """Build namespace for problem parsing."""

        class dt(operators.TimeDerivative):
            name = 'd' + self.time
            _scalar = field.Field(name=name, bases=self.domain)

            def base(self):
                return dt

        additions = {}
        additions[self.time] = self._t = field.Field(name=self.time, bases=self.domain)
        additions[dt.name] = self._dt = dt
        return additions

    def _check_conditions(self, temp):
        """Check object-form conditions."""
        self._require_independent(temp, 'LHS', [self._t])
        self._require_first_order(temp, 'LHS', [self._dt])

    def _set_matrix_expressions(self, temp):
        """Set expressions for building solver."""
        M, L = temp['LHS'].split(self._dt)
        if M:
            M = Operand.cast(M, self.domain)
            M = M.replace(self._dt, lambda x: x)
        #vars = [self.namespace[var] for var in self.variables]
        vars = self.variables
        temp['M'] = self._prep_linear_form(M, vars, name='M')
        temp['L'] = self._prep_linear_form(L, vars, name='L')
        temp['F'] = temp['RHS']
        if M and L:
            temp['separability'] = (temp['M'][0].separability(vars) &
                                    temp['L'][0].separability(vars))
        elif M:
            temp['separability'] = temp['M'][0].separability(vars)
        else:
            temp['separability'] = temp['L'][0].separability(vars)

class LinearBoundaryValueProblem(ProblemBase):
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
    terms must be linear in the specified variables and first-order in coupled
    derivatives, and the RHS must be independent of the specified variables.

        L.X = F

    """

    solver_class = solvers.LinearBoundaryValueSolver

    @CachedAttribute
    def namespace_additions(self):
        return {}

    def _check_conditions(self, temp):
        """Check object-form conditions."""
        #vars = [self.namespace[var] for var in self.variables]
        vars = self.variables
        self._require_independent(temp, 'RHS', vars)

    def _set_matrix_expressions(self, temp):
        """Set expressions for building solver."""
        #vars = [self.namespace[var] for var in self.variables]
        vars = self.variables
        temp['L'] = self._prep_linear_form(temp['LHS'], vars, name='L')
        temp['F'] = temp['RHS']
        temp['separability'] = temp['L'][0].separability(vars)


class NonlinearBoundaryValueProblem(ProblemBase):
    """
    Class for nonlinear boundary value problems.

    Parameters
    ----------
    domain : domain object
        Problem domain
    variables : list of str
        List of variable names, e.g. ['u', 'v', 'w']

    Notes
    -----
    This class supports nonlinear boundary value problems.  The LHS terms must
    be linear in the specified variables and first-order in coupled derivatives.

        L.X = F(X)

    The problem is reduced into a linear BVP for an update to the solution
    using the Newton-Kantorovich method and symbolically-computed Frechet
    derivatives of the RHS.

        L.(X0 + X1) = F(X0) + dF(X0).X1
        L.X1 - dF(X0).X1 = F(X0) - L.X0

    """

    solver_class = solvers.NonlinearBoundaryValueSolver

    @CachedAttribute
    def namespace_additions(self):
        """Build namespace for problem parsing."""
        additions = {}
        # Add variable perturbations
        for var in self.variables:
            pert = 'δ' + var.name
            additions[pert] = self.domain.new_field(name=pert)
            additions[pert].set_scales(1, keep_data=False)
        return additions

    def _check_conditions(self, temp):
        """Check object-form conditions."""
        pass

    def _set_matrix_expressions(self, temp):
        """Set expressions for building solver."""
        ep = field.Scalar(name='__epsilon__')
        #vars = [self.namespace[var] for var in self.variables]
        vars = self.variables
        perts = [self.namespace['δ'+var.name] for var in self.variables]
        # Build LHS operating on perturbations
        L = temp['LHS']
        for var, pert in zip(vars, perts):
            L = L.replace(var, pert)
        # Build Frechet derivative of RHS
        F = temp['RHS']
        dF = 0
        for var, pert in zip(vars, perts):
            dFi = F.replace(var, var + ep*pert)
            dFi = field.Operand.cast(dFi.sym_diff(ep), self.domain)
            dFi = dFi.replace(ep, 0)
            dF += dFi
        # Set expressions
        temp['L'] = self._prep_linear_form(L, perts, name='L')
        temp['dF'] = self._prep_linear_form(dF, perts, name='dF')
        temp['F-L'] = temp['RHS'] - temp['LHS']
        temp['separability'] = (temp['L'][0].separability(perts) &
                                temp['dF'][0].separability(perts))


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
    linear in the specified variables, first-order in coupled derivatives,
    and linear or independent of the specified eigenvalue.  The RHS must be zero.

        λM.X + L.X = 0

    """

    solver_class = solvers.EigenvalueSolver

    def __init__(self, domain, variables, eigenvalue, **kw):
        super().__init__(domain, variables, **kw)
        self.eigenvalue = eigenvalue

    @CachedAttribute
    def namespace_additions(self):
        """Build namespace for problem parsing."""
        additions = {}
        additions[self.eigenvalue] = self._ev = field.Scalar(name=self.eigenvalue)
        return additions

    def _check_conditions(self, temp):
        """Check object-form conditions."""
        self._require_first_order(temp, 'LHS', [self._ev])
        self._require_zero(temp, 'RHS')

    def _set_matrix_expressions(self, temp):
        """Set expressions for building solver."""
        M, L = temp['LHS'].split(self._ev)
        M = Operand.cast(M, self.domain)
        M = M.replace(self._ev, 1)
        #vars = [self.namespace[var] for var in self.variables]
        vars = self.variables
        temp['M'] = self._prep_linear_form(M, vars, name='M')
        temp['L'] = self._prep_linear_form(L, vars, name='L')
        temp['separability'] = (temp['M'][0].separability(vars) &
                                temp['L'][0].separability(Vars))


# Aliases
IVP = InitialValueProblem
LBVP = LinearBoundaryValueProblem
NLBVP = NonlinearBoundaryValueProblem
EVP = EigenvalueProblem

