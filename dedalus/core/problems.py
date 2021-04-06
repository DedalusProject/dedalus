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
from . import arithmetic
from . import operators
from . import solvers
from ..tools import parsing
from ..tools.cache import CachedAttribute
from ..tools.general import unify_attributes
from ..tools.exceptions import SymbolicParsingError
from ..tools.exceptions import UnsupportedEquationError

from ..tools.config import config

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


# class Equation:

#     def __init__(self, eq_string, condition):
#         self.eq_string = eq_string
#         self.condition = condition

#         self.LHS_string
#         self.RHS_string
#         self.LHS_object
#         self.RHS_object


    # @property
    # def bases(self):
    #     return self.LHS.bases

    # @property
    # def subdomain(self):
    #     return self.LHS.subdomain

    # @property
    # def separability(self):
    #     return self.LHS.separability

    # def check_condition(self, group_dict):
    #     pass



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

    def __init__(self, variables, ncc_cutoff=1e-10, max_ncc_terms=None):
        self.variables = variables
        self.LHS_variables = variables
        self.dist = unify_attributes(variables, 'dist')
        self.equations = self.eqs = []
        self.parameters = OrderedDict()
        self.substitutions = OrderedDict()
        #self.op_kw = {'cutoff': ncc_cutoff}

    def add_equation(self, equation, condition="True"):
        """Add equation to problem."""
        logger.debug("Parsing Eqn {}".format(len(self.eqs)))
        # Parse string-valued equations
        if isinstance(equation, str):
            raise NotImplementedError()
            # self._store_string_forms(eqn, equation, condition)
            # self._build_object_forms(eqn)
        # Determine equation domain
        LHS, RHS = equation
        expr = LHS - RHS
        # Build equation dictionary
        eqn = {'LHS': LHS,
               'RHS': RHS,
               'tensorsig': expr.tensorsig,
               'dtype': expr.dtype,
               'condition': condition}
        self._build_matrix_expressions(eqn)
        # Store equation dictionary
        self.equations.append(eqn)

    def _store_string_forms(self, eqn, equation, condition):
        """Split and store equation and condition strings."""
        eqn['equation_str'] = equation
        eqn['condition_str'] = condition
        eqn['LHS_str'], eqn['RHS_str'] = parsing.split_equation(equation)
        # Debug logging
        logger.debug("  Condition: {}".format(condition))
        logger.debug("  LHS string form: {}".format(eqn['LHS_str']))
        logger.debug("  RHS string form: {}".format(eqn['RHS_str']))

    def _build_object_forms(self, eqn):
        """Parse LHS/RHS strings to object forms."""
        # Parse LHS/RHS strings to object expressions
        eqn['LHS'] = self._parse(eqn['LHS_str'])
        eqn['RHS'] = self._parse(eqn['RHS_str'])
        # Determine equation bases using addition operator
        combo = eqn['LHS'] - eqn['RHS']
        eqn['bases'] = combo.bases
        eqn['subdomain'] = combo.subdomain
        # Debug logging
        logger.debug("  LHS object form: {}".format(eqn['LHS']))
        logger.debug("  RHS object form: {}".format(eqn['RHS']))

    def _parse(self, expr_str):
        """Parse expression using problem namespace."""
        # ENHANCEMENT: Better security / sanitization?
        expr = eval(expr_str, self.namespace)
        return operators.Cast(expr, self.domain)

    def _build_matrix_expressions(self, eqn):
        """Build LHS matrix expressions and check equation conditions."""
        raise NotImplementedError()

    @CachedAttribute
    def namespace(self):
        """Build namespace for problem parsing."""
        namespace = Namespace()
        # Space-specific items
        for spaceset in self.domain.spaces:
            for space in spaceset:
                # Grid
                namespace[space.name] = space.grid_field(scales=None)
                # Operators
                namespace.update(space.operators)
        # Variables
        namespace.update({var.name: var for var in self.variables})
        # Parameters
        namespace.update(self.parameters)
        # Built-in functions
        namespace.update(operators.parseables)
        # Additions from derived classes
        namespace.update(self.namespace_additions)
        # Substitutions
        namespace.add_substitutions(self.substitutions)
        return namespace

    @CachedAttribute
    def namespace_additions(self):
        return {}

    @staticmethod
    def _check_basis_containment(eqn, superkey, subkey):
        """Require one subdomain contain another."""
        if not eqn[subkey].subdomain in eqn[superkey].subdomain:
            raise UnsupportedEquationError("{} subdomain must be in {} subdomain.".format(subkey, superkey))

    @staticmethod
    def _require_zero(eqn, key):
        """Require expression to be equal to zero."""
        if eqn[key] != 0:
            raise UnsupportedEquationError("{} must be zero.".format(key))

    @staticmethod
    def _require_independent(eqn, key, vars):
        """Require expression to be independent of some variables."""
        if eqn[key].has(*vars):
            names = [var.name for var in vars]
            raise UnsupportedEquationError("{} must be independent of {}.".format(key, names))

    # def _require_first_order(self, temp, key, vars):
    #     """Require expression to be zeroth or first order in some variables."""
    #     order = temp[key].order(*vars)
    #     if order > 1:
    #         names = [var.name for var in vars]
    #         raise UnsupportedEquationError("{} must be first-order in {}.".format(key, names))
    #     return order

    # def _prep_linear_form(self, expr, vars, name=''):
    #     """Convert an expression into suitable form for LHS operator conversion."""
    #     if expr:
    #         expr = Operand.cast(expr, self.domain)
    #         expr = expr.expand(*vars)
    #         expr = expr.canonical_linear_form(*vars)
    #         logger.debug('  {} linear form: {}'.format(name, str(expr)))
    #     return expr
    #     #return (expr, vars)

    def build_solver(self, *args, **kw):
        """Build corresponding solver class."""
        return self.solver_class(self, *args, **kw)

    # def separability(self):
    #     seps = [eqn['separability'] for eqn in self.equations]
    #     return reduce(np.logical_and, seps)

    # @property
    # def separable(self):
    #     return self.separability()

    # @property
    # def coupled(self):
    #     return np.invert(self.separable)

    @property
    def matrix_dependence(self):
        return np.logical_or.reduce([eqn['matrix_dependence'] for eqn in self.equations])

    @property
    def matrix_coupling(self):
        return np.logical_or.reduce([eqn['matrix_coupling'] for eqn in self.equations])

    @property
    def dtype(self):
        return np.result_type(*[eqn['dtype'] for eqn in self.equations])

class LinearBoundaryValueProblem(ProblemBase):
    """
    Class for linear boundary value problems.

    Parameters
    ----------
    domain : domain object
        Problem domain

    Notes
    -----
    This class supports linear boundary value problems.  The LHS terms must be
    linear in the problem variables, and the RHS can be inhomogeneous but must
    be independent of the problem variables.

        L.X = F
        Solve for X

    """

    solver_class = solvers.LinearBoundaryValueSolver

    def _build_matrix_expressions(self, eqn):
        """Build LHS matrix expressions and check equation conditions."""
        vars = self.variables
        dist = self.dist
        #domain = eqn['domain']
        tensorsig = eqn['tensorsig']
        dtype = eqn['dtype']
        # TO-DO: check these conditions
        # Equation conditions
        #self._check_basis_containment(eqn, 'LHS', 'RHS')
        #eqn['LHS'].require_linearity(*vars, name='LHS')
        #eqn['RHS'].require_independent(*vars, name='RHS')
        # Matrix expressions
        # Update domain after ncc reinitialization
        L = eqn['LHS']
        L = L.reinitialize(ncc=True, ncc_vars=vars)
        L.prep_nccs(vars=vars)
        F = eqn['RHS']
        domain = eqn['domain'] = (L - F).domain
        L = operators.convert(L, domain.bases)
        if F:
            # Cast to match LHS
            F = Operand.cast(F, dist, tensorsig=tensorsig, dtype=dtype)
            F = operators.convert(F, domain.bases)
        else:
            # Allocate zero field
            F = Field(dist=dist, bases=domain.bases, tensorsig=tensorsig, dtype=dtype)
            F['c'] = 0
        eqn['L'] = L
        eqn['F'] = F
        eqn['matrix_dependence'] = L.matrix_dependence(*vars)
        eqn['matrix_coupling'] = L.matrix_coupling(*vars)
        #eqn['separability'] = eqn['L'].separability(*vars)
        # Debug logging
        logger.debug('  {} linear form: {}'.format('L', eqn['L']))


class NonlinearBoundaryValueProblem(ProblemBase):
    """
    Class for nonlinear boundary value problems.

    Parameters
    ----------
    domain : domain object
        Problem domain

    Notes
    -----
    This class supports nonlinear boundary value problems.  The LHS and RHS
    terms are recombined to form a root-finding problem:

        F(X) = G(X)  -->  H(X) = F(X) - G(X) = 0

    The problem is reduced into a linear BVP for an update to the solution
    using the Newton-Kantorovich method and symbolically-computed Frechet
    derivatives of the equation.

        H(X0 + X1) = 0
        H(X0) + dH(X0).X1 = 0
        dH(X0).X1 = -H(X0)
        Solve for X1

    """

    solver_class = solvers.NonlinearBoundaryValueSolver

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.perturbations = [var.copy() for var in self.variables]
        for pert, var in zip(self.perturbations, self.variables):
            pert['c'] = 0
            pert.name = 'δ'+var.name
        self.LHS_variables = self.perturbations

    @CachedAttribute
    def namespace_additions(self):
        """Build namespace for problem parsing."""
        # Variable perturbations
        self.perturbations = [Field(bases=var.bases, name='δ'+var.name) for var in self.variables]
        return {pert.name: pert for pert in self.perturbations}

    def _build_matrix_expressions(self, eqn):
        """Build LHS matrix expressions and check equation conditions."""
        vars = self.variables
        perts = self.perturbations
        tensorsig = eqn['tensorsig']
        dtype = eqn['dtype']
        ep = Field(dist=self.dist, name='ep', dtype=dtype)
        # TO-DO: check conditions
        # Equation conditions

        # Combine LHS and RHS into single expression
        H = eqn['LHS'] - eqn['RHS']
        # Build Frechet derivative
        dH = 0
        for var, pert in zip(vars, perts):
            dHi = H.replace(var, var + ep*pert)
            dHi = dHi.sym_diff(ep)
            if dHi:
                dHi = Operand.cast(dHi, self.dist, tensorsig=tensorsig, dtype=dtype)
                dHi = dHi.replace(ep, 0)
                dH += dHi
        dH = dH.reinitialize(ncc=True, ncc_vars=perts)
        dH.prep_nccs(vars=perts)
        domain = eqn['domain'] = (dH+H).domain
        dH = operators.convert(dH, domain.bases)
        H = operators.convert(H, domain.bases)
        # Matrix expressions
        #eqn['dH'] = convert(dH.expand(*perts), eqn['bases'])
        eqn['dH'] = dH
        eqn['-H'] = -H
        eqn['matrix_dependence'] = dH.matrix_dependence(*perts)
        eqn['matrix_coupling'] = dH.matrix_coupling(*perts)
        # Debug logging
        logger.debug('  {} linear form: {}'.format('dH', eqn['dH']))




class InitialValueProblem(ProblemBase):
    """
    Class for non-linear initial value problems.

    Parameters
    ----------
    domain : domain object
        Problem domain
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

    def __init__(self, variables, time='t', **kw):
        super().__init__(variables, **kw)
        self.time = time
        self.dt = operators.TimeDerivative

    @CachedAttribute
    def namespace_additions(self):
        """Build namespace for problem parsing."""

        class dt(operators.TimeDerivative):
            name = 'd' + self.time

        additions = {}
        additions[self.time] = self._t = field.Field(name=self.time, domain=self.domain)
        additions[dt.name] = self._dt = dt
        return additions

    def _check_conditions(self, temp):
        """Check object-form conditions."""
        self._require_independent(temp, 'LHS', [self._t])
        self._require_first_order(temp, 'LHS', [self._dt])

    # def _set_matrix_expressions(self, temp):
    #     """Set expressions for building solver."""
    #     M, L = temp['LHS'].split(self._dt)
    #     if M:
    #         M = operators.cast(M, self.domain)
    #         M = M.replace(self._dt, lambda x: x)
    #     #vars = [self.namespace[var] for var in self.variables]
    #     vars = self.variables
    #     temp['M'] = self._prep_linear_form(M, vars, name='M')
    #     temp['L'] = self._prep_linear_form(L, vars, name='L')
    #     temp['F'] = temp['RHS']
    #     temp['separability'] = temp['LHS'].separability(vars)

    # def _build_local_matrices(self, temp):
    #     vars = self.variables
    #     if temp['M'] != 0:
    #         temp['M_op'] = temp['M'].operator_dict(vars, **self.op_kw)
    #     else:
    #         temp['M_op'] = {}
    #     if temp['L'] != 0:
    #         temp['L_op'] = temp['L'].operator_dict(vars, **self.op_kw)
    #     else:
    #         temp['L_op'] = {}

    def _build_matrix_expressions(self, eqn):
        """Build LHS matrix expressions and check equation conditions."""
        # NEW CHECK!! boulder
        vars = self.variables
        dist = self.dist
        #domain = eqn['domain']
        tensorsig = eqn['tensorsig']
        dtype = eqn['dtype']
        # Equation conditions
        #self._check_basis_containment(eqn, 'LHS', 'RHS')
        #eqn['LHS'].require_linearity(*vars, name='LHS')
        #eqn['LHS'].require_linearity(self._dt, name='LHS')
        #eqn['LHS'].require_independent(self._t, name='LHS')
        # Matrix expressions
        # Split LHS into M and L
        M, L = eqn['LHS'].split(self.dt)
        # Drop time derivatives
        if M:
            M = M.replace(self.dt, lambda x:x)
        # Update domain after ncc reinitialization
        if M:
            M = M.reinitialize(ncc=True, ncc_vars=vars)
            M.prep_nccs(vars=vars)
        if L:
            L = L.reinitialize(ncc=True, ncc_vars=vars)
            L.prep_nccs(vars=vars)
        F = eqn['RHS']
        domain = eqn['domain'] = (M + L - F).domain
        # Convert to equation bases and store
        if M:
            M = operators.convert(M, domain.bases)
        if L:
            L = operators.convert(L, domain.bases)
        if F:
            # Cast to match LHS
            F = Operand.cast(F, dist, tensorsig=tensorsig, dtype=dtype)
            F = operators.convert(F, domain.bases)
        else:
            # Allocate zero field
            F = Field(dist=dist, bases=domain.bases, tensorsig=tensorsig, dtype=dtype)
            F['c'] = 0
        eqn['M'] = M
        eqn['L'] = L
        eqn['F'] = F
        # M = operators.Cast(M, self.domain)
        # M = M.expand(*vars)
        # M = operators.Cast(M, self.domain)
        # L = operators.Cast(L, self.domain)
        # L = L.expand(*vars)
        # L = operators.Cast(L, self.domain)
        # eqn['M'] = operators.convert(M, eqn['bases'])
        # eqn['L'] = operators.convert(L, eqn['bases'])
        # eqn['F'] = operators.convert(eqn['RHS'], eqn['bases'])
        eqn['matrix_dependence'] = (M + L).matrix_dependence(*vars)
        eqn['matrix_coupling'] = (M + L).matrix_coupling(*vars)
        # # Debug logging
        # logger.debug('  {} linear form: {}'.format('L', eqn['L']))



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

    def __init__(self, variables, eigenvalue, **kw):
        super().__init__(variables, **kw)
        self.eigenvalue = eigenvalue

    # @CachedAttribute
    # def namespace_additions(self):
    #     """Build namespace for problem parsing."""
    #     additions = {}
    #     additions[self.eigenvalue] = self._ev = field.Field(name=self.eigenvalue, domain=self.domain)
    #     return additions

    # def _check_conditions(self, temp):
    #     """Check object-form conditions."""
    #     self._require_first_order(temp, 'LHS', [self._ev])
    #     #self._require_zero(temp, 'RHS')

    def _set_matrix_expressions(self, temp):
        """Set expressions for building solver."""
        M, L = temp['LHS'].split(self._ev)
        if M:
            M = Operand.cast(M, self.domain)
            M = M.replace(self._ev, 1)
        #vars = [self.namespace[var] for var in self.variables]
        vars = self.variables
        temp['M'] = self._prep_linear_form(M, vars, name='M')
        temp['L'] = self._prep_linear_form(L, vars, name='L')
        temp['separability'] = temp['LHS'].separability(vars)

    def _build_matrix_expressions(self, eqn):
        """Build LHS matrix expressions and check equation conditions."""
        # NEW CHECK!! boulder
        vars = self.variables
        dist = self.dist
        #domain = eqn['domain']
        tensorsig = eqn['tensorsig']
        dtype = eqn['dtype']
        # Equation conditions
        #self._check_basis_containment(eqn, 'LHS', 'RHS')
        #eqn['LHS'].require_linearity(*vars, name='LHS')
        #eqn['LHS'].require_linearity(self._dt, name='LHS')
        #eqn['LHS'].require_independent(self._t, name='LHS')
        # Matrix expressions
        # Split LHS into M and L
        M, L = eqn['LHS'].split(self.eigenvalue)
        # Drop time derivatives
        if M:
            # TODO check that M is linear in eigenvalue
            M = M.replace(self.eigenvalue, 1)
        # Update domain after ncc reinitialization
        if M:
            M = M.reinitialize(ncc=True, ncc_vars=vars)
            M.prep_nccs(vars=vars)
        if L:
            L = L.reinitialize(ncc=True, ncc_vars=vars)
            L.prep_nccs(vars=vars)
        F = eqn['RHS']
        domain = eqn['domain'] = (M + L - F).domain
        # Convert to equation bases and store
        if M:
            M = operators.convert(M, domain.bases)
        if L:
            L = operators.convert(L, domain.bases)
        if F:
            # Cast to match LHS
            F = Operand.cast(F, dist, tensorsig=tensorsig, dtype=dtype)
            F = operators.convert(F, domain.bases)
        else:
            # Allocate zero field
            F = Field(dist=dist, bases=domain.bases, tensorsig=tensorsig, dtype=dtype)
            F['c'] = 0
        eqn['M'] = M
        eqn['L'] = L
        eqn['F'] = F
        # M = operators.Cast(M, self.domain)
        # M = M.expand(*vars)
        # M = operators.Cast(M, self.domain)
        # L = operators.Cast(L, self.domain)
        # L = L.expand(*vars)
        # L = operators.Cast(L, self.domain)
        # eqn['M'] = operators.convert(M, eqn['bases'])
        # eqn['L'] = operators.convert(L, eqn['bases'])
        # eqn['F'] = operators.convert(eqn['RHS'], eqn['bases'])
        eqn['matrix_dependence'] = (M + L).matrix_dependence(*vars)
        eqn['matrix_coupling'] = (M + L).matrix_coupling(*vars)
        # # Debug logging
        # logger.debug('  {} linear form: {}'.format('L', eqn['L']))



# Aliases
IVP = InitialValueProblem
LBVP = LinearBoundaryValueProblem
NLBVP = NonlinearBoundaryValueProblem
EVP = EigenvalueProblem
