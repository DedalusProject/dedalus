"""Classes for representing systems of equations."""

import numpy as np
from collections import ChainMap

from .field import Operand, Field
from . import arithmetic
from . import operators
from . import solvers
from ..tools import parsing
from ..tools.general import unify_attributes
from ..tools.exceptions import UnsupportedEquationError

import logging
logger = logging.getLogger(__name__.split('.')[-1])

# Public interface
__all__ = ['EigenvalueProblem',
           'LinearBoundaryValueProblem',
           'NonlinearBoundaryValueProblem',
           'InitialValueProblem',
           'EVP',
           'LBVP',
           'NLBVP',
           'IVP']


# Build basic parsing namespace
parseables = {}
parseables.update({name: getattr(operators, name) for name in operators.__all__})
parseables.update(operators.aliases)
parseables.update({name: getattr(arithmetic, name) for name in arithmetic.__all__})
parseables.update(arithmetic.aliases)


class ProblemBase:
    """Base class for all problem types."""

    def __init__(self, variables, namespace=None):
        self.variables = variables
        self.LHS_variables = variables
        self.dist = unify_attributes(variables, 'dist')
        self.equations = self.eqs = []
        # Build namespace via chainmap to react to upstream changes
        # Priority: local namespace, external namespace, parseables
        self.local_namespace = {}
        if namespace is None:
            self.namespace = ChainMap(self.local_namespace, parseables)
        else:
            self.namespace = ChainMap(self.local_namespace, namespace, parseables)

    @property
    def matrix_dependence(self):
        return np.logical_or.reduce([eqn['matrix_dependence'] for eqn in self.equations])

    @property
    def matrix_coupling(self):
        return np.logical_or.reduce([eqn['matrix_coupling'] for eqn in self.equations])

    @property
    def dtype(self):
        return np.result_type(*[eqn['dtype'] for eqn in self.equations])

    def add_equation(self, equation, condition="True"):
        """Add equation to problem."""
        logger.debug(f"Adding equation {len(self.eqs)}")
        # Split equation into LHS and RHS expressions
        if isinstance(equation, str):
            # Parse string-valued equations
            namespace = dict(self.namespace)
            LHS_str, RHS_str = parsing.split_equation(equation)
            LHS = eval(LHS_str, namespace)
            RHS = eval(RHS_str, namespace)
        else:
            # Split operator tuples
            LHS, RHS = equation
        logger.debug(f"  LHS: {LHS}")
        logger.debug(f"  RHS: {RHS}")
        logger.debug(f"  condition: {condition}")
        # Build basic equation dictionary
        # Note: domain determined after NCC reinitialization
        expr = LHS - RHS
        eqn = {'LHS': LHS,
               'RHS': RHS,
               'condition': condition,
               'tensorsig': expr.tensorsig,
               'dtype': expr.dtype}
        self._check_equation_conditions(eqn)
        self._build_matrix_expressions(eqn)
        self.equations.append(eqn)

    def build_solver(self, *args, **kw):
        """Build corresponding solver class."""
        return self.solver_class(self, *args, **kw)

    def _check_equation_conditions(self, eqn):
        """Check equation conditions."""
        raise NotImplementedError("Subclassses must implement.")

    def _build_matrix_expressions(self, eqn):
        """Build LHS matrix expressions."""
        raise NotImplementedError("Subclassses must implement.")

    @staticmethod
    def _check_domain_containment(subexpr, supexpr, subname, supname):
        """Require one domain contain another."""
        if np.any(np.logical_and(subexpr.domain.nonconstant, supexpr.domain.constant)):
            raise UnsupportedEquationError("{} domain cannot be larger than {} domain.".format(subname, supname))


class LinearBoundaryValueProblem(ProblemBase):
    """
    Linear boundary value problems.

    Parameters
    ----------
    variables : list of Field objects
        Problem variables to solve for.
    namespace : dict-like, optional
        Dictionary for namespace additions to use when parsing strings as equations
        (default: None). It is recommended to pass "locals()" from the user script.

    Notes
    -----
    This class supports linear boundary value problems of the form:
        L.X = F
    The LHS terms must be linear in the problem variables, and the RHS can be
    inhomogeneous but must be independent of the problem variables.

    Solution procedure:
        - Form L
        - Evaluate F
        - Solve X = L \ F
    """

    solver_class = solvers.LinearBoundaryValueSolver

    def _check_equation_conditions(self, eqn):
        """Check equation conditions."""
        # Cast LHS and RHS to operands
        LHS = Operand.cast(eqn['LHS'], self.dist, tensorsig=eqn['tensorsig'], dtype=eqn['dtype'])
        RHS = Operand.cast(eqn['RHS'], self.dist, tensorsig=eqn['tensorsig'], dtype=eqn['dtype'])
        # Check conditions
        LHS.require_linearity(*self.variables, allow_affine=False,
            self_name='LBVP LHS', vars_name='problem variables', error=UnsupportedEquationError)
        RHS.require_independent(*self.variables,
            self_name='LBVP RHS', vars_name='problem variables', error=UnsupportedEquationError)
        self._check_domain_containment(RHS, LHS, 'RHS', 'LHS')

    def _build_matrix_expressions(self, eqn):
        """Build LHS matrix expressions."""
        vars = self.variables
        dist = self.dist
        tensorsig = eqn['tensorsig']
        dtype = eqn['dtype']
        # Extract matrix expressions
        L = eqn['LHS']
        F = eqn['RHS']
        # Reinitialize and prep NCCs
        L = L.reinitialize(ncc=True, ncc_vars=vars)
        L.prep_nccs(vars=vars)
        # Convert to same domain
        domain = (L - F).domain
        L = operators.convert(L, domain.bases)
        if F:
            # Cast to match LHS
            F = Operand.cast(F, dist, tensorsig=tensorsig, dtype=dtype)
            F = operators.convert(F, domain.bases)
        else:
            # Allocate zero field
            F = Field(dist=dist, bases=domain.bases, tensorsig=tensorsig, dtype=dtype)
            F['c'] = 0
        # Save expressions and metadata
        eqn['L'] = L
        eqn['F'] = F
        eqn['domain'] = domain
        eqn['matrix_dependence'] = L.matrix_dependence(*vars)
        eqn['matrix_coupling'] = L.matrix_coupling(*vars)
        # Debug logging
        logger.debug(f"  L: {L}")
        logger.debug(f"  F: {F}")


class NonlinearBoundaryValueProblem(ProblemBase):
    """
    Nonlinear boundary value problems.

    Parameters
    ----------
    variables : list of Field objects
        Problem variables to solve for.
    namespace : dict-like, optional
        Dictionary for namespace additions to use when parsing strings as equations
        (default: None). It is recommended to pass "locals()" from the user script.

    Notes
    -----
    This class supports nonlinear boundary value problems of the form:
        F(X) = G(X)
    which are recombined to form the root-finding problem:
        H(X) = F(X) - G(X) = 0

    The problem is reduced into a linear BVP for an update to the solution
    using the Newton-Kantorovich method and the symbolically-computed Frechet
    differential of the equation:
        H(X[n+1]) = 0
        H(X[n] + dX) = 0
        H(X[n]) + dH(X[n]).dX = 0
        dH(X[n]).dX = - H(X[n])

    Iteration procedure:
        - Form dH(X[n])
        - Evaluate H(X[n])
        - Solve dX = - dH(X[n]) \ H(X[n])
        - Update X[n+1] = X[n] + dX
    """

    solver_class = solvers.NonlinearBoundaryValueSolver

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # Build perturbation variables
        self.perturbations = [var.copy() for var in self.variables]
        for pert, var in zip(self.perturbations, self.variables):
            pert['c'] = 0
            pert.name = 'δ'+var.name
        self.LHS_variables = self.perturbations

    def _check_equation_conditions(self, eqn):
        """Check equation conditions."""
        # No conditions
        pass

    def _build_matrix_expressions(self, eqn):
        """Build LHS matrix expressions."""
        vars = self.variables
        perts = self.perturbations
        # Extract matrix expressions
        H = eqn['LHS'] - eqn['RHS']
        dH = H.frechet_differential(vars, perts)
        # Reinitialize and prep NCCs
        dH = dH.reinitialize(ncc=True, ncc_vars=perts)
        dH.prep_nccs(vars=perts)
        # Convert to same domain
        domain = (dH + H).domain
        H = operators.convert(H, domain.bases)
        dH = operators.convert(dH, domain.bases)
        # Save expressions and metadata
        eqn['H'] = H
        eqn['dH'] = dH
        eqn['domain'] = domain
        eqn['matrix_dependence'] = dH.matrix_dependence(*perts)
        eqn['matrix_coupling'] = dH.matrix_coupling(*perts)
        # Debug logging
        logger.debug(f"   H: {H}")
        logger.debug(f"  dH: {dH}")


class InitialValueProblem(ProblemBase):
    """
    Initial value problems.

    Parameters
    ----------
    variables : list of Field objects
        Problem variables to solve for.
    time : str or Field object, optional
        Name (if str) or field for time variable (default: 't').
    namespace : dict-like, optional
        Dictionary for namespace additions to use when parsing strings as equations
        (default: None). It is recommended to pass "locals()" from the user script.

    Notes
    -----
    This class supports non-linear initial value problems of the form:
        M.dt(X) + L.X = F(X, t)
    The LHS terms must be linear in the problem variables, time independent,
    and first-order in time derivatives. The RHS terms must not contain any
    explicit time derivatives.
    """

    solver_class = solvers.InitialValueSolver

    def __init__(self, variables, time='t', **kw):
        super().__init__(variables, **kw)
        if isinstance(time, str):
            self.time = Field(dist=self.dist, name=time, dtype=np.float64)
        elif isinstance(time, Field):
            if any(time.domain.nonconstant):
                raise ValueError("Time field cannot have any bases.")
            self.time = time
        else:
            raise ValueError("Time must be str or Field object.")

    def _check_equation_conditions(self, eqn):
        """Check equation conditions."""
        # Cast LHS and RHS to operands
        LHS = Operand.cast(eqn['LHS'], self.dist, tensorsig=eqn['tensorsig'], dtype=eqn['dtype'])
        RHS = Operand.cast(eqn['RHS'], self.dist, tensorsig=eqn['tensorsig'], dtype=eqn['dtype'])
        # Check conditions
        LHS.require_linearity(*self.variables, allow_affine=False,
            self_name='IVP LHS', vars_name='problem variables', error=UnsupportedEquationError)
        LHS.require_independent(self.time,
            self_name='IVP LHS', vars_name='time', error=UnsupportedEquationError)
        LHS.require_first_order(operators.TimeDerivative,
            self_name='IVP LHS', ops_name='time derivatives', error=UnsupportedEquationError)
        RHS.require_independent(operators.TimeDerivative,
            self_name='IVP RHS', vars_name='time derivatives', error=UnsupportedEquationError)
        self._check_domain_containment(RHS, LHS, 'RHS', 'LHS')

    def _build_matrix_expressions(self, eqn):
        """Build LHS matrix expressions."""
        vars = self.variables
        dist = self.dist
        tensorsig = eqn['tensorsig']
        dtype = eqn['dtype']
        # Extract matrix expressions
        M, L = eqn['LHS'].split(operators.TimeDerivative)
        F = eqn['RHS']
        # Drop time derivatives
        if M:
            M = M.replace(operators.TimeDerivative, lambda x: x)
        # Reinitialize and prep NCCs
        if M:
            M = M.reinitialize(ncc=True, ncc_vars=vars)
            M.prep_nccs(vars=vars)
        if L:
            L = L.reinitialize(ncc=True, ncc_vars=vars)
            L.prep_nccs(vars=vars)
        # Convert to same domain
        domain = (M + L - F).domain
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
        # Save expressions and metadata
        eqn['M'] = M
        eqn['L'] = L
        eqn['F'] = F
        eqn['domain'] = domain
        eqn['matrix_dependence'] = (M + L).matrix_dependence(*vars)
        eqn['matrix_coupling'] = (M + L).matrix_coupling(*vars)
        # Debug logging
        logger.debug(f"  M: {M}")
        logger.debug(f"  L: {L}")
        logger.debug(f"  F: {F}")


class EigenvalueProblem(ProblemBase):
    """
    Linear eigenvalue problems.

    Parameters
    ----------
    variables : list of Field objects
        Problem variables to solve for.
    eigenvalue : Field object
        Field object representing the eigenvalue.
    namespace : dict-like, optional
        Dictionary for namespace additions to use when parsing strings as equations
        (default: None). It is recommended to pass "locals()" from the user script.

    Notes
    -----
    This class supports linear eigenvalue problems of the form:
        s*M.X + L.X = 0
    The LHS terms must be linear in the specified variables and affine in the eigenvalue.
    The RHS must be zero.
    """

    solver_class = solvers.EigenvalueSolver

    def __init__(self, variables, eigenvalue, **kw):
        super().__init__(variables, **kw)
        if any(eigenvalue.domain.nonconstant):
            raise ValueError("Eigenvalue field cannot have any bases.")
        self.eigenvalue = eigenvalue

    def _check_equation_conditions(self, eqn):
        """Check equation conditions."""
        # Cast LHS to operand
        LHS = Operand.cast(eqn['LHS'], self.dist, tensorsig=eqn['tensorsig'], dtype=eqn['dtype'])
        # Check conditions
        LHS.require_linearity(*self.variables, allow_affine=False,
            self_name='EVP LHS', vars_name='problem variables', error=UnsupportedEquationError)
        LHS.require_linearity(self.eigenvalue, allow_affine=True,
            self_name='EVP LHS', vars_name='the eigenvalue', error=UnsupportedEquationError)
        if eqn['RHS'] != 0:
            raise UnsupportedEquationError("EVP RHS must be identically zero.")

    def _build_matrix_expressions(self, eqn):
        """Build LHS matrix expressions."""
        vars = self.variables
        # Extract matrix expressions
        M, L = eqn['LHS'].split(self.eigenvalue)
        # Drop eigenvalue
        if M:
            M = M.replace(self.eigenvalue, 1)
        # Reinitialize and prep NCCs
        if M:
            M = M.reinitialize(ncc=True, ncc_vars=vars)
            M.prep_nccs(vars=vars)
        if L:
            L = L.reinitialize(ncc=True, ncc_vars=vars)
            L.prep_nccs(vars=vars)
        # Convert to same domain
        domain = (M + L).domain
        if M:
            M = operators.convert(M, domain.bases)
        if L:
            L = operators.convert(L, domain.bases)
        # Save expressions and metadata
        eqn['M'] = M
        eqn['L'] = L
        eqn['domain'] = domain
        eqn['matrix_dependence'] = (M + L).matrix_dependence(*vars)
        eqn['matrix_coupling'] = (M + L).matrix_coupling(*vars)
        # Debug logging
        logger.debug(f"  M: {M}")
        logger.debug(f"  L: {L}")


# Aliases
IVP = InitialValueProblem
LBVP = LinearBoundaryValueProblem
NLBVP = NonlinearBoundaryValueProblem
EVP = EigenvalueProblem

