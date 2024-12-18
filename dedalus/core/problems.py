"""Classes for representing systems of equations."""

import numpy as np
from collections import ChainMap

from .field import Operand, Field, LockedField
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
parseables.update({name: getattr(arithmetic, name) for name in arithmetic.__all__})


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
        for var in variables:
            if var.name:
                self.local_namespace[var.name] = var
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
        eqn = {'eqn': expr,
               'LHS': LHS,
               'RHS': RHS,
               'condition': condition,
               'tensorsig': expr.tensorsig,
               'dtype': expr.dtype,
               'valid_modes': expr.valid_modes.copy()}
        self._check_equation_conditions(eqn)
        self._build_matrix_expressions(eqn)
        self.equations.append(eqn)
        return eqn

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
        G(X) = H(X)
    which are recombined to form the root-finding problem:
        F(X) = G(X) - H(X) = 0

    The problem is reduced into a linear BVP for an update to the solution
    using the Newton-Kantorovich method and the symbolically-computed Frechet
    differential of the equation:
        F(X[n+1]) = 0
        F(X[n] + dX) = 0
        F(X[n]) + dF(X[n]).dX = 0
        dF(X[n]).dX = - F(X[n])

    Iteration procedure:
        - Form dF(X[n])
        - Evaluate F(X[n])
        - Solve dX = - dF(X[n]) \ F(X[n])
        - Update X[n+1] = X[n] + dX
    """

    solver_class = solvers.NonlinearBoundaryValueSolver

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # Build perturbation variables
        self.perturbations = [var.copy() for var in self.variables]
        for pert, var in zip(self.perturbations, self.variables):
            pert.preset_scales(1)
            pert['c'] = 0
            if var.name:
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
        F = eqn['LHS'] - eqn['RHS']
        dF = F.frechet_differential(vars, perts)
        # Remove any field locks
        dF = dF.replace(operators.Lock, lambda x: x)
        for field in dF.atoms(LockedField):
            dF = dF.replace(field, field.unlock())
        # Reinitialize and prep NCCs
        dF = dF.reinitialize(ncc=True, ncc_vars=perts)
        dF.prep_nccs(vars=perts)
        # Convert to same domain
        domain = (dF + F).domain
        F = operators.convert(F, domain.bases)
        dF = operators.convert(dF, domain.bases)
        # Save expressions and metadata
        eqn['F'] = F
        eqn['dF'] = dF
        eqn['domain'] = domain
        eqn['matrix_dependence'] = dF.matrix_dependence(*perts)
        eqn['matrix_coupling'] = dF.matrix_coupling(*perts)
        # Debug logging
        logger.debug(f"  F: {F}")
        logger.debug(f"  dF: {dF}")


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

    def build_EVP(self, eigenvalue=None, backgrounds=None, perturbations=None, **kw):
        """
        Create an eigenvalue problem from an initial value problem.

        Parameters
        ----------
        eigenvalue : Field, optional
            Eigenvalue field.
        backgrounds : list of Fields, optional
            Background fields for linearizing the RHS. Default: the IVP variables.
        perturbations : list of Fields, optional
            Perturbation fields for the EVP. Default: copies of IVP variables.

        Notes
        -----
        This method converts time-independent IVP equations of the form
            M.dt(X) + L.X = F(X)
        to EVP equations as
            λ*M.X1 + L.X1 - F'(X0).X1 = 0.
        If backgrounds (X0) are not specified, the IVP variables (X) are used.
        """
        # Create eigenvalue problem for perturbations
        variables = self.variables
        if eigenvalue is None:
            eigenvalue = self.dist.Field(name='λ')
        if perturbations is None:
            perturbations = [var.copy() for var in variables]
            for pert, var in zip(perturbations, variables):
                if var.name:
                    pert.name = 'δ'+var.name
        for pert, var in zip(perturbations, variables):
            pert.valid_modes[:] = var.valid_modes
        EVP = EigenvalueProblem(perturbations, eigenvalue, **kw)
        # Convert equations from IVP
        for eqn in self.equations:
            # Extract IVP expressions
            M, L = eqn['LHS'].split(operators.TimeDerivative)
            F = eqn['RHS']
            # Convert M@dt(X) to λ*M@Y
            if M:
                M = M.replace(operators.TimeDerivative, lambda x: eigenvalue*x)
                for var, pert in zip(variables, perturbations):
                    M = M.replace(var, pert)
            # Convert L@X to L@Y
            if L:
                for var, pert in zip(variables, perturbations):
                    L = L.replace(var, pert)
            # Take Frechet differential of F(X)
            if F:
                if F.has(self.time):
                    raise UnsupportedEquationError("Cannot convert time-dependent IVP to EVP.")
                dF = F.frechet_differential(variables=variables, perturbations=perturbations, backgrounds=backgrounds)
            else:
                dF = 0
            # Add linearized equation and copy valid modes
            evp_eqn = EVP.add_equation((M + L - dF, 0))
            evp_eqn['valid_modes'][:] = eqn['valid_modes']
        # Add backgrounds to EVP namespace
        for var in backgrounds:
            EVP.local_namespace[var.name] = var
        return EVP


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
        λ*M.X + L.X = 0
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

