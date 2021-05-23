"""
Classes for representing systems of equations.

"""

from collections import OrderedDict
import numpy as np
from mpi4py import MPI

from .metadata import MultiDict, Metadata
from . import field
from .field import Operand
from . import future
from . import operators
from . import solvers
from ..tools import parsing
from ..tools.cache import CachedAttribute
from ..tools.exceptions import SymbolicParsingError
from ..tools.exceptions import UnsupportedEquationError

from ..tools.config import config
BC_TOP = lambda: config['matrix construction'].getboolean('BC_TOP')
INTERLEAVE_SUBBASES = lambda: config['matrix construction'].getboolean('INTERLEAVE_SUBBASES')
STORE_EXPANDED_MATRICES = lambda: config['matrix construction'].getboolean('STORE_EXPANDED_MATRICES')

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class Namespace(OrderedDict):
    """Class ensuring a conflict-free namespace for parsing."""

    __slots__ = 'allow_overwrites'

    def __init__(self):
        super().__init__()
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
            head, sub_str = parsing.lambdify_functions(call, result)
            # Evaluate in current namespace
            self[head] = sub = eval(sub_str, self)
            # Enable output caching for expression substitutions
            # Avoids some deadlocking issues when evaluating redundant subtrees
            if isinstance(sub, future.FutureField):
                sub.store_last = True


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
        Mode amplitude cutoff for LHS NCC expansions (default: 1e-6)
    max_ncc_terms : int, optional
        Maximum terms to include in LHS NCC expansions (default: None (no limit))
    entry_cutoff : float, optional
        Matrix entry cutoff to avoid fill-in from cancellation errors (default: 1e-12)

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

    def __init__(self, domain, variables, ncc_cutoff=1e-6, max_ncc_terms=None, entry_cutoff=1e-12):
        self.domain = domain
        self.variables = variables
        self.nvars = len(variables)
        self.meta = MultiDict({var: Metadata(domain) for var in variables})
        self.equations = self.eqs = []
        self.parameters = OrderedDict()
        self.substitutions = OrderedDict()
        self.ncc_kw = {'cutoff': ncc_cutoff, 'max_terms': max_ncc_terms}
        self.entry_cutoff = entry_cutoff
        self.coupled = domain.bases[-1].coupled
        # Matrix construction config options
        self.BC_TOP = BC_TOP()
        self.INTERLEAVE_SUBBASES = INTERLEAVE_SUBBASES()
        self.STORE_EXPANDED_MATRICES = STORE_EXPANDED_MATRICES()

    @property
    def nvars_const(self):
        return len([var for var in self.variables if self.meta[var][-1]['constant']])

    @property
    def nvars_nonconst(self):
        return self.nvars - self.nvars_const

    def add_equation(self, equation, condition="True", tau=None):
        """Add equation to problem."""
        logger.debug("Parsing Eqn {}".format(len(self.eqs)))
        temp = {}
        temp['tau'] = tau
        self._build_basic_dictionary(temp, equation, condition)
        self._build_object_forms(temp)
        self._check_eqn_conditions(temp)
        self._set_matrix_expressions(temp)
        self.eqs.append(temp)

    def add_bc(self, *args, **kw):
        """Add boundary condition to problem."""
        # Deprecated. Pass to add_equation.
        return self.add_equation(*args, **kw)

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
        temp['LHS'] = field.Operand.parse(temp['raw_LHS'], self.namespace, self.domain)
        temp['RHS'] = future.FutureField.parse(temp['raw_RHS'], self.namespace, self.domain)
        temp['constant'] = temp['LHS'].meta[-1]['constant']
        logger.debug("  LHS object form: {}".format(temp['LHS']))
        logger.debug("  RHS object form: {}".format(temp['RHS']))

    @CachedAttribute
    def namespace(self):
        """Build namespace for problem parsing."""
        namespace = Namespace()
        # Basis-specific items
        for axis, basis in enumerate(self.domain.bases):
            # Grids
            namespace[basis.name] = basis.grid_array_object(self.domain, axis)
            namespace[f's{basis.name}'] = basis.grid_spacing_object(self.domain, axis)
            # Basis operators
            for op in basis.operators:
                namespace[op.name] = op
        # Fields
        for var in self.variables:
            namespace[var] = self.domain.new_field(name=var)
            namespace[var].meta = self.meta[var]
            namespace[var].set_scales(1, keep_data=False)
        # Parameters
        for name, param in self.parameters.items():
            # Cast parameters to operands
            casted_param = field.Operand.cast(param)
            casted_param.name = name
            namespace[name] = casted_param
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
        self._check_differential_order(temp)
        self._check_meta_consistency(temp['LHS'], temp['RHS'])

    def _check_differential_order(self, temp):
        """Require LHS To be first order in coupled derivatives."""
        coupled_diffs = [basis.Differentiate for basis in self.domain.bases if not basis.separable]
        order = self._require_first_order_comp(temp, 'LHS', coupled_diffs)
        temp['differential'] = int(order)
        if temp['constant']:
            temp['tau'] = 0
        elif temp['tau'] is None:
            temp['tau'] = temp['differential']
        else:
            temp['tau'] = int(temp['tau'])

    def _check_if_zero(self, expr):
        ''' Checks if the expression is equal to zero, within a tolerance '''
        if expr == 0: #If exactly zero, don't bother with anything else
            return True

        #Evaluate the expression and pick out the absolute max value.
        # Compare this value with the absolute max of the NCCs that went into the calculation.
        # IF max_evaluated / max_parameters < tol, then it's homogeneous.
        evaluated_expr = expr.evaluate()
        max_val = np.max(np.abs(evaluated_expr['g']))
        max_param = self._find_max_param(params = expr.atoms())

        #Compare the max value across ALL processors to make sure that everyone agrees.
        global_max = self.domain.dist.comm.allreduce(max_val, op=MPI.MAX)
        homogeneous = (global_max <= max_param*self.tol)
        global_homogeneous = self.domain.dist.comm.allreduce(homogeneous, op=MPI.LAND)

        if not global_homogeneous:
            logger.info(str(expr) + ' is not homogeneous; '+\
                        'max_val = {:.3e} (above tolerance ({:.1e}) range of max parameter value {:.3e}).'.format(max_val, self.tol, max_param))
            logger.info('You may need to adjust your resolution or re-examine your equations.')
            return False
        else:
            if max_val != 0:
                logger.info('WARNING: ' + str(expr) + ' will be considered homogeneous '+\
                            '(Max value: {:.3e}; below tolerance ({:.1e}) of max param: {:.3e}). '.format(max_val, self.tol, max_param))
            return True

    def _check_meta_consistency(self, LHS, RHS):
        """Check LHS and RHS metadata for compatability."""
        self._check_meta_constant(LHS, RHS)
        self._check_meta_parity(LHS, RHS)
        self._check_meta_envelope(LHS, RHS)

    def _check_meta_constant(self, LHS, RHS):
        """Check that RHS is constant if LHS is consant."""
        for axis in range(self.domain.dim):
            if LHS.meta[axis]['constant']:
                if not RHS.meta[axis]['constant']:
                    raise SymbolicParsingError("LHS is constant but RHS is nonconstant along axis {}.".format(axis))

    def _check_meta_parity(self, LHS, RHS):
        """Check that LHS parity matches RHS parity if RHS is nonzero."""
        for axis in range(self.domain.dim):
            # Skip if not a parity axis
            if "parity" not in LHS.meta[axis]:
                continue
            if RHS != 0:
                if LHS.meta[axis]['parity'] != RHS.meta[axis]['parity']:
                    raise SymbolicParsingError("LHS and RHS parities along axis {} do not match.".format(axis))

    def _check_meta_envelope(self, LHS, RHS):
        """Check that LHS envelope matches RHS envelope if RHS is nonzero."""
        for axis in range(self.domain.dim):
            # Skip if not an envelope axis
            if "envelope" not in LHS.meta[axis]:
                continue
            if RHS != 0:
                if LHS.meta[axis]['envelope'] != RHS.meta[axis]['envelope']:
                    raise SymbolicParsingError("LHS and RHS envelopes along axis {} do not match.".format(axis))

    def _find_max_param(self, params):
        """Finds the maximum value of the specified parameters"""
        max_val = 0
        for param in params:
            if type(param) == field.Scalar:
                if np.abs(param.value) > max_val:
                    max_val = np.abs(param.value)
            elif np.max(np.abs(param['g'])) > max_val:
                max_val = np.max(np.abs(param['g']))
        return max_val

    def _require_homogeneous(self, temp, key, vars):
        """Require expression to be homogeneous with some variables set to zero."""
        expr = temp[key]
        for var in vars:
            if expr != 0:
                expr = expr.replace(var, 0)
        if not self._check_if_zero(expr):
            raise UnsupportedEquationError("{} must be homogeneous.".format(key))

    def _require_independent(self, temp, key, vars):
        """Require expression to be independent of some variables."""
        if temp[key].has(*vars):
            names = [var.name for var in vars]
            raise UnsupportedEquationError("{} must be independent of {}.".format(key, names))

    def _require_first_order_comp(self, temp, key, ops):
        """Require expression to be zeroth or first order compositionally in some operators."""
        vars = [self.namespace[var] for var in self.variables]
        order = temp[key].comp_order(ops, vars)
        if order > 1:
            names = [op.name for op in ops]
            raise UnsupportedEquationError("{} must be first-order in {}.".format(key, names))
        return order

    def _require_first_order_mul(self, temp, key, fields):
        """Require expression to be zeroth or first order multiplicatively in some fields."""
        order = temp[key].mul_order(fields)
        if order > 1:
            names = [field.name for field in fields]
            raise UnsupportedEquationError("{} must be first-order in {}.".format(key, names))
        return order

    def _prep_linear_form(self, expr, vars, name=''):
        """Convert an expression into suitable form for LHS operator conversion."""
        expr = Operand.cast(expr)
        expr = expr.expand(*vars)
        expr = expr.canonical_linear_form(*vars)
        logger.debug('  {} linear form: {}'.format(name, str(expr)))
        return (expr, vars)

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
    terms must be linear in the specified variables, first-order in coupled
    derivatives, first-order in time derivatives, and contain no explicit
    time dependence.

        M.dt(X) + L.X = F(X, t)

    """

    solver_class = solvers.InitialValueSolver

    def __init__(self, domain, variables, time='t', **kw):
        super().__init__(domain, variables, **kw)
        self.time = time

    @CachedAttribute
    def namespace_additions(self):
        """Build namespace for problem parsing."""

        class dt(operators.TimeDerivative):
            name = 'd' + self.time
            _scalar = field.Scalar(name=name)

            @property
            def base(self):
                return dt

        additions = {}
        additions[self.time] = self._t = field.Scalar(name=self.time)
        additions[dt.name] = self._dt = dt
        return additions

    def _check_conditions(self, temp):
        """Check object-form conditions."""
        self._require_independent(temp, 'LHS', [self._t])
        self._require_first_order_comp(temp, 'LHS', [self._dt])
        self._require_independent(temp, 'RHS', [self._dt])

    def _set_matrix_expressions(self, temp):
        """Set expressions for building solver."""
        M, L = temp['LHS'].split(self._dt)
        M = Operand.cast(M)
        M = M.replace(self._dt, lambda x: x)
        vars = [self.namespace[var] for var in self.variables]
        temp['M'] = self._prep_linear_form(M, vars, name='M')
        temp['L'] = self._prep_linear_form(L, vars, name='L')
        temp['F'] = temp['RHS']


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
        vars = [self.namespace[var] for var in self.variables]
        self._require_independent(temp, 'RHS', vars)

    def _set_matrix_expressions(self, temp):
        """Set expressions for building solver."""
        vars = [self.namespace[var] for var in self.variables]
        temp['L'] = self._prep_linear_form(temp['LHS'], vars, name='L')
        temp['F'] = temp['RHS']


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
            pert = 'δ' + var
            additions[pert] = self.domain.new_field(name=pert)
            additions[pert].meta = self.meta[var]
            additions[pert].set_scales(1, keep_data=False)
        return additions

    def _check_conditions(self, temp):
        """Check object-form conditions."""
        pass

    def _set_matrix_expressions(self, temp):
        """Set expressions for building solver."""
        ep = field.Scalar(name='__epsilon__')
        vars = [self.namespace[var] for var in self.variables]
        perts = [self.namespace['δ'+var] for var in self.variables]
        # Build LHS operating on perturbations
        L = temp['LHS']
        for var, pert in zip(vars, perts):
            L = L.replace(var, pert)
        # Build Frechet derivative of RHS
        F = temp['RHS']
        dF = 0
        for var, pert in zip(vars, perts):
            dFi = F.replace(var, var + ep*pert)
            dFi = field.Operand.cast(dFi.sym_diff(ep))
            dFi = dFi.replace(ep, 0)
            dF += dFi
        # Set expressions
        temp['L'] = self._prep_linear_form(L, perts, name='L')
        temp['dF'] = self._prep_linear_form(dF, perts, name='dF')
        temp['F-L'] = temp['RHS'] - temp['LHS']


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
        Eigenvalue label, e.g. 'sigma' WARNING: 'lambda' is a python reserved word.
        You *cannot* use it as your eigenvalue. Also, note that unicode symbols
        don't work on all machines.
    tolerance : float
        A floating point number (>= 0) which helps 'define' zero for the RHS of the equation.
        If the RHS has nonzero NCCs which add to zero, dedalus will check to make sure that
        the max of the expression on the RHS normalized by the max of all NCCs going in that expression
        is smaller than this tolerance (see ProblemBase._check_if_zero() )

    Notes
    -----
    This class supports linear eigenvalue problems.  The LHS terms must be
    linear in the specified variables, first-order in coupled derivatives,
    and linear or independent of the specified eigenvalue.  The RHS must be zero.

        σM.X + L.X = 0

    """

    solver_class = solvers.EigenvalueSolver

    def __init__(self, domain, variables, eigenvalue, tolerance=1e-10, **kw):
        super().__init__(domain, variables, **kw)
        self.eigenvalue = eigenvalue
        self.tol=tolerance
        logger.info('Solving EVP with homogeneity tolerance of {:.3e}'.format(self.tol))

    @CachedAttribute
    def namespace_additions(self):
        """Build namespace for problem parsing."""
        additions = {}
        additions[self.eigenvalue] = self._ev = field.Scalar(name=self.eigenvalue)
        return additions

    def _check_conditions(self, temp):
        """Check object-form conditions."""
        vars = [self.namespace[var] for var in self.variables]
        self._require_homogeneous(temp, 'RHS', vars)
        self._require_first_order_mul(temp, 'LHS', [self._ev])
        self._require_first_order_mul(temp, 'RHS', [self._ev])

    def _set_matrix_expressions(self, temp):
        """Set expressions for building solver."""
        # Add RHS linearization to LHS
        ep = field.Scalar(name='__epsilon__')
        vars = [self.namespace[var] for var in self.variables]
        dF = temp['RHS']
        for var in vars:
            dF = dF.replace(var, ep*var)
        dF = field.Operand.cast(dF.sym_diff(ep))
        dF = dF.replace(ep, 0)
        temp['linearization'] = temp['LHS'] - dF
        # Build matrices from linearization
        M, L = temp['linearization'].split(self._ev)
        M = Operand.cast(M)
        M = M.replace(self._ev, 1)
        vars = [self.namespace[var] for var in self.variables]
        temp['M'] = self._prep_linear_form(M, vars, name='M')
        temp['L'] = self._prep_linear_form(L, vars, name='L')


# Aliases
IVP = InitialValueProblem
LBVP = LinearBoundaryValueProblem
NLBVP = NonlinearBoundaryValueProblem
EVP = EigenvalueProblem

