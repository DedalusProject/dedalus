"""
Classes for representing systems of equations.

"""

import re
from collections import OrderedDict
import numpy as np
import sympy as sy

from ..data.metadata import MultiDict, Metadata
from ..data import field
from ..data import future
from ..data import operators
from ..tools.cache import CachedAttribute

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class SymbolicParsingError(Exception):
    pass

class UnsupportedEquationError(Exception):
    pass


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


class ProblemBase:
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


    def __init__(self, domain, variables):

        self.domain = domain
        self.variables = variables
        self.nvars = len(variables)

        self.meta = MultiDict({var: Metadata(domain) for var in variables})

        self.equations = self.eqs = []
        self.boundary_conditions = self.bcs = []

        self.parameters = OrderedDict()
        self.substitutions = OrderedDict()

    @CachedAttribute
    def namespace(self):

        namespace = Namespace()
        namespace.disallow_overwrites()
        # Basis-specific items
        for axis, basis in enumerate(self.domain.bases):
            # Grids
            grid = field.Array(self.domain, name=basis.name)
            grid.from_global_vector(basis.grid(1), axis)
            namespace[basis.name] = grid
            # Basis operators
            for op in basis.operators:
                namespace[op.name] = op
        # Fields
        for var in self.variables:
            namespace[var] = self.domain.new_field(name=var, allocate=False)
            namespace[var].meta = self.meta[var]
        # Parameters
        namespace.update(self.parameters)
        # Built-in functions
        namespace['Identity'] = 1
        namespace.update(operators.parseables)
        # Substitutions
        for call, result in self.substitutions.items():
            # Convert function calls to lambda expressions
            call, result = self._convert_functions(call, result)
            # Evaluate in current namespace
            namespace[call] = eval(result, namespace)

        return namespace

    def _build_basic_dictionary(self, equation, condition):
        LHS, RHS = self._split_sides(equation)
        dct = dict()
        dct['raw_equation'] = equation
        dct['raw_condition'] = condition
        dct['raw_LHS'] = LHS
        dct['raw_RHS'] = RHS
        logger.debug("  Condition: {}".format(condition))
        logger.debug("  LHS string form: {}".format(LHS))
        logger.debug("  RHS string form: {}".format(RHS))
        dct['condition'] = self._lambdify_condition(dct['raw_condition'])
        return dct

    @staticmethod
    def _split_sides(equation):
        """Split equation string into LHS and RHS strings."""
        # Track parenthetical level to only capture top-level equals signs,
        # which avoids capturing equals signs in keyword assigments
        parentheses = 0
        top_level_equals = []
        for i, character in enumerate(equation):
            if character == '(':
                parentheses += 1
            elif character == ')':
                parentheses -= 1
            elif parentheses == 0:
                if character == '=':
                    top_level_equals.append(i)
        if len(top_level_equals) == 0:
            raise SymbolicParsingError("Equation contains no top-level equals signs.")
        elif len(top_level_equals) > 1:
            raise SymbolicParsingError("Equation contains multiple top-level equals signs.")
        else:
            i, = top_level_equals
            return equation[:i], equation[i+1:]

    def _lambdify_condition(self, condition):
        """Lambdify condition test for fast evaluation."""
        # Write call signiture for function of separable indices
        index_names = ['n{}'.format(basis.name) for basis in self.domain.bases if basis.separable]
        call = 'c({})'.format(','.join(index_names))
        # Convert to lambda definition
        call, result = self._convert_functions(call, condition)
        # Evaluate to lambda expression
        return eval(result, {})

    @staticmethod
    def _convert_functions(call, result):
        """Convert math-style function definitions into Python lambda expressions."""
        # Use regular expressions to see if name matches a function call
        match = re.match('(.+)\((.*)\)', call)
        if match:
            # Build lambda expression
            func, args = match.groups()
            lambda_def = 'lambda {}: {}'.format(args, result)
            return func, lambda_def
        else:
            # Return original rule
            return call, result

    def _build_object_forms(self, dct):
        LHS = future.FutureField.parse(dct['raw_LHS'], self.namespace, self.domain)
        RHS = future.FutureField.parse(dct['raw_RHS'], self.namespace, self.domain)
        logger.debug("  LHS object form: {}".format(LHS))
        logger.debug("  RHS object form: {}".format(RHS))
        return LHS, RHS

    def _check_object_conditions(self, LHS, RHS, BC):
        # Compare metadata
        self._check_meta_consistency(LHS, RHS)
        if BC:
            # Check constant along last axis
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
        var_dummies = [self.namespace[var] for var in self.variables]
        LHS = LHS.distribute_over(var_dummies)
        linear, op_form = LHS.as_symbolic_operator(var_dummies)
        LHS = (op_form).simplify().expand()
        logger.debug("  LHS operator form: {}".format(LHS))
        if not linear:
            raise UnsupportedEquationError("LHS must contain terms that are linear in fields.")
        return LHS

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

    def lambdify_variable_coefficients(self, expr):
        # Extract symbolic variable coefficients
        vars = [sy.Symbol(var, commutative=False) for var in self.variables]
        coeffs = self._extract_left_coefficients(expr, vars)
        # Multiply through by identity
        I = sy.Symbol('Identity')
        coeffs = [I*c for c in coeffs]
        # Find atoms (operators and NCC expressions) needed to build rows
        atoms = self._reduce_atoms(coeffs, sy.Symbol)
        atom_objects = [eval(str(atom), self.namespace) for atom in atoms]
        # Impose NCC constraints
        self._require_independent(atom_objects, self.namespace[self.time])
        self._require_separable(atom_objects)

        atom_names = [atom.name for atom in atoms]
        lambdified_coeffs = self._lambdify_row(atoms, coeffs)
        return atom_names, lambdified_coeffs

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

    @staticmethod
    def _reduce_atoms(exprs, *args, **kw):
        """Extract symbolic atoms across a list of expressions."""
        atoms = set()
        for expr in exprs:
            expr = sy.sympify(expr)
            atoms.update(expr.atoms(*args, **kw))
        atoms = sorted(list(atoms), key=sy.default_sort_key)
        return atoms

    def _require_independent(self, atoms, dep):
        """Require NCCs to be independent of some atom."""
        for atom in atoms:
            # Impose condition on NCCs, skipping raw operator classes
            if isinstance(atom, field.Operand):
                if atom.has(dep):
                    raise UnsupportedEquationError("LHS coefficient '{}' is not independent of '{}'.".format(atom, dep))

    def _require_separable(self, atoms):
        """Require NCCs to be constant along separable directions."""
        for atom in atoms:
            #print(atom)
            # Impose condition on NCCs, skipping raw operator classes
            if isinstance(atom, field.Operand):
                for axis, basis in enumerate(self.domain.bases):
                    if basis.separable:
                        if not atom.meta[axis]['constant']:
                            raise UnsupportedEquationError("LHS coefficient '{}' is non-constant along separable direction '{}'.".format(atom, basis.name))
            elif isinstance(atom, type):
                if issubclass(atom, operators.Coupled):
                    if atom.basis.separable:
                        raise UnsupportedEquationError("LHS coefficient '{}' is coupled along direction '{}'.".format(atom.name, atom.basis.name))

    @staticmethod
    def _lambdify_row(atoms, row):

        def dummy_lambdify(atoms, dummies, expr):
            expr = sy.sympify(expr)
            # Substitute dummies
            expr = expr.subs(zip(atoms, dummies))
            # Collect atoms to try to minimize dot products
            expr = expr.collect(dummies).simplify()
            # Lambdify over dummies
            return sy.lambdify(dummies, expr, modules='numpy')

        # Replace atoms with dummies
        # (NCC expressions might not be valid identifiers)
        dummies = [sy.Dummy(**atom.assumptions0) for atom in atoms]
        return [dummy_lambdify(atoms, dummies, coeff) for coeff in row]

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

    # def num_coeffs(self, sym_coeffs, domain, order):
    #     def op_matrix(op, domain):
    #         if op == 1:
    #             return 1
    #         if op.name in self.coupled_ops:
    #             # Lookup through domain
    #             dummy = domain.new_field()
    #             return coupled_ops[op.name](dummy).matrix_form()
    #         else:
    #             # Evaluate NCCs as operator trees
    #             print(op)
    #             operator = operators.Operator.from_string(op, self.parameters, domain)
    #             field = operator.evaluate()
    #             field.require_coeff_space()
    #             # Scatter const coeffs from root
    #             if domain.dist.rank == 0:
    #                 select = (0,) * (domain.dim - 1)
    #                 coeffs = field['c'][select]
    #             else:
    #                 coeffs = None
    #             coeffs = domain.dist.comm_cart.scatter(coeffs, root=0)
    #             return domain.bases[-1].build_mult(coeffs, order)


class Problem(ProblemBase):

    def __init__(self, domain, variables, time='t'):

        super().__init__(domain, variables)
        self.time = time

    @CachedAttribute
    def namespace(self):

        # Build time derivative operator
        class dt(future.FutureScalar):
            name = 'd' + self.time
            def as_symbolic_operator(self, fields):
                arg_linear, arg_op = self.args[0].as_symbolic_operator(fields)
                if arg_linear:
                    op = sy.Symbol(self.name) * arg_op
                    return True, op
                else:
                    raise ValueError("Cannot take time derivative of non-field")
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

    def add_equation(self, equation, condition="True"):
        """Add equation to problem."""

        logger.debug("Parsing Eqn {}".format(len(self.boundary_conditions)))
        dct = self._build_basic_dictionary(equation, condition)

        LHS, RHS = self._build_object_forms(dct)
        self._check_object_conditions(LHS, RHS, BC=False)

        LHS = self._build_operator_form(LHS)
        #self._check_operator_conditions(dct, LHS)
        dct['differential'] = self._coupled_differential_order(LHS)

        L, M = self._split_linear(LHS, sy.Symbol('d'+self.time))
        dct['L_atoms'], dct['L'] = self.lambdify_variable_coefficients(L)
        dct['M_atoms'], dct['M'] = self.lambdify_variable_coefficients(M)

        self.equations.append(dct)

    def add_bc(self, equation, condition="True"):
        """Add equation to problem."""

        logger.debug("Parsing BC {}".format(len(self.boundary_conditions)))
        dct = self._build_basic_dictionary(equation, condition)

        LHS, RHS = self._build_object_forms(dct)
        self._check_object_conditions(LHS, RHS, BC=True)

        LHS = self._build_operator_form(LHS)
        #self._check_operator_conditions(dct, LHS)
        dct['differential'] = self._coupled_differential_order(LHS)

        L, M = self._split_linear(LHS, sy.Symbol('d'+self.time))
        dct['L_atoms'], dct['L'] = self.lambdify_variable_coefficients(L)
        dct['M_atoms'], dct['M'] = self.lambdify_variable_coefficients(M)

        self.boundary_conditions.append(dct)



