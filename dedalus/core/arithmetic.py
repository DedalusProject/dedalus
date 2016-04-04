



from collections import defaultdict
from functools import partial, reduce
import numpy as np
from scipy import sparse
from numbers import Number

from .domain import Subdomain
from .field import Operand, Data, Array, Field
from .future import Future, FutureArray, FutureField
from .operators import NonlinearOperator, convert
from ..tools.array import reshape_vector, apply_matrix, add_sparse, axslice
from ..tools.cache import CachedAttribute
from ..tools.dispatch import MultiClass
from ..tools.exceptions import NonlinearOperatorError
from ..tools.exceptions import SymbolicParsingError
from ..tools.exceptions import UndefinedParityError
from ..tools.general import unify, unify_attributes


class Rand:
    pass
Scalar = FutureScalar = Rand







class Arithmetic(Future):

    arity = 2

    def __str__(self):
        def substring(arg):
            if isinstance(arg, Arithmetic):
                return '({})'.format(arg)
            else:
                return str(arg)
        str_args = map(substring, self.args)
        return '%s' %self.str_op.join(str_args)

    def order(self, *ops):
        return max(arg.order(*ops) for arg in self.args)


class Add(Arithmetic, metaclass=MultiClass):

    name = 'Add'
    str_op = ' + '

    def __init__(self, arg0, arg1, out=None):
        # Convert arguments to output bases
        bases = self._build_bases(arg0, arg1)
        arg0 = convert(arg0, bases)
        arg1 = convert(arg1, bases)
        super().__init__(arg0, arg1, out=out)

    def _build_bases(self, arg0, arg1):
        bases = []
        for b0, b1 in zip(arg0.bases, arg1.bases):
            if (b0 is None) and (b1 is None):
                bases.append(None)
            else:
                bases.append(b0 + b1)
        return tuple(bases)

    @classmethod
    def _preprocess_args(cls, *args, **kw):
        domain = unify_attributes(args, 'domain', require=False)
        args = tuple(Operand.cast(arg, domain) for arg in args)
        return args, kw

    @classmethod
    def _check_args(cls, *args, **kw):
        match = (isinstance(args[i], types) for i,types in cls.argtypes.items())
        return all(match)

    def base(self):
        return Add

    def expand(self, *vars):
        """Expand arguments containing specified variables (default: all)."""
        arg0, arg1 = self.args
        if (not vars) or arg0.has(*vars):
            arg0 = arg0.expand(*vars)
        if (not vars) or arg1.has(*vars):
            arg1 = arg1.expand(*vars)
        if (not vars) or arg0.has(*vars) or arg1.has(*vars):
            return arg0 + arg1
        else:
            return self

    def canonical_linear_form(self, *vars):
        """Ensure arguments have same dependency on specified variables."""
        arg0, arg1 = self.args
        if arg0.has(*vars) and arg1.has(*vars):
            arg0 = arg0.canonical_linear_form(*vars)
            arg1 = arg1.canonical_linear_form(*vars)
            return arg0 + arg1
        elif arg0.has(*vars) or arg1.has(*vars):
            raise NonlinearOperatorError("Cannot add dependent and independent terms.")
        else:
            return self

    def separability(self, vars):
        arg0, arg1 = self.args
        sep0 = arg0.separability(vars)
        sep1 = arg1.separability(vars)
        return (sep0 & sep1)

    def split(self, *vars):
        """Split expression based on presence of variables."""
        S0 = self.args[0].split(*vars)
        S1 = self.args[1].split(*vars)
        return [S0[0]+S1[0], S0[1]+S1[1]]

    def subproblem_matrices(self, subproblem, vars, **kw):
        """Build expression matrices acting on subproblem group data."""
        arg0, arg1 = self.args
        # Build argument matrices
        mat0 = arg0.subproblem_matrices(subproblem, vars, **kw)
        mat1 = arg1.subproblem_matrices(subproblem, vars, **kw)
        # Add matrices from each argument
        out = defaultdict(int)
        for var in mat0:
            out[var] = out[var] + mat0[var]
        for var in mat1:
            out[var] = out[var] + mat1[var]
        return out

    def sym_diff(self, var):
        """Symbolically differentiate with respect to var."""
        arg0, arg1 = self.args
        diff0 = arg0.sym_diff(var)
        diff1 = arg1.sym_diff(var)
        return diff0 + diff1

    # def add_subdata(self, arg, out):
    #     # (Only called if out.data.size != 0)
    #     arg_slices, out_slices = [], []
    #     for axis in range(self.domain.dim):
    #         if arg.bases[axis] is out.bases[axis]:
    #             # (Both None or both not None)
    #             # Directly add all data
    #             arg_slices.append(slice(None))
    #             out_slices.append(slice(None))
    #         else:
    #             # (arg basis is None)
    #             if out.layout.grid_space[axis]:
    #                 # Broadcast addition
    #                 arg_slices.append(slice(None))
    #                 out_slices.append(slice(None))
    #             else:
    #                 # Select constant mode
    #                 #const_slice = arg.layout.select_global(0, axis=axis)
    #                 if out.global_start[axis] == 0:
    #                     const_slice = slice(1)
    #                 else:
    #                     const_slice = slice(0)
    #                 arg_slices.append(const_slice)
    #                 out_slices.append(const_slice)
    #     arg_data = arg.data[tuple(arg_slices)]
    #     out_data = out.data[tuple(out_slices)]
    #     np.add(arg_data, out_data, out=out_data)


class AddArrayArray(Add, FutureArray):

    argtypes = {0: (Array, FutureArray),
                1: (Array, FutureArray)}

    def check_conditions(self):
        return True

    def enforce_conditions(self):
        pass

    def operate(self, out):
        arg0, arg1 = self.args
        if out.data.size:
            out.data.fill(0)
            self.add_subdata(arg0, out)
            self.add_subdata(arg1, out)


class AddFieldField(Add, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Field, FutureField)}

    def check_conditions(self):
        # Layouts must match
        return (self.args[0].layout is self.args[1].layout)

    def enforce_conditions(self):
        arg0, arg1 = self.args
        layout = self.choose_layout()
        arg0.require_layout(layout)
        arg1.require_layout(layout)

    def choose_layout(self):
        arg0, arg1 = self.args
        # Pick arg0 layout (arbitrary choice)
        return arg0.layout

    def operate(self, out):
        arg0, arg1 = self.args
        out.set_layout(arg0.layout)
        np.add(arg0.data, arg1.data, out.data)


class AddScalarArray(Add, FutureArray):

    argtypes = {0: (Scalar, FutureScalar),
                1: (Array, FutureArray)}

    def check_conditions(self):
        return True

    def enforce_conditions(self):
        pass

    def operate(self, out):
        arg0, arg1 = self.args
        np.add(arg0.value, arg1.data, out.data)


class AddArrayScalar(Add, FutureArray):

    argtypes = {0: (Array, FutureArray),
                1: (Scalar, FutureScalar)}

    def check_conditions(self):
        return True

    def enforce_conditions(self):
        pass

    def operate(self, out):
        arg0, arg1 = self.args
        np.add(arg0.data, arg1.value, out.data)


class AddScalarField(Add, FutureField):

    argtypes = {0: (Scalar, FutureScalar),
                1: (Field, FutureField)}

    def check_conditions(self):
        # Field must be in grid layout
        return (self.args[1].layout is self._grid_layout)

    def enforce_conditions(self):
        self.args[1].require_layout(self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Add in grid layout
        arg1.require_grid_space()
        out.set_layout(self._grid_layout)
        np.add(arg0.value, arg1.data, out.data)


class AddFieldScalar(Add, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Scalar, FutureScalar)}

    def check_conditions(self):
        # Field must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def enforce_conditions(self):
        self.args[0].require_layout(self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Add in grid layout
        arg0.require_grid_space()
        out.set_layout(self._grid_layout)
        np.add(arg0.data, arg1.value, out.data)


class AddArrayField(Add, FutureField):

    argtypes = {0: (Array, FutureArray),
                1: (Field, FutureField)}

    def check_conditions(self):
        # Field must be in grid layout
        return (self.args[1].layout is self._grid_layout)

    def enforce_conditions(self):
        self.args[1].require_layout(self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Add in grid layout
        out.set_layout(self._grid_layout)
        np.add(arg0.data, arg1.data, out.data)


class AddFieldArray(Add, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Array, FutureArray)}

    def check_conditions(self):
        # Field must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def enforce_conditions(self):
        self.args[0].require_layout(self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Add in grid layout
        out.set_layout(self._grid_layout)
        np.add(arg0.data, arg1.data, out.data)


class Multiply(Arithmetic, metaclass=MultiClass):

    name = 'Mul'
    str_op = '*'

    def _build_bases(self, arg0, arg1):
        bases = []
        for b0, b1 in zip(arg0.bases, arg1.bases):
            if (b0 is None) and (b1 is None):
                bases.append(None)
            else:
                bases.append(b0 * b1)
        return bases

    @classmethod
    def _preprocess_args(cls, *args, **kw):
        domain = unify_attributes(args, 'domain', require=False)
        args = tuple(Operand.cast(arg, domain) for arg in args)
        return args, kw

    @classmethod
    def _check_args(cls, *args, **kw):
        match = (isinstance(args[i], types) for i,types in cls.argtypes.items())
        return all(match)

    def base(self):
        return Multiply

    def expand(self, *vars):
        """Distribute over sums containing specified variables (default: all)."""
        arg0, arg1 = self.args
        if (not vars) or arg0.has(*vars):
            arg0 = self.args[0].expand(*vars)
            if isinstance(arg0, Add):
                arg0a, arg0b = arg0.args
                return (arg0a*arg1 + arg0b*arg1).expand(*vars)
        if (not vars) or arg1.has(*vars):
            arg1 = self.args[1].expand(*vars)
            if isinstance(arg1, Add):
                arg1a, arg1b = arg1.args
                return (arg0*arg1a + arg0*arg1b).expand(*vars)
        return self

    def canonical_linear_form(self, *vars):
        """Eliminate nonlinear multiplications and float specified variables right."""
        arg0, arg1 = self.args
        if arg0.has(*vars) and arg1.has(*vars):
            raise NonlinearOperatorError("Cannot multiply two linear terms.")
        elif arg0.has(*vars):
            arg0 = arg0.canonical_linear_form(*vars)
            if isinstance(arg0, Multiply):
                arg0a, arg0b = arg0.args
                return (arg0a * arg1) * arg0b
            else:
                return arg1 * arg0
        elif arg1.has(*vars):
            arg1 = arg1.canonical_linear_form(*vars)
            if isinstance(arg1, Multiply):
                arg1a, arg1b = arg1.args
                return (arg0 * arg1a) * arg1b
            else:
                return arg0 * arg1
        else:
            return self

    def split(self, *vars):
        S0 = self.args[0].split(*vars)
        S1 = self.args[1].split(*vars)
        return [S0[0]*S1[0] + S0[0]*S1[1] + S0[1]*S1[0], S0[1]*S1[1]]

    def subproblem_matrices(self, subproblem, vars, **kw):
        """Build expression matrices acting on subproblem group data."""
        arg0, arg1 = self.args
        mat0 = arg0.as_ncc_matrix(arg1, subproblem, **kw)
        mat1 = arg1.subproblem_matrices(subproblem, vars, **kw)
        return {var: mat0*mat1[var] for var in mat1}

    def separability(self, vars):
        """Determine separability as linear operator over specified variables."""
        # Assume canonical linear form: arg1 linearly depends on vars
        arg0, arg1 = self.args
        # NCC multiplication is separable on constant axes
        sep0 = arg0.separability(vars)
        sep1 = arg1.separability(vars)
        return (sep0 & sep1)

    def sym_diff(self, var):
        """Symbolically differentiate with respect to var."""
        arg0, arg1 = self.args
        diff0 = arg0.sym_diff(var)
        diff1 = arg1.sym_diff(var)
        return diff0*arg1 + arg0*diff1

    # def simplify(self, retain):
    #     arg0 = self.args[0].simplify(retain)
    #     arg1 = self.arts[0].simplify(retain)

    #     if arg0 not in retain:
    #         if arg0 == 0:
    #             return 0
    #         elif arg0 == 1:
    #             return arg1
    #     if arg1 not in retain:
    #         elif arg1 == 0:
    #             return 0
    #         elif arg1 == 1:
    #             return arg0
    #     return (arg0 * arg1)




class MultiplyArrayArray(Multiply, FutureArray):

    argtypes = {0: (Array, FutureArray),
                1: (Array, FutureArray)}

    def check_conditions(self):
        return True

    def enforce_conditions(self):
        pass

    def operate(self, out):
        arg0, arg1 = self.args
        np.multiply(arg0.data, arg1.data, out.data)


class MultiplyFieldField(Multiply, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Field, FutureField)}

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        arg0, arg1 = self.args
        for axis, (b0, b1) in enumerate(zip(arg0.bases, arg1.bases)):
            if (b0 is not None) and (b1 is not None):
                self.require_grid_axis = axis
                break
        else:
            self.require_grid_axis = None

    def check_conditions(self):
        layout0 = self.args[0].layout
        layout1 = self.args[1].layout
        # Fields must be in grid layout
        if self.require_grid_axis is not None:
            axis = self.require_grid_axis
            return (layout0.grid_space[axis] and (layout0 is layout1))
        else:
            return (layout0 is layout1)

    def enforce_conditions(self):
        arg0, arg1 = self.args
        if self.require_grid_axis is not None:
            axis = self.require_grid_axis
            arg0.require_grid_space(axis=axis)
        arg1.require_layout(arg0.layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Multiply in grid layout
        out.set_layout(arg0.layout)
        if out.data.size:
            np.multiply(arg0.data, arg1.data, out.data)


    ## Ideas for separating condition enforcement from operation to potentially
    ## trim down the boilerplate for the dispatching subclasses
    # def enforce_conditions(self):
    #     self.args[0].require_grid_space()
    #     self.args[1].require_grid_space()
    #     out.set_layout(self._grid_layout)
    # def _operate(self):
    #     np.multiply(self.args[0].data, self.args[1].data, out.data)


class MultiplyScalarArray(Multiply, FutureArray):

    argtypes = {0: Number,
                1: (Array, FutureArray)}

    def check_conditions(self):
        return True

    def enforce_conditions(self):
        pass

    def operate(self, out):
        arg0, arg1 = self.args
        np.multiply(arg0, arg1.data, out.data)


class MultiplyArrayScalar(Multiply, FutureArray):

    argtypes = {0: (Array, FutureArray),
                1: Number}

    def check_conditions(self):
        return True

    def enforce_conditions(self):
        pass

    def operate(self, out):
        arg0, arg1 = self.args
        np.multiply(arg0.data, arg1, out.data)


class MultiplyScalarField(Multiply, FutureField):

    argtypes = {0: (Scalar, FutureScalar),
                1: (Field, FutureField)}

    def check_conditions(self):
        return True

    def enforce_conditions(self):
        pass

    def operate(self, out):
        arg0, arg1 = self.args
        # Multiply in current layout
        out.set_layout(arg1.layout)
        np.multiply(arg0.value, arg1.data, out.data)


class MultiplyFieldScalar(Multiply, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Scalar, FutureScalar)}

    def check_conditions(self):
        return True

    def enforce_conditions(self):
        pass

    def operate(self, out):
        arg0, arg1 = self.args
        # Multiply in current layout
        out.set_layout(arg0.layout)
        np.multiply(arg0.data, arg1.value, out.data)


class MultiplyArrayField(Multiply, FutureField):

    argtypes = {0: (Array, FutureArray),
                1: (Field, FutureField)}

    def check_conditions(self):
        # Field must be in grid layout
        return (self.args[1].layout is self._grid_layout)

    def enforce_conditions(self):
        self.args[1].require_layout(self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Multiply in grid layout
        arg1.require_grid_space()
        out.set_layout(self._grid_layout)
        np.multiply(arg0.data, arg1.data, out.data)


class MultiplyFieldArray(Multiply, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Array, FutureArray)}

    def check_conditions(self):
        # Field must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def enforce_conditions(self):
        self.args[0].require_layout(self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Multiply in grid layout
        arg0.require_grid_space()
        out.set_layout(self._grid_layout)
        np.multiply(arg0.data, arg1.data, out.data)


class Power(NonlinearOperator, Arithmetic, metaclass=MultiClass):

    name = 'Pow'
    str_op = '**'

    def _build_bases(self, arg0, arg1):
        return arg0.bases

    @classmethod
    def _preprocess_args(cls, *args, **kw):
        domain = unify_attributes(args, 'domain', require=False)
        args = tuple(Operand.cast(arg, domain) for arg in args)
        return args, kw

    @classmethod
    def _check_args(cls, *args, **kw):
        match = (isinstance(args[i], types) for i,types in cls.argtypes.items())
        arg1_constant = all(b is None for b in args[1].bases)
        return (all(match) and arg1_constant)

    def base(self):
        return Power

    def sym_diff(self, var):
        """Symbolically differentiate with respect to var."""
        arg0, arg1 = self.args
        diff0 = arg0.sym_diff(var)
        return arg1 * arg0**(arg1-1) * diff0

    def separability(self, vars):
        """Determine separability as linear operator over specified variables."""
        return self.args[0].separability(vars)


class PowerFieldConstant(Power, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Field, FutureField)}

    def __new__(cls, arg0, arg1, *args, **kw):
        if (arg1.name is None) and (arg1.value == 0):
            return 1
        elif (arg1.name is None) and (arg1.value == 1):
            return arg0
        else:
            return object.__new__(cls)

    def __init__(self, arg0, arg1, out=None):
        super().__init__(arg0, arg1, out=out)
        for axis, b0 in enumerate(arg0.bases):
            if b0 is not None:
                self.require_grid_axis = axis
                break
        else:
            self.require_grid_axis = None

    def check_conditions(self):
        layout0 = self.args[0].layout
        layout1 = self.args[1].layout
        # Fields must be in grid layout
        if self.require_grid_axis is not None:
            axis = self.require_grid_axis
            return (layout0.grid_space[axis] and (layout0 is layout1))
        else:
            return (layout0 is layout1)

    def enforce_conditions(self):
        arg0, arg1 = self.args
        if self.require_grid_axis is not None:
            axis = self.require_grid_axis
            arg0.require_grid_space(axis=axis)
        arg1.require_layout(arg0.layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Multiply in grid layout
        out.set_layout(arg0.layout)
        if out.data.size:
            np.power(arg0.data, arg1.data, out.data)


# class PowerArrayScalar(PowerDataScalar, FutureArray):

#     argtypes = {0: (Array, FutureArray),
#                 1: Number}

#     def check_conditions(self):
#         return True

#     def operate(self, out):
#         arg0, arg1 = self.args
#         np.power(arg0.data, arg1.value, out.data)


# class PowerFieldNumber(PowerDataScalar, FutureField):

#     argtypes = {0: (Field, FutureField),
#                 1: Number}

#     def check_conditions(self):
#         # Field must be in grid layout
#         return (self.args[0].layout is self._grid_layout)

#     def operate(self, out):
#         arg0, arg1 = self.args
#         # Raise in grid layout
#         arg0.require_grid_space()
#         out.set_layout(self._grid_layout)
#         np.power(arg0.data, arg1.value, out.data)

