

from scipy import sparse
from itertools import izip


class Nonlinear(object):

    def __init__(self, F):

        # Convert to easily iterable structures
        F = [sparse.coo_matrix(f) for f in F]

        # Make sure each f is upper-diagonal
        for f in F:
            for i, j, v in izip(f.row, f.col, f.data):
                if i > j:
                    raise ValueError("Specify F matrices in upper-diagonal form.")

        # Get set of factor indeces
        factor_indeces = set()
        for f in F:
            factor_indeces.update(set(f.row))
            factor_indeces.update(set(f.col))

        # Build factor dictionary
        factors = {}
        for fi in factor_indeces:
            operator_index, field_index = divmod(fi, problem.size)
            factors[fi] = {'operator_index': operator_index,
                          'field_index': field_index,
                          'computed': False,
                          'field': state.fields[field_index]}

        # Loop through dimensions
        for i in reveresed(xrange(dimensions)):
            # Transpose fields
            for f in factors:
                f['field'].require_local(i)
            # Check operators
            for f in factors:
                attempt_operation(f)
            # Tranform fields
            for f in factors:
                f['field'].require_space(i, 'x')
            # Check operators
            for f in factors:
                attempt_operation(f)

        # Compute products

        # Loop back through spaces
            # Check if value operators can operate
                # If yes, reduce

        # Done

    def attempt_operation(factor):

        if not factor['computed']:
            if f['operator'].check_layout(layout):
                operator.apply(f['field'])











    # schedule = {}
    # for lo in layouts:
    #     schedule[lo] = {'pre-ops': [],
    #                     'transforms': [],
    #                     'post-ops': []}
