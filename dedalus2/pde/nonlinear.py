"""
Functions for evaluating operator trees.

"""

import numpy as np
from scipy import sparse

from ..tools.general import OrderedSet


def get_fields(expressions):
    """Get set of field leaves from multiple operators."""

    fields = OrderedSet()
    for re in expressions:
        if re is not None:
            fields.update(re.field_set())

    return fields


def compute_expressions(rhs_expressions, out_system):
    """
    Evaluate a list of expressions by incrementally transforming all fields.

    Parameters
    ----------
    rhs_expressions : list of operator objects
        Expressions to evaluate
    out_system : system object
        System to copy results to

    """

    # Make local copy of list
    expressions = list(rhs_expressions)

    # Skip where no F is specified
    for j, re in enumerate(expressions):
        if np.isscalar(re):
            out_system.fields[j]['g'] = re
            expressions[j] = None

    # Start all from coefficient space
    fields = get_fields(expressions)
    for f in fields:
        f.require_coeff_space()

    # Attempt evaluation
    for j, re in enumerate(expressions):
        if re is not None:
            re_eval = re.attempt()
            if re_eval is not None:
                layout = re_eval.layout
                out_system.fields[j][layout] = re_eval[layout]
                expressions[j] = None

    n_layouts = len(out_system.domain.distributor.layouts)

    # Loop from coefficient to grid space
    for i in range(n_layouts - 1):

        # Next layout
        fields = get_fields(expressions)
        for f in fields:
            f.towards_grid_space()

        # Attempt evaluation
        for j, re in enumerate(expressions):
            if re is not None:
                re_eval = re.attempt()
                if re_eval is not None:
                    layout = re_eval.layout
                    out_system.fields[j][layout] = re_eval[layout]
                    expressions[j] = None

    # Non-linear products should now be computed (we are in grid space)

    # Repeat looping from grid to coefficient space to compute operators acting on products?

    # Loop from grid to coefficient space
    for i in range(n_layouts - 1):

        # Previous layout
        fields = get_fields(expressions)
        for f in fields:
            f.towards_coeff_space()

        # Attempt evaluation
        for j, re in enumerate(expressions):
            if re is not None:
                re_eval = re.attempt()
                if re_eval is not None:
                    layout = re_eval.layout
                    out_system.fields[j][layout] = re_eval[layout]
                    expressions[j] = None

    if any(expressions):
        raise ValueError("Not all expressions evaluated.")

    # If necessary, perform this multiple times until everything is computed
    #   (In case someone does "dx(u*v) * dx(u*v)" or similar)


# It may be the case that the loops to check if evaluations can occur take up
# too much time.  In this case, we want to perform this process once at the
# beginning to build some sort of schedule of tasks that can then be done
# quickly on the fly.

