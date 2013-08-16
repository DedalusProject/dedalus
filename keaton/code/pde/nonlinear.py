

from scipy import sparse


def get_fields(rhs_expressions):

    fields = set()
    for re in rhs_expressions:
        if re is not None:
            fields.update(re.field_set())

    return fields


def compute_expressions(rhs_expressions, out_system):

    # Make local copy of list
    rhs_expressions = [re for re in rhs_expressions]

    # Skip where no F is specified
    for i, re in enumerate(rhs_expressions):
        if re is None:
            fn = out_system.field_names[i]
            out_system[fn]['K'] = 0.

    # Start all from k space
    fields = get_fields(rhs_expressions)
    for f in fields:
        f.require_global_space('K')

    dimensions = out_system.domain.dim

    # Loop from K to X
    for i in reversed(range(dimensions)):

        # Transpose fields
        fields = get_fields(rhs_expressions)
        for f in fields:
            f.require_local(i)

        # Attempt evaluation
        for j, re in enumerate(rhs_expressions):
            if re is not None:
                re_eval = re.evaluate()
                if re_eval is not None:
                    fn = out_system.field_names[j]
                    space = ''.join(re_eval.space)
                    out_system[fn][space] = re_eval[space]
                    rhs_expressions[j] = None

        # Transform fields
        fields = get_fields(rhs_expressions)
        for f in fields:
            f.require_space(i, 'x')

        # Attempt evaluation
        for j, re in enumerate(rhs_expressions):
            if re is not None:
                re_eval = re.evaluate()
                if re_eval is not None:
                    fn = out_system.field_names[j]
                    space = ''.join(re_eval.space)
                    out_system[fn][space] = re_eval[space]
                    rhs_expressions[j] = None

    # Non-linear products should now be computed (we are in X)

    # Repeat looping from X to K to compute operators acting on products

    # Loop from X to K
    for i in range(dimensions):

        # Transpose fields
        fields = get_fields(rhs_expressions)
        for f in fields:
            f.require_local(i)

        # Attempt evaluation
        for j, re in enumerate(rhs_expressions):
            if re is not None:
                re_eval = re.evaluate()
                if re_eval is not None:
                    fn = out_system.field_names[j]
                    space = ''.join(re_eval.space)
                    out_system[fn][space] = re_eval[space]
                    rhs_expressions[j] = None

        # Transform fields
        fields = get_fields(rhs_expressions)
        for f in fields:
            f.require_space(i, 'k')

        # Attempt evaluation
        for j, re in enumerate(rhs_expressions):
            if re is not None:
                re_eval = re.evaluate()
                if re_eval is not None:
                    fn = out_system.field_names[j]
                    space = ''.join(re_eval.space)
                    out_system[fn][space] = re_eval[space]
                    rhs_expressions[j] = None

    # If necessary, perform this multiple times until everything is computed
    #   (In case someone does "dx(u*v) * dx(u*v)" or similar)


# It may be the case that the loops to check if evaluations can occur take up
# too much time.  In this case, we want to perform this process once at the
# beginning to build some sort of schedule of tasks that can then be done
# quickly on the fly.

