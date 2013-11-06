

from scipy import sparse


def get_fields(expressions):

    fields = set()
    for re in expressions:
        if re is not None:
            fields.update(re.field_set())

    return fields


def compute_expressions(rhs_expressions, out_system):

    # Make local copy of list
    expressions = list(rhs_expressions)

    # Skip where no F is specified
    for i, re in enumerate(expressions):
        if re is None:
            fn = out_system.field_names[i]
            out_system[fn]['c'] = 0.

    # Start all from coefficient space
    fields = get_fields(expressions)
    for f in fields:
        f.require_coeff_space()

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
                re_eval = re.evaluate()
                if re_eval is not None:
                    fn = out_system.field_names[j]
                    layout = re_eval.layout
                    out_system[fn][layout] = re_eval[layout]
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
                re_eval = re.evaluate()
                if re_eval is not None:
                    fn = out_system.field_names[j]
                    layout = re_eval.layout
                    out_system[fn][layout] = re_eval[layout]
                    expressions[j] = None

    # If necessary, perform this multiple times until everything is computed
    #   (In case someone does "dx(u*v) * dx(u*v)" or similar)


# It may be the case that the loops to check if evaluations can occur take up
# too much time.  In this case, we want to perform this process once at the
# beginning to build some sort of schedule of tasks that can then be done
# quickly on the fly.

