

from scipy import sparse
from itertools import izip


def compute_expressions(rhs_expressions):

    # Loop from K to X
    for i in reveresed(xrange(dimensions)):

        # Transpose fields
        fields = set([])
        for re in rhs_expressions:
            fields.update(re.field_set())
        for f in fields:
            f.require_local(i)

        # Attempt evaluation
        for re in rhs_expressions:
            re.attempt_evaluation()

        # Transform fields
        fields = set([])
        for re in rhs_expressions:
            fields.update(re.field_set())
        for f in fields:
            f.require_space(i, 'x')

        # Attempt evaluation
        for re in rhs_expressions:
            re.attempt_evaluation()

    # Non-linear products should now be computed (we are in X)

    # Repeat looping from X to K to compute operators acting on products

    # If necessary, perform this multiple times until everything is computed
    #   (In case someone does "dx(u*v) * dx(u*v)" or similar)


# It may be the case that the loops to check if evaluations can occur take up
# too much time.  In this case, we want to perform this process once at the
# beginning to build some sort of schedule of tasks that can then be done
# quickly on the fly.

