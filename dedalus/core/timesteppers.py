"""
ODE solvers for timestepping.

"""

from collections import deque, OrderedDict
import numpy as np
from scipy.sparse import linalg

from .system import CoeffSystem
from ..tools.sparse import fast_csr_matvec


# Track implemented schemes
schemes = OrderedDict()
def add_scheme(scheme):
    schemes[scheme.__name__] = scheme
    return scheme


class MultistepIMEX:
    """
    Base class for implicit-explicit multistep methods.

    Parameters
    ----------
    pencil_length : int
        Number of coefficients in a single pencil
    domain : domain object
        Problem domain

    Notes
    -----
    These timesteppers discretize the system
        M.dt(X) + L.X = F
    into the general form
        aj M.X(n-j) + bj L.X(n-j) = cj F(n-j)
    where j runs from {0, 0, 1} to {amax, bmax, cmax}.

    The system is then solved as
        (a0 M + b0 L).X(n) = cj F(n-j) - aj M.X(n-j) - bj L.X(n-j)
    where j runs from {1, 1, 1} to {cmax, amax, bmax}.

    References
    ----------
    D. Wang and S. J. Ruuth, Journal of Computational Mathematics 26, (2008).*

    * Our coefficients are related to those used by Wang as:
        amax = bmax = cmax = s
        aj = α(s-j) / k(n+s-1)
        bj = γ(s-j)
        cj = β(s-j)

    """

    stages = 1

    def __init__(self, pencil_length, domain):

        self.RHS = CoeffSystem(pencil_length, domain)

        # Create deque for storing recent timesteps
        N = max(self.amax, self.bmax, self.cmax)
        self._history = N
        self.dt = deque([0.]*N)

        # Create coefficient systems for multistep history
        self.MX = MX = deque()
        self.LX = LX = deque()
        self.F = F = deque()
        for j in range(self.amax):
            MX.append(CoeffSystem(pencil_length, domain))
        for j in range(self.bmax):
            LX.append(CoeffSystem(pencil_length, domain))
        for j in range(self.cmax):
            F.append(CoeffSystem(pencil_length, domain))

        # Attributes
        self._iteration = 0
        self._LHS_params = None

    def step(self, solver, dt):
        """Advance solver by one timestep."""

        # Solver references
        pencils = solver.pencils
        evaluator = solver.evaluator
        state = solver.state
        STORE_EXPANDED_MATRICES = solver.problem.STORE_EXPANDED_MATRICES

        evaluator_kw = {}
        evaluator_kw['world_time'] = world_time = solver.get_world_time()
        evaluator_kw['wall_time'] = world_time - solver.start_time
        evaluator_kw['sim_time'] = solver.sim_time
        evaluator_kw['timestep'] = dt
        evaluator_kw['iteration'] = solver.iteration

        # References
        MX = self.MX
        LX = self.LX
        F = self.F
        RHS = self.RHS

        # Cycle and compute timesteps
        self.dt.rotate()
        self.dt[0] = dt

        # Compute IMEX coefficients
        a, b, c = self.compute_coefficients(self.dt, self._iteration)
        self._iteration += 1

        # Update RHS components and LHS matrices
        MX.rotate()
        LX.rotate()
        F.rotate()

        MX0 = MX[0]
        LX0 = LX[0]
        F0 = F[0]
        a0 = a[0]
        b0 = b[0]

        # Check on updating LHS
        update_LHS = ((a0, b0) != self._LHS_params)
        self._LHS_params = (a0, b0)

        # Evaluate M.X0 and L.X0
        MX0.data.fill(0)
        LX0.data.fill(0)
        for p in pencils:
            x = state.get_pencil(p)
            fast_csr_matvec(p.M, x, MX0.get_pencil(p))
            fast_csr_matvec(p.L, x, LX0.get_pencil(p))
            if update_LHS:
                # Remove old solver reference
                p.LHS_solver = None

        # Run evaluator and compute F0
        # No need to scatter since gather occured just before step was called
        evaluator.evaluate_scheduled(**evaluator_kw)
        F0.data.fill(0)
        for p in pencils:
            fast_csr_matvec(p.pre_left, solver.F.get_pencil(p), F0.get_pencil(p))

        # Build RHS
        np.multiply(c[1], F0.data, out=RHS.data)
        for j in range(2, len(c)):
            RHS.data += c[j] * F[j-1].data  # CREATES TEMPORARY
        for j in range(1, len(a)):
            RHS.data -= a[j] * MX[j-1].data  # CREATES TEMPORARY
        for j in range(1, len(b)):
            RHS.data -= b[j] * LX[j-1].data  # CREATES TEMPORARY

        # Solve
        state.data.fill(0)
        for p in pencils:
            if update_LHS:
                if STORE_EXPANDED_MATRICES:
                    np.copyto(p.LHS.data, a0*p.M_exp.data + b0*p.L_exp.data)  # CREATES TEMPORARY
                else:
                    p.LHS = (a0*p.M + b0*p.L) @ p.pre_right
                # Remove old solver reference before building new solver
                p.LHS_solver = None
                p.LHS_solver = solver.matsolver(p.LHS, solver)
            pRHS = RHS.get_pencil(p)
            pX = p.LHS_solver.solve(pRHS)
            if p.pre_right is None:
                state.set_pencil(p, pX)
            else:
                fast_csr_matvec(p.pre_right, pX, state.get_pencil(p))

        # Update solver
        solver.sim_time += dt


@add_scheme
class CNAB1(MultistepIMEX):
    """
    1st-order Crank-Nicolson Adams-Bashforth scheme [Wang 2008 eqn 2.5.3]

    Implicit: 2nd-order Crank-Nicolson
    Explicit: 1st-order Adams-Bashforth (forward Euler)

    """

    amax = 1
    bmax = 1
    cmax = 1

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k0, *rest = timesteps

        a[0] = 1 / k0
        a[1] = -1 / k0
        b[0] = 1 / 2
        b[1] = 1 / 2
        c[1] = 1

        return a, b, c


@add_scheme
class SBDF1(MultistepIMEX):
    """
    1st-order semi-implicit BDF scheme [Wang 2008 eqn 2.6]

    Implicit: 1st-order BDF (backward Euler)
    Explicit: 1st-order extrapolation (forward Euler)

    """

    amax = 1
    bmax = 1
    cmax = 1

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k0, *rest = timesteps

        a[0] = 1 / k0
        a[1] = -1 / k0
        b[0] = 1
        c[1] = 1

        return a, b, c


@add_scheme
class CNAB2(MultistepIMEX):
    """
    2nd-order Crank-Nicolson Adams-Bashforth scheme [Wang 2008 eqn 2.9]

    Implicit: 2nd-order Crank-Nicolson
    Explicit: 2nd-order Adams-Bashforth

    """

    amax = 2
    bmax = 2
    cmax = 2

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 1:
            return CNAB1.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k0, *rest = timesteps
        w1 = k1 / k0

        a[0] = 1 / k1
        a[1] = -1 / k1
        b[0] = 1 / 2
        b[1] = 1 / 2
        c[1] = 1 + w1/2
        c[2] = -w1 / 2

        return a, b, c


@add_scheme
class MCNAB2(MultistepIMEX):
    """
    2nd-order modified Crank-Nicolson Adams-Bashforth scheme [Wang 2008 eqn 2.10]

    Implicit: 2nd-order modified Crank-Nicolson
    Explicit: 2nd-order Adams-Bashforth

    """

    amax = 2
    bmax = 2
    cmax = 2

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 1:
            return CNAB1.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k0, *rest = timesteps
        w1 = k1 / k0

        a[0] = 1 / k1
        a[1] = -1 / k1
        b[0] = (8 + 1/w1) / 16
        b[1] = (7 - 1/w1) / 16
        b[2] = 1 / 16
        c[1] = 1 + w1/2
        c[2] = -w1 / 2

        return a, b, c


@add_scheme
class SBDF2(MultistepIMEX):
    """
    2nd-order semi-implicit BDF scheme [Wang 2008 eqn 2.8]

    Implicit: 2nd-order BDF
    Explicit: 2nd-order extrapolation

    """

    amax = 2
    bmax = 2
    cmax = 2

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 1:
            return SBDF1.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k0, *rest = timesteps
        w1 = k1 / k0

        a[0] = (1 + 2*w1) / (1 + w1) / k1
        a[1] = -(1 + w1) / k1
        a[2] = w1**2 / (1 + w1) / k1
        b[0] = 1
        c[1] = 1 + w1
        c[2] = -w1

        return a, b, c


@add_scheme
class CNLF2(MultistepIMEX):
    """
    2nd-order Crank-Nicolson leap-frog scheme [Wang 2008 eqn 2.11]

    Implicit: ?-order wide Crank-Nicolson
    Explicit: 2nd-order leap-frog

    """

    amax = 2
    bmax = 2
    cmax = 2

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 1:
            return CNAB1.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k0, *rest = timesteps
        w1 = k1 / k0

        a[0] = 1 / (1 + w1) / k1
        a[1] = (w1 - 1) / k1
        a[2] = -w1**2 / (1 + w1) / k1
        b[0] = 1 / w1 / 2
        b[1] = (1 - 1/w1) / 2
        b[2] = 1 / 2
        c[1] = 1

        return a, b, c


@add_scheme
class SBDF3(MultistepIMEX):
    """
    3rd-order semi-implicit BDF scheme [Wang 2008 eqn 2.14]

    Implicit: 3rd-order BDF
    Explicit: 3rd-order extrapolation

    """

    amax = 3
    bmax = 3
    cmax = 3

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 2:
            return SBDF2.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k2, k1, k0, *rest = timesteps
        w2 = k2 / k1
        w1 = k1 / k0

        a[0] = (1 + w2/(1 + w2) + w1*w2/(1 + w1*(1 + w2))) / k2
        a[1] = (-1 - w2 - w1*w2*(1 + w2)/(1 + w1)) / k2
        a[2] = w2**2 * (w1 + 1/(1 + w2)) / k2
        a[3] = -w1**3 * w2**2 * (1 + w2) / (1 + w1) / (1 + w1 + w1*w2) / k2
        b[0] = 1
        c[1] = (1 + w2)*(1 + w1*(1 + w2)) / (1 + w1)
        c[2] = -w2*(1 + w1*(1 + w2))
        c[3] = w1*w1*w2*(1 + w2) / (1 + w1)

        return a, b, c


@add_scheme
class SBDF4(MultistepIMEX):
    """
    4th-order semi-implicit BDF scheme [Wang 2008 eqn 2.15]

    Implicit: 4th-order BDF
    Explicit: 4th-order extrapolation

    """

    amax = 4
    bmax = 4
    cmax = 4

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 3:
            return SBDF3.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k3, k2, k1, k0, *rest = timesteps
        w3 = k3 / k2
        w2 = k2 / k1
        w1 = k1 / k0

        A1 = 1 + w1*(1 + w2)
        A2 = 1 + w2*(1 + w3)
        A3 = 1 + w1*A2

        a[0] = (1 + w3/(1 + w3) + w2*w3/A2 + w1*w2*w3/A3) / k3
        a[1] = (-1 - w3*(1 + w2*(1 + w3)/(1 + w2)*(1 + w1*A2/A1))) / k3
        a[2] = w3 * (w3/(1 + w3) + w2*w3*(A3 + w1)/(1 + w1)) / k3
        a[3] = -w2**3 * w3**2 * (1 + w3) / (1 + w2) * A3 / A2 / k3
        a[4] = (1 + w3) / (1 + w1) * A2 / A1 * w1**4 * w2**3 * w3**2 / A3 / k3
        b[0] = 1
        c[1] = w2 * (1 + w3) / (1 + w2) * ((1 + w3)*(A3 + w1) + (1 + w1)/w2) / A1
        c[2] = -A2 * A3 * w3 / (1 + w1)
        c[3] = w2**2 * w3 * (1 + w3) / (1 + w2) * A3
        c[4] = -w1**3 * w2**2 * w3 * (1 + w3) / (1 + w1) * A2 / A1

        return a, b, c


class RungeKuttaIMEX:
    """
    Base class for implicit-explicit multistep methods.

    Parameters
    ----------
    pencil_length : int
        Number of fields in problem
    domain : domain object
        Problem domain

    Notes
    -----
    These timesteppers discretize the system
        M.dt(X) + L.X = F
    by constructing s stages
        M.X(n,i) - M.X(n,0) + k Hij L.X(n,j) = k Aij F(n,j)
    where j runs from {0, 0} to {i, i-1}, and F(n,i) is evaluated at time
        t(n,i) = t(n,0) + k ci

    The s stages are solved as
        (M + k Hii L).X(n,i) = M.X(n,0) + k Aij F(n,j) - k Hij L.X(n,j)
    where j runs from {0, 0} to {i-1, i-1}.

    The final stage is used as the advanced solution*:
        X(n+1,0) = X(n,s)
        t(n+1,0) = t(n,s) = t(n,0) + k

    * Equivalently the Butcher tableaus must follow
        b_im = H[s, :]
        b_ex = A[s, :]
        c[s] = 1

    References
    ----------
    U. M. Ascher, S. J. Ruuth, and R. J. Spiteri, Applied Numerical Mathematics (1997).

    """

    _history = 0

    def __init__(self, pencil_length, domain):

        self.RHS = CoeffSystem(pencil_length, domain)

        # Create coefficient systems for multistep history
        self.MX0 = CoeffSystem(pencil_length, domain)
        self.LX = LX = [CoeffSystem(pencil_length, domain) for i in range(self.stages)]
        self.F = F = [CoeffSystem(pencil_length, domain) for i in range(self.stages)]

        self._LHS_params = None

    def step(self, solver, dt):
        """Advance solver by one timestep."""

        # Solver references
        pencils = solver.pencils
        evaluator = solver.evaluator
        state = solver.state
        STORE_EXPANDED_MATRICES = solver.problem.STORE_EXPANDED_MATRICES

        evaluator_kw = {}
        evaluator_kw['world_time'] = world_time = solver.get_world_time()
        evaluator_kw['wall_time'] = world_time - solver.start_time
        evaluator_kw['sim_time'] = sim_time_0 = solver.sim_time
        evaluator_kw['timestep'] = dt
        evaluator_kw['iteration'] = solver.iteration

        # Other references
        RHS = self.RHS
        MX0 = self.MX0
        LX = self.LX
        F = self.F
        A = self.A
        H = self.H
        c = self.c
        k = dt

        # Check on updating LHS
        update_LHS = (k != self._LHS_params)
        self._LHS_params = k

        # Compute M.X(n,0)
        MX0.data.fill(0)
        for p in pencils:
            fast_csr_matvec(p.M, state.get_pencil(p), MX0.get_pencil(p))
            if update_LHS:
                # Remove old solver references
                p.LHS_solvers = [None] * (self.stages+1)

        # Compute stages
        # (M + k Hii L).X(n,i) = M.X(n,0) + k Aij F(n,j) - k Hij L.X(n,j)
        for i in range(1, self.stages+1):
            # Compute L.X(n,i-1)
            LXi = LX[i-1]
            LXi.data.fill(0)
            for p in pencils:
                fast_csr_matvec(p.L, state.get_pencil(p), LXi.get_pencil(p))

            # Compute F(n,i-1)
            evaluator_kw['sim_time'] = solver.sim_time
            if i == 1:
                # No need to scatter since gather occured just before step was called
                evaluator.evaluate_scheduled(**evaluator_kw)
            else:
                state.scatter()
                evaluator.evaluate_group('F', **evaluator_kw)
            Fi = F[i-1]
            Fi.data.fill(0)
            for p in pencils:
                fast_csr_matvec(p.pre_left, solver.F.get_pencil(p), Fi.get_pencil(p))

            # Construct RHS(n,i)
            np.copyto(RHS.data, MX0.data)
            for j in range(0, i):
                RHS.data += (k * A[i,j]) * F[j].data  # CREATES TEMPORARY
                RHS.data -= (k * H[i,j]) * LX[j].data  # CREATES TEMPORARY

            # Solve for stage
            state.data.fill(0)
            for p in pencils:
                # Construct LHS(n,i)
                if update_LHS:
                    if STORE_EXPANDED_MATRICES:
                        np.copyto(p.LHS.data, p.M_exp.data + (k*H[i,i])*p.L_exp.data)  # CREATES TEMPORARY
                    else:
                        p.LHS = (p.M + (k*H[i,i])*p.L) @ p.pre_right
                    # Remove old solver reference before building new solver
                    p.LHS_solvers[i] = None
                    p.LHS_solvers[i] = solver.matsolver(p.LHS, solver)
                pRHS = RHS.get_pencil(p)
                pX = p.LHS_solvers[i].solve(pRHS)  # CREATES TEMPORARY
                if p.pre_right is None:
                    state.set_pencil(p, pX)
                else:
                    fast_csr_matvec(p.pre_right, pX, state.get_pencil(p))
            solver.sim_time = sim_time_0 + k*c[i]


@add_scheme
class RK111(RungeKuttaIMEX):
    """1st-order 1-stage DIRK+ERK scheme [Ascher 1997 sec 2.1]"""

    stages = 1

    c = np.array([0, 1])

    A = np.array([[0, 0],
                  [1, 0]])

    H = np.array([[0, 0],
                  [0, 1]])


@add_scheme
class RK222(RungeKuttaIMEX):
    """2nd-order 2-stage DIRK+ERK scheme [Ascher 1997 sec 2.6]"""

    stages = 2

    γ = (2 - np.sqrt(2)) / 2
    δ = 1 - 1 / γ / 2

    c = np.array([0, γ, 1])

    A = np.array([[0,  0 , 0],
                  [γ,  0 , 0],
                  [δ, 1-δ, 0]])

    H = np.array([[0,  0 , 0],
                  [0,  γ , 0],
                  [0, 1-γ, γ]])


@add_scheme
class RK443(RungeKuttaIMEX):
    """3rd-order 4-stage DIRK+ERK scheme [Ascher 1997 sec 2.8]"""

    stages = 4

    c = np.array([0, 1/2, 2/3, 1/2, 1])

    A = np.array([[  0  ,   0  ,  0 ,   0 , 0],
                  [ 1/2 ,   0  ,  0 ,   0 , 0],
                  [11/18,  1/18,  0 ,   0 , 0],
                  [ 5/6 , -5/6 , 1/2,   0 , 0],
                  [ 1/4 ,  7/4 , 3/4, -7/4, 0]])

    H = np.array([[0,   0 ,   0 ,  0 ,  0 ],
                  [0,  1/2,   0 ,  0 ,  0 ],
                  [0,  1/6,  1/2,  0 ,  0 ],
                  [0, -1/2,  1/2, 1/2,  0 ],
                  [0,  3/2, -3/2, 1/2, 1/2]])


@add_scheme
class RKSMR(RungeKuttaIMEX):
    """(3-ε)-order 3rd-stage DIRK+ERK scheme [Spalart 1991 Appendix]"""

    stages = 3

    α1, α2, α3 = (29/96, -3/40, 1/6)
    β1, β2, β3 = (37/160, 5/24, 1/6)
    γ1, γ2, γ3 = (8/15, 5/12, 3/4)
    ζ2, ζ3 = (-17/60, -5/12)

    c = np.array([0, 8/15, 2/3, 1])

    A = np.array([[    0,     0,  0, 0],
                  [   γ1,     0,  0, 0],
                  [γ1+ζ2,    γ2,  0, 0],
                  [γ1+ζ2, γ2+ζ3, γ3, 0]])

    H = np.array([[ 0,     0,     0,  0],
                  [α1,    β1,     0,  0],
                  [α1, β1+α2,    β2,  0],
                  [α1, β1+α2, β2+α3, β3]])

