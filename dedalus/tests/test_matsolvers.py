"""
Test matrix solvers.
"""

import pytest
import numpy as np
import functools
import scipy.sparse as sp
from dedalus import public as de


def bench_wrapper(test):
    @functools.wraps(test)
    def wrapper(benchmark, *args, **kw):
        benchmark.pedantic(test, args=(None,)+args, kwargs=kw)
    return wrapper


def diagonal_solver(Nx, Ny, dtype):
    # Bases and domain
    x_basis = de.Fourier('x', Nx, interval=(0, 2*np.pi))
    y_basis = de.Fourier('y', Ny, interval=(0, 2*np.pi))
    domain = de.Domain([x_basis, y_basis], grid_dtype=dtype)
    # Forcing
    F = domain.new_field(name='F')
    x, y = domain.grids()
    F['g'] = -2 * np.sin(x) * np.sin(y)
    # Problem
    problem = de.LBVP(domain, variables=['u'])
    problem.parameters['F'] = F
    problem.add_equation("dx(dx(u)) + dy(dy(u)) = F", condition="(nx != 0) or (ny != 0)")
    problem.add_equation("u = 0", condition="(nx == 0) and (ny == 0)")
    # Solver
    solver = problem.build_solver()
    return solver


def block_solver(Nx, Ny, dtype):
    # Bases and domain
    x_basis = de.Fourier('x', Nx, interval=(0, 2*np.pi))
    y_basis = de.Fourier('y', Ny, interval=(0, 2*np.pi))
    domain = de.Domain([x_basis, y_basis], grid_dtype=dtype)
    # Forcing
    F = domain.new_field(name='F')
    x, y = domain.grids()
    F['g'] = -2 * np.sin(x) * np.sin(y)
    # Problem
    problem = de.LBVP(domain, variables=['u','ux','uy','Lu'])
    problem.parameters['F'] = F
    problem.add_equation("ux - dx(u) = 0")
    problem.add_equation("uy - dy(u) = 0")
    problem.add_equation("Lu - dx(ux) - dy(uy) = 0")
    problem.add_equation("Lu = F", condition="(nx != 0) or (ny != 0)")
    problem.add_equation("u = 0", condition="(nx == 0) and (ny == 0)")
    # Solver
    solver = problem.build_solver()
    return solver


def coupled_solver(Nx, Ny, dtype):
    # Bases and domain
    x_basis = de.Fourier('x', Nx, interval=(0, 2*np.pi))
    y_basis = de.Chebyshev('y', Ny, interval=(0, 2*np.pi))
    domain = de.Domain([x_basis, y_basis], grid_dtype=dtype)
    # Forcing
    F = domain.new_field(name='F')
    x, y = domain.grids()
    F['g'] = -2 * np.sin(x) * np.sin(y)
    # Problem
    problem = de.LBVP(domain, variables=['u','ux','uy','Lu'])
    problem.parameters['F'] = F
    problem.add_equation("ux - dx(u) = 0")
    problem.add_equation("uy - dy(u) = 0")
    problem.add_equation("Lu - dx(ux) - dy(uy) = 0")
    problem.add_equation("Lu = F")
    problem.add_bc("left(u) - right(u) = 0")
    problem.add_bc("left(uy) - right(uy) = 0", condition="nx != 0")
    problem.add_bc("left(u) = 0", condition="nx == 0")
    # Solver
    solver = problem.build_solver()
    return solver


solvers = {'diagonal': diagonal_solver(8, 512, np.float64),
           'block': block_solver(8, 128, np.float64),
           'coupled': coupled_solver(8, 128, np.float64)}

ids, solvers = list(zip(*solvers.items()))

@pytest.mark.parametrize('matsolver', de.matsolvers.matsolvers.values())
@pytest.mark.parametrize('solver', solvers, ids=ids)
def test_matsolver_setup_bench(benchmark, solver, matsolver):
    # Setup new matsolver
    solver.matsolver = matsolver
    # Benchmark matsolver setup
    def setup():
        try:
            solver._setup_pencil_matsolvers()
        except ModuleNotFoundError:
            pytest.skip("Matsolver requirements not present.")
        except ValueError:
            pytest.xfail("Invalid input for matsolver.")
    if matsolver in [de.matsolvers.SparseInverse, de.matsolvers.DenseInverse]:
        benchmark.pedantic(setup, rounds=1, iterations=1)
    else:
        benchmark.pedantic(setup, rounds=10, iterations=10)


@pytest.mark.parametrize('matsolver', de.matsolvers.matsolvers.values())
@pytest.mark.parametrize('solver', solvers, ids=ids)
def test_matsolver_solve_bench(benchmark, solver, matsolver):
    # Setup new matsolver
    solver.matsolver = matsolver
    # Setup pencil matsolvers
    try:
        solver._setup_pencil_matsolvers()
    except ModuleNotFoundError:
        pytest.skip("Matsolver requirements not present.")
    except ValueError:
        pytest.xfail("Invalid input for matsolver.")
    # Benchmark solve
    benchmark.pedantic(solver.solve, rounds=10, iterations=10)
    # Check solution
    x, y = solver.domain.grids()
    u_true = np.sin(x) * np.sin(y)
    u = solver.state['u']
    assert np.allclose(u['g'], u_true)


# @pytest.mark.parametrize('loops', [1, 10])
# @pytest.mark.parametrize('matsolver', de.matsolvers.matsolvers.values())
# @pytest.mark.parametrize('solver', [block_solver(8, 128, np.float64)])
# @bench_wrapper
# def test_matsolver_block(benchmark, solver, matsolver, loops):
#     test_matsolver(benchmark, solver, matsolver, loops)


# @pytest.mark.parametrize('loops', [1, 10])
# @pytest.mark.parametrize('matsolver', de.matsolvers.matsolvers.values())
# @pytest.mark.parametrize('solver', [coupled_solver(8, 128, np.float64)])
# @bench_wrapper
# def test_matsolver_gen(benchmark, solver, matsolver, loops):
#     test_matsolver(benchmark, solver, matsolver, loops)

