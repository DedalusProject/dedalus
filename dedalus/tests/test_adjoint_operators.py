"""Test vjp and jvp routines"""

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators

# dtype_range = [np.float64]
# layout_range = ['g','c']


# ## Fourier tests
# N_range = [12]
# bounds_range = [(0, 2*np.pi)]
# test_operators = ['u**2','np.sin(u)-np.cos(u)','0.8*u**3-1.2*u**2','d3.Differentiate(u,coord)','1.2*u*d3.Differentiate(u**2+u,coord)','0.5*d3.integ(u**2)']


# @pytest.mark.parametrize('N', N_range)
# @pytest.mark.parametrize('bounds', bounds_range)
# @pytest.mark.parametrize('dtype', dtype_range)
# @pytest.mark.parametrize('layout', layout_range)
# @pytest.mark.parametrize('operator', test_operators)
# def test_fourier_jvp(N, bounds, dtype, layout, operator):
#     # Check that jvp matches symbolic and that the primals match the operator evaluation
#     c = coords.Coordinate('x')
#     d = distributor.Distributor((c,))
#     if dtype == np.float64:
#         b = basis.RealFourier(c, size=N, bounds=bounds)
#     elif dtype == np.complex128:
#         b = basis.ComplexFourier(c, size=N, bounds=bounds)
#     u   = field.Field(dist=d, name='u',  bases=(b,),dtype=dtype)
#     du  = field.Field(dist=d, name='du', bases=(b,),dtype=dtype)
#     # Must fill in 'c' to avoid Nyquist mode
#     u.fill_random(layout='c'); u.change_layout(layout)
#     du.fill_random(layout='c'); du.change_layout(layout)
#     expr = operator.strip()
#     op = eval(expr,{"u":u, "np":np, "d3":operators, "coord":c})
#     op.frechet_differential([u], [du])
#     g_eval = op.evaluate()
#     g_jvp, dg_fwd = op.evaluate_jvp({u: du})
#     dG_sym = op.frechet_differential([u], [du]).evaluate()
#     assert(np.allclose(dg_fwd[layout], dG_sym[layout]) and np.allclose(g_jvp[layout],g_eval[layout]))

# @pytest.mark.parametrize('N', N_range)
# @pytest.mark.parametrize('bounds', bounds_range)
# @pytest.mark.parametrize('dtype', dtype_range)
# @pytest.mark.parametrize('layout', layout_range)
# @pytest.mark.parametrize('operator', test_operators)
# def test_fourier_vjp(N, bounds, dtype, layout, operator):
#     # Check that vjp matches jvp
#     c = coords.Coordinate('x')
#     d = distributor.Distributor((c,))
#     if dtype == np.float64:
#         b = basis.RealFourier(c, size=N, bounds=bounds)
#     elif dtype == np.complex128:
#         b = basis.ComplexFourier(c, size=N, bounds=bounds)
#     u   = field.Field(dist=d, name='u',  bases=(b,),dtype=dtype)
#     du  = field.Field(dist=d, name='du', bases=(b,),dtype=dtype)
#     # Must fill in 'c' to avoid Nyquist mode
#     u.fill_random(layout='c'); u.change_layout(layout)
#     du.fill_random(layout='c'); du.change_layout(layout)
#     expr = operator.strip()
#     op = eval(expr,{"u": u, "np": np, "d3": operators, "coord": c})
#     g_eval = op.evaluate()
#     _, dg_fwd = op.evaluate_jvp({u: du})
#     dg = g_eval.copy_adjoint()
#     dg.fill_random(layout='c'); dg.change_layout(layout)
#     g_vjp, dg_rev = op.evaluate_vjp({op: dg}, id=np.random.randint(0, 1000000), force=True)
#     term1 = np.vdot(dg[layout],dg_fwd[layout])
#     term2 = np.vdot(dg_rev[u][layout],du[layout])
#     assert(np.allclose(term1,term2) and np.allclose(g_vjp[layout],g_eval[layout]))

# ## Jacobi tests
# # N_range = [8, 9]
# N_range = [8]
# ab_range = [-0.5, 0]
# k_range = [0, 1]
# d_range = [1,2,3]


# @pytest.mark.parametrize('N', N_range)
# @pytest.mark.parametrize('a', ab_range)
# @pytest.mark.parametrize('b_', ab_range)
# @pytest.mark.parametrize('k', k_range)
# @pytest.mark.parametrize('dtype', dtype_range)
# @pytest.mark.parametrize('d_range', d_range)
# @pytest.mark.parametrize('layout', layout_range)
# @pytest.mark.parametrize('operator', test_operators)
# def test_Jacobi_jvp(N, a, b_, k, dtype, d_range, layout, operator):
#     # Check that jvp matches symbolic and that the primals match the operator evaluation
#     c = coords.Coordinate('x')
#     d = distributor.Distributor((c,))
#     b = basis.Jacobi(c, size=N, a0=a, b0=b_, a=a+k, b=b_+k, bounds=(0, 1))
#     u   = field.Field(dist=d, name='u',  bases=(b,),dtype=dtype)
#     du  = field.Field(dist=d, name='du', bases=(b,),dtype=dtype)
#     u.fill_random(layout=layout)
#     du.fill_random(layout=layout)
#     expr = operator.strip()
#     op = eval(expr,{"u":u, "np":np, "d3":operators, "coord":c})
#     op.frechet_differential([u], [du])
#     g_eval = op.evaluate()
#     g_jvp, dg_fwd = op.evaluate_jvp({u: du})
#     dG_sym = op.frechet_differential([u], [du]).evaluate()
#     assert(np.allclose(dg_fwd[layout], dG_sym[layout]) and np.allclose(g_jvp[layout],g_eval[layout]))


# @pytest.mark.parametrize('N', N_range)
# @pytest.mark.parametrize('a', ab_range)
# @pytest.mark.parametrize('b_', ab_range)
# @pytest.mark.parametrize('k', k_range)
# @pytest.mark.parametrize('dtype', dtype_range)
# @pytest.mark.parametrize('d_range', d_range)
# @pytest.mark.parametrize('layout', layout_range)
# @pytest.mark.parametrize('operator', test_operators)
# def test_jacobi_vjp(N, a, b_, k, dtype, d_range, layout, operator, bounds=(0, 1)):
#     # Check that vjp matches jvp
#     c = coords.Coordinate('x')
#     d = distributor.Distributor((c,))
#     b = basis.Jacobi(c, size=N, a0=a, b0=b_, a=a+k, b=b_+k, bounds=(0, 1))
#     u   = field.Field(dist=d, name='u',  bases=(b,),dtype=dtype)
#     du  = field.Field(dist=d, name='du', bases=(b,),dtype=dtype)
#     u.fill_random(layout=layout)
#     du.fill_random(layout=layout)
#     expr = operator.strip()
#     op = eval(expr,{"u":u, "np":np, "d3":operators, "coord":c})
#     g_eval = op.evaluate()
#     _, dg_fwd = op.evaluate_jvp({u: du})
#     dg = g_eval.copy_adjoint()
#     dg.fill_random(layout=layout)
#     g_vjp, dg_rev = op.evaluate_vjp({op: dg}, id=np.random.randint(0, 1000000), force=True)
#     term1 = np.vdot(dg[layout],dg_fwd[layout])
#     term2 = np.vdot(dg_rev[u][layout],du[layout])
#     assert(np.allclose(term1,term2) and np.allclose(g_vjp[layout],g_eval[layout]))

# ## Multidimensional tests
# @pytest.mark.parametrize('dtype', dtype_range)
# @pytest.mark.parametrize('layout', layout_range)
# def test_vjp_multidimension(dtype, layout):
#     # Test vjp for two bases
#     c = coords.CartesianCoordinates('x','y')
#     d = distributor.Distributor((c,))
#     if dtype == np.float64:
#         xbasis = basis.RealFourier(c['x'], size=12, bounds=(0,2*np.pi))
#     elif dtype == np.complex128:
#         xbasis = basis.ComplexFourier(c['x'], size=12, bounds=(0,2*np.pi))
#     ybasis = basis.Chebyshev(c['y'], size=14, bounds=(0,2))
#     u   = field.Field(dist=d, name='u',  bases=(xbasis,ybasis), dtype=dtype)
#     du  = field.Field(dist=d, name='du', bases=(xbasis,ybasis), dtype=dtype)
#     v   = field.Field(dist=d, name='v',  bases=(xbasis,ybasis), dtype=dtype)
#     dv  = field.Field(dist=d, name='dv', bases=(xbasis,ybasis), dtype=dtype)
#     u.fill_random(layout='c'); u.change_layout(layout)
#     du.fill_random(layout='c'); du.change_layout(layout)
#     v.fill_random(layout='c'); v.change_layout(layout)
#     dv.fill_random(layout='c'); dv.change_layout(layout)
#     dx = lambda A: operators.Differentiate(A, c['x'])
#     dy = lambda A: operators.Differentiate(A, c['y'])
#     op = u*dx(u) + v*dy(u)
#     g_eval = op.evaluate()
#     _, dg_fwd = op.evaluate_jvp({u: du, v: dv})
#     dg = g_eval.copy_adjoint()
#     dg.fill_random(layout='c'); dg.change_layout(layout)
#     _, dg_rev = op.evaluate_vjp({op: dg}, id=np.random.randint(0, 1000000), force=True)
#     term1 = np.vdot(dg[layout],dg_fwd[layout])
#     term2 = np.vdot(dg_rev[u][layout],du[layout]) + np.vdot(dg_rev[v][layout],dv[layout])
#     assert(np.allclose(term1,term2))

# @pytest.mark.parametrize('dtype', dtype_range)
# @pytest.mark.parametrize('layout', layout_range)
# def test_vjp_accumulate(dtype, layout):
#     # Test that cotangents accumulate correctly
#     c = coords.CartesianCoordinates('x','y')
#     d = distributor.Distributor((c,))
#     if dtype == np.float64:
#         xbasis = basis.RealFourier(c['x'], size=12, bounds=(0,2*np.pi))
#     elif dtype == np.complex128:
#         xbasis = basis.ComplexFourier(c['x'], size=12, bounds=(0,2*np.pi))
#     ybasis = basis.Chebyshev(c['y'], size=14, bounds=(0,2))
#     u   = field.Field(dist=d, name='u',  bases=(xbasis,ybasis), dtype=dtype)
#     du  = field.Field(dist=d, name='du', bases=(xbasis,ybasis), dtype=dtype)
#     v   = field.Field(dist=d, name='v',  bases=(xbasis,ybasis), dtype=dtype)
#     dv  = field.Field(dist=d, name='dv', bases=(xbasis,ybasis), dtype=dtype)
#     u.fill_random(layout='c'); u.change_layout(layout)
#     du.fill_random(layout='c'); du.change_layout(layout)
#     v.fill_random(layout='c'); v.change_layout(layout)
#     dv.fill_random(layout='c'); dv.change_layout(layout)
#     dx = lambda A: operators.Differentiate(A, c['x'])
#     dy = lambda A: operators.Differentiate(A, c['y'])
#     op1 = u*dx(u) + v*dy(u)
#     op2 = u*dx(v) + v*dy(v)
#     op3 = dx(u) + dy(v)
#     op4 = -0.5*operators.integ(u**2+v**2)
#     op_list = [op1,op2,op3,op4]
#     cotangents = {}
#     term1 = 0
#     for op in op_list:
#         g_eval = op.evaluate()
#         _, dg_fwd = op.evaluate_jvp({u: du, v: dv})
#         dg = g_eval.copy_adjoint()
#         dg.fill_random(layout='c'); dg.change_layout(layout)
#         cotangents[op] = dg
#         _, cotangents = op.evaluate_vjp(cotangents, id=np.random.randint(0, 1000000), force=True)
#         term1 += np.vdot(dg[layout],dg_fwd[layout])
#     term2 = np.vdot(cotangents[u][layout],du[layout]) + np.vdot(cotangents[v][layout],dv[layout])
#     assert(np.allclose(term1,term2))

def test_S2_operators():
    """Tests"""
    lmax = 7
    Ntheta = lmax+1
    Nphi = 2*(lmax+1)
    c = coords.S2Coordinates('phi', 'theta')
    dist = distributor.Distributor(c, dtype=np.float64)
    sphere = basis.SphereBasis(c, shape=(Nphi, Ntheta), radius=1.4, dealias=1, dtype=np.float64)
    f  = dist.VectorField(c, name='f', bases=sphere)
    f.fill_random(layout='c')
    op = operators.Gradient(f)
    cotangents = {}
    df = f.copy()
    g_eval = op.evaluate()
    _, dg_fwd = op.evaluate_jvp({f: df})
    dg = g_eval.copy_adjoint()
    dg.fill_random(layout='c'); dg.change_layout('c')
    cotangents[op] = dg
    _, cotangents = op.evaluate_vjp(cotangents, id=np.random.randint(0, 1000000), force=True)
    term1 = np.vdot(dg['c'],dg_fwd['c'])
    term2 = np.vdot(cotangents[f]['c'],df['c'])
    assert(np.allclose(term1,term2))


# def test_S2_operators():
#     """Tests"""
#     lmax = 7
#     Ntheta = lmax+1
#     Nphi = 2*(lmax+1)
#     Nr = 8
#     c = coords.SphericalCoordinates('phi', 'theta', 'r')
#     dist = distributor.Distributor(c, dtype=np.float64)
#     sphere = basis.BallBasis(c, shape=(Nphi, Ntheta, Nr), radius=1.4, dealias=1, dtype=np.float64, k=1)
#     f  = dist.VectorField(c, name='f', bases=sphere)
#     op = operators.Divergence(f)
#     cotangents = {}
#     df = f.copy()
#     g_eval = op.evaluate()
#     _, dg_fwd = op.evaluate_jvp({f: df})
#     dg = g_eval.copy_adjoint()
#     dg.fill_random(layout='c'); dg.change_layout('c')
#     cotangents[op] = dg
#     _, cotangents = op.evaluate_vjp(cotangents, id=np.random.randint(0, 1000000), force=True)
#     term1 = np.vdot(dg['c'],dg_fwd['c'])
#     term2 = np.vdot(cotangents[f]['c'],df['c'])
#     assert(np.allclose(term1,term2))

