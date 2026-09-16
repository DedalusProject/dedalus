"""Adjoint identities <y, E x> = <E^H y, x> on a spherical shell for the operators a shell eigenvalue
problem uses in its boundary conditions and background terms: the S2 component selectors
(radial / angular of a surface trace), the surface trace itself (ShellRadialInterpolate, whose
adjoint carries the regularity recombination), the spherical transpose, and the cross product with
an operand on the meridional sub-basis (an axisymmetric background field), which requires the
cotangent to be reduced over the broadcast azimuthal axis.  Each case is checked on two legs:
the harness JVP (evaluate_jvp) and the explicit forward operator on the tangent field."""

import pytest
import numpy as np
import dedalus.public as d3


dtype = np.complex128


def make_problem(shape=(8, 24, 12), radii=(0.7, 1.0), seed=1234):
    coords = d3.SphericalCoordinates('phi', 'theta', 'r')
    dist = d3.Distributor(coords, dtype=dtype)
    shell = d3.ShellBasis(coords, shape=shape, radii=radii, dtype=dtype, dealias=1)
    rng = np.random.default_rng(seed)
    return coords, dist, shell, rng


def rand_field(f, rng, layout='c'):
    f.fill_random(layout=layout, seed=int(rng.integers(1 << 30)))
    f.change_layout(layout)


def adjoint_identity(build, tangent_fields, rng, zero_tangent_fields=(), layout='c'):
    """Relative errors of <y, E x> = <E^H y, x> on the explicit-forward leg and the JVP leg.
    build(**subs) -> operator expression with the named fields replaced by subs."""
    op = build()
    tangents = {}
    for f in tangent_fields:
        df = f.copy(); df.name = 'd' + f.name; rand_field(df, rng, layout); tangents[f] = df
    for f in zero_tangent_fields:
        df = f.copy(); df.name = 'd' + f.name; df['c'] = 0; tangents[f] = df
    g_eval = op.evaluate()
    g_jvp, dg_fwd = op.evaluate_jvp(tangents)
    Edx = None
    for f in tangent_fields:
        e = build(**{f.name: tangents[f]}).evaluate()
        Edx = e[layout].copy() if Edx is None else Edx + e[layout]
    dg = g_eval.get_cotangent(); rand_field(dg, rng, layout)
    _, cot = op.evaluate_vjp({op: dg}, id=int(rng.integers(1_000_000)), force=True)
    term1_fwd = np.vdot(dg[layout], Edx)
    term1_jvp = np.vdot(dg[layout], dg_fwd[layout])
    term2 = sum(np.vdot(cot[f][layout], tangents[f][layout]) for f in tangent_fields)
    rel_fwd = abs(term1_fwd - term2) / max(abs(term1_fwd), abs(term2), 1e-300)
    rel_jvp = abs(term1_jvp - term2) / max(abs(term1_jvp), abs(term2), 1e-300)
    return rel_fwd, rel_jvp


def fields():
    coords, dist, shell, rng = make_problem()
    u = dist.VectorField(coords, name='u', bases=shell)
    T = dist.TensorField((coords, coords), name='T', bases=shell)
    B0m = dist.VectorField(coords, name='B0m', bases=shell.meridional_basis)
    for f in (u, T, B0m):
        rand_field(f, rng)
    return u, T, B0m, rng


@pytest.mark.parametrize('case', ['u(r=1)', 'radial(u(r=1))', 'angular(u(r=1))', 'radial(u(r=0.7))',
                                  'angular(grad(u)(r=1))', 'radial(transpose(grad(u))(r=1))'])
def test_shell_surface_components(case):
    u, T, B0m, rng = fields()
    builds = {
        'u(r=1)': lambda u=u: u(r=1.0),
        'radial(u(r=1))': lambda u=u: d3.radial(u(r=1.0)),
        'angular(u(r=1))': lambda u=u: d3.angular(u(r=1.0)),
        'radial(u(r=0.7))': lambda u=u: d3.radial(u(r=0.7)),
        'angular(grad(u)(r=1))': lambda u=u: d3.angular(d3.grad(u)(r=1.0)),
        'radial(transpose(grad(u))(r=1))': lambda u=u: d3.radial(d3.transpose(d3.grad(u))(r=1.0)),
    }
    rel_fwd, rel_jvp = adjoint_identity(builds[case], [u], rng)
    assert rel_fwd < 1e-10
    assert rel_jvp < 1e-10


@pytest.mark.parametrize('case', ['transpose(grad(u))', 'transpose(T)'])
def test_shell_transpose(case):
    u, T, B0m, rng = fields()
    builds = {'transpose(grad(u))': lambda u=u: d3.transpose(d3.grad(u)),
              'transpose(T)': lambda T=T: d3.transpose(T)}
    args = [u] if case == 'transpose(grad(u))' else [T]
    rel_fwd, rel_jvp = adjoint_identity(builds[case], args, rng)
    assert rel_fwd < 1e-10
    assert rel_jvp < 1e-10


def test_shell_transpose_grid_layout():
    u, T, B0m, rng = fields()
    Tg = T.copy(); Tg.name = 'Tg'; Tg.change_layout('g')
    rel_fwd, rel_jvp = adjoint_identity(lambda Tg=Tg: d3.transpose(Tg), [Tg], rng, layout='g')
    assert rel_fwd < 1e-10
    assert rel_jvp < 1e-10


@pytest.mark.parametrize('case', ['du', 'dB0', 'both', 'reversed', 'curl'])
def test_shell_cross_subbasis(case):
    """Cross product with an operand on the meridional sub-basis: the cotangent of the sub-basis
    operand must be reduced over the broadcast azimuthal axis."""
    u, T, B0m, rng = fields()
    if case == 'du':
        r = adjoint_identity(lambda u=u, B0m=B0m: d3.cross(u, B0m), [u], rng, zero_tangent_fields=[B0m])
    elif case == 'dB0':
        r = adjoint_identity(lambda u=u, B0m=B0m: d3.cross(u, B0m), [B0m], rng, zero_tangent_fields=[u])
    elif case == 'both':
        r = adjoint_identity(lambda u=u, B0m=B0m: d3.cross(u, B0m), [u, B0m], rng)
    elif case == 'reversed':
        r = adjoint_identity(lambda u=u, B0m=B0m: d3.cross(B0m, u), [u, B0m], rng)
    else:
        r = adjoint_identity(lambda u=u, B0m=B0m: d3.curl(d3.cross(u, B0m)), [u, B0m], rng)
    assert r[0] < 1e-10
    assert r[1] < 1e-10
