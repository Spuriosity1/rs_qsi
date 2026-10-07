"""Arbitrary off-diagonal terms: S^z-dressed strings, fields, and dagger().

These exercise the generic machinery -- any ``list[StringTerm]`` (ladder factors
plus diagonal S^z factors) can be fed to :func:`generate_bw`.
"""
import os
import sys

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsqsi.lattice_model import build_pyrochlore            # noqa: E402
from rsqsi.terms import (StringTerm, hop, field, zpm,        # noqa: E402
                         transverse_field)
from rsqsi.bw import generate_bw, Sz                          # noqa: E402

g = sp.Symbol('g', positive=True)


def _model():
    return build_pyrochlore(L=2)


def _a_bond(m):
    return sorted(next(b for b in m.bonds
                       if m.tetras[m.bond_tetra[b]].kind == 'A'))


def _spectator(m, a, c):
    for ti in m.site_tetras[a]:
        for x in m.tetras[ti].members:
            if x not in (a, c):
                return x
    raise RuntimeError


def test_sz_factor_bookkeeping():
    """A spectator S^z_k dressing multiplies the amplitude by (S^z_k)^2 = 1/4
    and introduces no S^z_k dependence (spin-1/2)."""
    m = _model()
    a, c = _a_bond(m)
    k = _spectator(m, a, c)

    plain = generate_bw(m, [hop(a, c, g), hop(c, a, g)], 2, frozenset((a, c)))

    T = StringTerm(factors=((a, +1), (c, -1)), coupling=g, sz_sites=(k,))
    dressed = generate_bw(m, [T, T.dagger()], 2, frozenset((a, c, k)))

    assert not sp.expand(dressed.diag).has(Sz(k))
    assert sp.simplify(sp.expand(dressed.diag) - sp.expand(plain.diag) / 4) == 0


def test_dagger():
    T = StringTerm(factors=((3, +1), (7, -1)), coupling=g, sz_sites=(5,))
    Td = T.dagger()
    assert Td.factors == ((7, +1), (3, -1))
    assert Td.sz_sites == (5,)
    assert T.net_flip == {3: +2, 7: -2}


def test_transverse_field_second_order():
    """h S^x at N=2 on a single site: diagonal response -2 h^2 / (Jzz_A + Jzz_B)
    (the site sits on one A and one B tetrahedron)."""
    m = _model()
    h = sp.Symbol('h', positive=True)
    V = transverse_field(m, h)
    j = m.sites[0]
    res = generate_bw(m, V, 2, frozenset((j,)))
    expected = -2 * h**2 / (m.jzz['A'] + m.jzz['B'])
    assert sp.simplify(sp.expand(res.diag) - expected) == 0


def test_arbitrary_list_runs():
    """A heterogeneous V (hops + field + an S^z-dressed string) is accepted."""
    m = _model()
    a, c = _a_bond(m)
    k = _spectator(m, a, c)
    V = [hop(a, c, g), hop(c, a, g)]
    V += field(a, sp.Symbol('h'))
    V += [zpm(k, a, +1, sp.Symbol('x')), zpm(k, a, -1, sp.Symbol('x'))]
    res = generate_bw(m, V, 2, frozenset((a, c, k)))
    assert res.support == tuple(sorted((a, c, k)))
