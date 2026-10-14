"""Regression against the thesis breathing-pyrochlore XXZ worked example.

Thesis (Sanders 2026), Ch.1 "Algorithm for generating terms":
  * N=2 Ising renormalisation ~ J_pm^2 / J_zz, virtual spinons on the opposite
    sublattice to the active bond.
  * N=3 ring exchange   3!(J_pm,A^3 / J_zz,B^2 + J_pm,B^3 / J_zz,A^2) O_p .

The model is built from pmpm bond operators J_pm (S+_i S-_j + h.c.) on every
pair of spins sharing a tetrahedron, with the H0 strengths set to
constraints[v] = J_zz_v / 2  (physical H0 = J_zz*sum S^z S^z = (J_zz/2) sum_t Q_t^2).
"""
import itertools

import numpy as np
import pytest
import sympy as sp

from rsqsi import lattice as lat
from rsqsi import brillouin_wigner as bw

JpmA, JpmB, JzzA, JzzB = sp.symbols('J_pm_A J_pm_B J_zz_A J_zz_B')


@pytest.fixture(scope="module")
def model():
    prim = lat.PrimitiveCell(
        sublattices=[sp.Matrix([0, 0, 0]), sp.Matrix([2, 2, 2])],
        bonds=[(0, sp.Matrix([2, 2, 2])), (0, sp.Matrix([2, -2, -2])),
               (0, sp.Matrix([-2, 2, -2])), (0, sp.Matrix([-2, -2, 2]))],
        lattice_vectors=sp.Matrix([[0, 4, 4], [4, 0, 4], [4, 4, 0]]),
    )
    m = bw.Model(lat.build(prim, (2, 2, 2), periodic=True))
    nL = len(m.links)
    lpos = {lid: i for i, lid in enumerate(m.links)}
    Jpm = {0: JpmA, 1: JpmB}                            # by tetra sublattice
    for v in m.verts:
        for a, b in itertools.combinations(sorted(m.lattice.incident[v]), 2):
            i, j = lpos[a], lpos[b]
            for si, sj in ((+1, -1), (-1, +1)):         # S+_i S-_j  and its h.c.
                od = np.zeros(nL, int); od[i] = si; od[j] = sj
                m.add_operator(Jpm[v[1]], od, np.zeros(nL, int))
    for idx, v in enumerate(m.verts):
        m.set_constraint((JzzA if v[1] == 0 else JzzB) / 2, idx)
    return m


def _channel(m, lp):
    """'A' or 'B' if every operator of the loop is a pmpm on that sublattice."""
    cs = {m.operators[a].coeff for a in lp}
    return 'A' if cs == {JpmA} else 'B' if cs == {JpmB} else None


def test_n3_ring_exchange(model):
    """Each hexagon's two channels give 3! J_pm,A^3/J_zz,B^2 and 3! J_pm,B^3/J_zz,A^2."""
    rings = {'A': [], 'B': []}
    for lp in bw.find_closed_loops(model, 3):
        comp = sum(model.operators[a].od_rep for a in lp)
        if np.count_nonzero(comp) == 6 and np.abs(comp).max() == 1:   # a hexagon O_p
            ch = _channel(model, lp)
            if ch:
                rings[ch].append(lp)
    assert rings['A'] and rings['B']

    cA = bw.calc_coeff(model, rings['A'][0])
    cB = bw.calc_coeff(model, rings['B'][0])
    assert sp.simplify(cA - sp.factorial(3) * JpmA**3 / JzzB**2) == 0
    assert sp.simplify(cB - sp.factorial(3) * JpmB**3 / JzzA**2) == 0


def test_n2_ising(model):
    """A there-and-back on an A bond gives -2 J_pm,A^2 / J_zz,B (opposite sublattice)."""
    lp = next(lp for lp in bw.find_closed_loops(model, 2)
              if _channel(model, lp) == 'A'
              and not np.any(sum(model.operators[a].od_rep for a in lp)))
    c = bw.calc_coeff(model, lp)
    assert sp.simplify(c + 2 * JpmA**2 / JzzB) == 0


# --------------------------------------------------------------------------
# Rayleigh-Schroedinger machinery
# --------------------------------------------------------------------------

def _a_bond(model):
    return next(lp for lp in bw.find_closed_loops(model, 2)
                if _channel(model, lp) == 'A'
                and not np.any(sum(model.operators[a].od_rep for a in lp)))


def _a_hexagon(model):
    return next(lp for lp in bw.find_closed_loops(model, 3)
                if _channel(model, lp) == 'A'
                and np.count_nonzero(sum(model.operators[a].od_rep for a in lp)) == 6)


def _szsz_coeff(expr):
    '''The coefficient of the single Sz_i*Sz_j monomial in a 2-spin operator.'''
    szs = sorted((s for s in expr.free_symbols if str(s).startswith('Sz_')), key=str)
    return sp.cancel(sp.expand(expr).coeff(szs[0] * szs[1]))


def test_g0_dressing_matches_thesis():
    """G0^(a) = sum_{compositions} (prod E^(n_i)) g^{j+1}, incl. the 2 E1 E2 weight."""
    E1, E2, E3, g = sp.symbols('E1 E2 E3 g')
    Ec = {1: E1, 2: E2, 3: E3}
    dress = lambda a: sp.expand(sum(w * g**p for w, p in bw._g0_dressing(a, Ec)))
    assert dress(0) == g
    assert dress(1) == E1 * g**2
    assert dress(2) == E2 * g**2 + E1**2 * g**3
    assert dress(3) == E3 * g**2 + 2 * E1 * E2 * g**3 + E1**3 * g**4
    # E1 = 0 kills the order-1 counterterm (why N=3 is pure-BW)
    assert bw._g0_dressing(1, {1: sp.Integer(0)}) == [(sp.Integer(0), 2)]


def test_walsh_n2_is_ising_operator(model):
    """walsh_amplitude resolves the N=2 diagonal into 2 J_pm,A^2/J_zz,B SzSz + identity."""
    w = bw.walsh_amplitude(model, _a_bond(model))
    assert sp.simplify(_szsz_coeff(w) - 2 * JpmA**2 / JzzB) == 0
    assert sp.simplify(bw._identity_channel(w) + JpmA**2 / (2 * JzzB)) == 0


def test_walsh_n3_is_pure_ring(model):
    """walsh_amplitude on a hexagon is the pure ring 3! J_pm,A^3/J_zz,B^2 (no Sz dressing)."""
    w = bw.walsh_amplitude(model, _a_hexagon(model))
    assert sp.simplify(w - sp.factorial(3) * JpmA**3 / JzzB**2) == 0
    assert not any(str(s).startswith('Sz_') for s in w.free_symbols)


def test_e1_vanishes(model):
    assert bw.energy_corrections(model, 1)[1] == 0


def test_e2_is_sz_free_scalar(model):
    """E^(2) is a pure scalar (identity channel) -- the degenerate-manifold guard."""
    E2 = bw.energy_corrections(model, 2)[2]
    assert not any(str(s).startswith('Sz_') for s in E2.free_symbols)
    assert E2 != 0


def test_connectivity_filter(model):
    """A hexagon is connected; a disjoint pair of there-and-backs is not."""
    assert bw.is_connected_loop(model, _a_hexagon(model))
    tabs = [lp for lp in bw.find_closed_loops(model, 2)
            if not np.any(sum(model.operators[a].od_rep for a in lp))]

    def supp(lp):
        return {i for a in lp for i in np.nonzero(model.operators[a].od_rep)[0]}

    def adjacent(sx, sy):
        return any(model.links[j] in lat.link_neighbors(model.lattice, model.links[i])
                   for i in sx for j in sy)

    for x, y in itertools.combinations(tabs, 2):
        sx, sy = supp(x), supp(y)
        if not (sx & sy) and not adjacent(sx, sy):
            assert not bw.is_connected_loop(model, tuple(sorted(x + y)))
            return
    pytest.skip("no disjoint pair found in this cell")


def test_order4_counterterm_structure(model):
    """The order-4 G0^(2)=E2 g^2 counterterm dresses the N=2 Ising by E2/J_zz,B."""
    E2 = sp.Symbol('E2')
    w = bw.walsh_amplitude(model, _a_bond(model), slot_orders=(2,),
                           Ecorr={1: sp.Integer(0), 2: E2})
    assert sp.simplify(_szsz_coeff(w) + 2 * E2 * JpmA**2 / JzzB**2) == 0
    assert sp.simplify(bw._identity_channel(w) - E2 * JpmA**2 / (2 * JzzB**2)) == 0
