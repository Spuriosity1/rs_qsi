"""Validation against the thesis breathing-pyrochlore XXZ worked example.

Thesis (Sanders 2026), Ch.1 "Algorithm for generating terms":
  * N=2 nearest-neighbour Ising renormalisation proportional to J_pm^2 / J_zz,
    with the virtual spinons living on the *opposite* sublattice to the bond.
  * N=3 ring exchange with coefficient 3! J_pm,B^3/J_zz,A^2 + 3! J_pm,A^3/J_zz,B^2.
"""
import os
import sys

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsqsi.lattice_model import build_pyrochlore          # noqa: E402
from rsqsi.terms import xxz_perturbation                  # noqa: E402
from rsqsi.bw import generate_bw, Sz                       # noqa: E402

JpA, JpB, JzA, JzB = sp.symbols('J_pm_A J_pm_B J_zz_A J_zz_B', positive=True)


def _model():
    m = build_pyrochlore(L=2)
    return m, xxz_perturbation(m)


def test_n3_ring_exchange_coefficient():
    """O_p coefficient = 3! J_pm,A^3/J_zz,B^2 + 3! J_pm,B^3/J_zz,A^2."""
    m, V = _model()
    res = generate_bw(m, V, N=3, support=frozenset(m.rings[0].members))
    expected = 6 * (JpA**3 / JzB**2 + JpB**3 / JzA**2)
    assert sp.simplify(res.ring_coefficient() - expected) == 0


def test_n3_two_conjugate_ring_channels():
    """Exactly two full-ring channels (O_p and O_p^dagger), equal coefficients."""
    m, V = _model()
    res = generate_bw(m, V, N=3, support=frozenset(m.rings[0].members))
    full = frozenset(res.support)
    coeffs = [sp.simplify(c) for f, c in res.offdiag.items()
              if frozenset(s for s, _ in f) == full]
    assert len(coeffs) == 2
    assert sp.simplify(coeffs[0] - coeffs[1]) == 0


def test_n2_ising_A_bond():
    """A-bond Ising coeff = 2 J_pm,A^2 / J_zz,B (virtual spinons on B)."""
    m, V = _model()
    bond = next(b for b in m.bonds if m.tetras[m.bond_tetra[b]].kind == 'A')
    a, c = tuple(bond)
    res = generate_bw(m, V, N=2, support=bond)
    diag = sp.expand(res.diag)
    assert sp.simplify(diag.coeff(Sz(a) * Sz(c)) - 2 * JpA**2 / JzB) == 0
    assert len(res.offdiag) == 0  # purely diagonal at this order


def test_n2_ising_B_bond():
    """B-bond Ising coeff = 2 J_pm,B^2 / J_zz,A (A/B swapped)."""
    m, V = _model()
    bond = next(b for b in m.bonds if m.tetras[m.bond_tetra[b]].kind == 'B')
    a, c = tuple(bond)
    res = generate_bw(m, V, N=2, support=bond)
    diag = sp.expand(res.diag)
    assert sp.simplify(diag.coeff(Sz(a) * Sz(c)) - 2 * JpB**2 / JzA) == 0


def test_reference_state_is_ice():
    m, _ = _model()
    assert m.is_ice(m.ref)
    assert m.spinon_energy(m.ref) == 0
