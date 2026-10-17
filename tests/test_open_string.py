"""Regression for open strings on odd-order (half-integer "zero point") vertices.

Thesis (Sanders 2026), Ch.4: a 1-/3-member constrainer has a two-fold ground state
C = +-1/2, so an effective operator may be an *open string* beginning and ending
there, flipping the zero point at no energy cost while passing through gapped
spinons on the even interior vertices.

Worked example: the open transverse-field Ising chain (spins on links, field
V = Gamma sum_i (S+_i + S-_i), constraint strength J/2). An N-site chain has two
odd (order-1) end 0-cells and even (order-2) interior ones; the ferromagnetic
ground manifold is two-fold and the leading connecting operator flips every spin at
order N-1. The engine returns, for that full-flip channel,

    (-1)^N * 4^(N-2) * Gamma^(N-1) / J^(N-2).

N=3 is checked by hand: two orderings, each one J/2 domain-wall denominator
g = -2/J, times Gamma^2 -> -4 Gamma^2 / J.
"""
import numpy as np
import pytest
import sympy as sp

from rsqsi import lattice as lat
from rsqsi import brillouin_wigner as bw

Gamma, J = sp.symbols("Gamma J", positive=True)


def _chain(N):
    cell = lat.PrimitiveCell(
        sublattices=[sp.Matrix([0])],
        bonds=[(0, sp.Matrix([1]))],
        lattice_vectors=sp.Matrix([[1]]),
    )
    m = bw.Model(lat.build(cell, (N,), periodic=False))
    nL = len(m.links)
    for i in range(nL):
        for s in (+1, -1):
            od = np.zeros(nL, int); od[i] = s
            m.add_operator(Gamma, od, np.zeros(nL, int))
    for idx, _ in enumerate(m.verts):
        m.set_constraint(J / 2, idx)
    return m


def _connecting_amplitude(m, order):
    """Sum of walsh amplitudes over the full-flip (all-+1) open-string channel."""
    key = (1,) * len(m.links)
    amp = sp.Integer(0)
    for lp in bw.nonvanishing_strings(m, order):
        if bw.is_connected_loop(m, lp) and bw._net_flip(m, lp) == key:
            amp += bw.walsh_amplitude(m, lp)
    return sp.simplify(amp)


def test_odd_order_detection():
    """End 0-cells are 1-member (odd); interior are 2-member (even)."""
    m = _chain(4)
    orders = [len(m.lattice.incident[v]) for v in m.verts]
    assert orders == [1, 2, 2, 1]
    assert m.odd == [True, False, False, True]


def test_open_string_is_admissible():
    """find_closed_loops returns the connecting open string (nonzero net flip)
    even though it is *not* a closed 1-cycle -- it terminates on the odd ends."""
    m = _chain(3)
    loops = [lp for lp in bw.nonvanishing_strings(m, 2) if bw.is_connected_loop(m, lp)]
    flips = {bw._net_flip(m, lp) for lp in loops}
    assert (1, 1) in flips and (-1, -1) in flips      # open strings, boundary != 0


def test_n3_chain_coefficient():
    """Hand target: the N=3 full-flip open string is -4 Gamma^2 / J."""
    m = _chain(3)
    assert sp.simplify(_connecting_amplitude(m, 2) + 4 * Gamma**2 / J) == 0


@pytest.mark.parametrize("N", [2, 3, 4, 5])
def test_chain_coefficient_pattern(N):
    """Full-flip open string = (-1)^N 4^(N-2) Gamma^(N-1) / J^(N-2)."""
    m = _chain(N)
    expected = (-1)**N * 4**(N - 2) * Gamma**(N - 1) / J**(N - 2)
    assert sp.simplify(_connecting_amplitude(m, N - 1) - expected) == 0


def test_hermiticity():
    """The +flip and -flip channels (Hermitian conjugates) share a real coefficient."""
    m = _chain(4)
    plus = _connecting_amplitude(m, 3)
    minus = sp.Integer(0)
    key = (-1,) * len(m.links)
    for lp in bw.nonvanishing_strings(m, 3):
        if bw.is_connected_loop(m, lp) and bw._net_flip(m, lp) == key:
            minus += bw.walsh_amplitude(m, lp)
    assert sp.simplify(plus - minus) == 0


def test_second_order_ground_energy():
    """RS E^(2): two flippable spins, each -2 Gamma^2 / J -> -4 Gamma^2 / J."""
    m = _chain(3)
    Ecorr = bw.energy_corrections(m, 2)
    assert sp.simplify(Ecorr[2] + 4 * Gamma**2 / J) == 0


def test_calc_coeff_rejects_odd_loops():
    """The scalar path is background-independent and cannot represent an
    operator-valued open string: it must refuse an odd-vertex loop."""
    m = _chain(3)
    lp = next(lp for lp in bw.nonvanishing_strings(m, 2)
              if bw.is_connected_loop(m, lp) and any(bw._net_flip(m, lp)))
    with pytest.raises(NotImplementedError):
        bw.calc_coeff(m, lp)
