"""Stringent cross-check: exact diagonalisation vs the perturbative effective H.

The engine's effective Hamiltonian, truncated at order `n`, must reproduce the lowest
`dim(manifold)` exact eigenvalues of the microscopic `H = H0 + V` with error
`O(J_pm^(n+1))` -- *where the perturbation theory is complete*. We verify the
convergence rate by sweeping the hop scale `lam = J_pm` and fitting the log-log slope of
the max eigenvalue deviation.

* Even conventional cubic cell (`PVP = 0`): complete at all orders; slopes ~3 (order<=2)
  and ~4 (order<=3), uniform and disordered.
* Odd complete-tetra + two-triangles cell (`PVP != 0`, docs ch.4): the engine's
  first-order term equals the exact `PVP` (bit-exact), and order<=2 converges (slope ~3).
  Order>=3 is *incomplete* for `PVP != 0` -- the scalar RS counterterms omit the
  operator-valued `E^(1)` feedback -- so the order<=3 rate is capped (documented xfail).

Marked `slow` (several ED solves + order-3 engine passes); run with `pytest -m slow`.
"""
import numpy as np
import pytest

from rsqsi import brillouin_wigner as bw
import ed_bridge as edb

pytestmark = pytest.mark.slow

LAMS = (0.05, 0.02, 0.01, 0.005)        # hop scales; slopes are fit over these
_LOGL = np.log(LAMS)


# Model of error scaling: error term between ED and eff. ham should scale like \lambda^(order +1)
def _slope(errs):
    return float(np.polyfit(_LOGL, np.log(errs), 1)[0])

# --------------------------------------------------------------------------
# Even cubic cell:  PVP = 0, complete at all orders
# --------------------------------------------------------------------------

def _even_convergence(jzz, jpm):
    model = edb.build_cubic_model(jzz, jpm)
    states, idx = edb.ground_states(model)          # == ice states for an even model
    D = len(states)
    H0, V = edb.exact_sector_parts(model)
    M2 = edb.effective_order_matrix(model, 2, states, idx)
    M3 = edb.effective_order_matrix(model, 3, states, idx)
    err2, err3 = [], []
    for lam in LAMS:
        exact = edb.lowest_eigenvalues(H0 + lam * V, D)
        err2.append(float(np.max(np.abs(
            exact - np.sort(np.linalg.eigvalsh(lam**2 * M2))))))
        err3.append(float(np.max(np.abs(
            exact - np.sort(np.linalg.eigvalsh(lam**2 * M2 + lam**3 * M3))))))
    return {"D": D, "M2": M2, "M3": M3, "err2": err2, "err3": err3,
            "slope2": _slope(err2), "slope3": _slope(err3)}


@pytest.fixture(scope="module")
def uniform():
    return _even_convergence(1.0, 1.0)


@pytest.fixture(scope="module")
def bond_disordered():
    # 16 site pyrochlore XXZ with random couplings
    rng = np.random.default_rng(1234)
    verts = edb.cubic_vertices()
    jzz = {v: float(rng.uniform(0.7, 1.3)) for v in verts}
    jpm = {v: float(rng.uniform(0.5, 1.5)) for v in verts}       # shape at lam = 1
    return _even_convergence(jzz, jpm)


def test_cubic_cell_is_clean():
    """Precondition for the clean even rate: the cubic cell has all-even vertices and no
    parallel bonds, so PVP = 0 (no order-1 ice-preserving hop); plus the expected size."""
    m = edb.build_cubic_model(1.0, 1.0)
    assert len(m.links) == 16 and len(m.verts) == 8
    assert all(len(m.lattice.incident[v]) == 4 for v in m.verts)
    states, _ = edb.ground_states(m)
    assert len(states) == 90
    assert bw.nonvanishing_strings(m, 1) == []          # PVP = 0


def test_effective_hamiltonian_hermitian(uniform):
    assert np.max(np.abs(uniform["M2"] - uniform["M2"].T)) < 1e-10
    assert np.max(np.abs(uniform["M3"] - uniform["M3"].T)) < 1e-10


def test_uniform_convergence(uniform):
    assert 2.6 < uniform["slope2"] < 3.4
    assert 3.6 < uniform["slope3"] < 4.4
    assert uniform["err3"][-1] < uniform["err2"][-1] < 1e-2


def test_disordered_convergence(disordered):
    assert 2.6 < disordered["slope2"] < 3.4
    assert 3.6 < disordered["slope3"] < 4.4


# --------------------------------------------------------------------------
# Odd complete-tetra + two-triangles cell:  PVP != 0 (docs ch.4)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def odd():
    model = edb.build_tetra_triangles(jpm=1.0, jzz=1.0)
    states, idx = edb.ground_states(model)
    D = len(states)
    H0, V = edb.exact_dense_parts(model)
    E0 = edb.ground_energy(model, states)
    pvp = edb.exact_pvp(model, states, idx)
    M = {k: edb.effective_order_matrix(model, k, states, idx) for k in (1, 2, 3)}
    err = {1: [], 2: [], 3: []}
    for lam in LAMS:
        exact = np.sort(np.linalg.eigvalsh(H0 + lam * V))[:D] - E0
        for n in (1, 2, 3):
            heff = sum(lam**k * M[k] for k in range(1, n + 1))
            err[n].append(float(np.max(np.abs(exact - np.sort(np.linalg.eigvalsh(heff))))))
    return {"D": D, "M1": M[1], "pvp": pvp,
            "slope": {n: _slope(err[n]) for n in (1, 2, 3)}}


def test_odd_cell_has_nonzero_pvp():
    """The odd (triangle) geometry really is first-order active: it has order-3 vertices
    and an ice-preserving single hop (a non-empty order-1 loop set)."""
    m = edb.build_tetra_triangles()
    assert any(len(m.lattice.incident[v]) == 3 for v in m.verts)   # triangles present
    assert bw.nonvanishing_strings(m, 1)                              # PVP != 0


def test_odd_first_order_is_exact_pvp(odd):
    """The new order-1 term is exactly the projection of V onto the ground manifold --
    the projector-dressed open-string H^(1) of docs ch.4."""
    assert np.max(np.abs(odd["pvp"])) > 0                          # PVP genuinely nonzero
    assert np.max(np.abs(odd["M1"] - odd["pvp"])) < 1e-12


def test_odd_convergence_through_order2(odd):
    """Where the series is complete (orders 1 and 2), the rate is correct: slope ~2 at
    order<=1 and ~3 at order<=2."""
    assert 1.7 < odd["slope"][1] < 2.3
    assert 2.6 < odd["slope"][2] < 3.4


@pytest.mark.xfail(strict=True, reason="order>=3 incomplete for PVP!=0: the scalar RS "
                   "counterterms omit the operator-valued E^(1) (van-Vleck) feedback. "
                   "If this starts passing, that limitation has been fixed.")
def test_odd_order3_reaches_slope4(odd):
    assert odd["slope"][3] > 3.6
