"""Exact-diagonalisation bridge for cross-checking the perturbation engine.

The engine produces a perturbative effective Hamiltonian on the ice / ground manifold.
Its spectrum must reproduce the lowest `dim(manifold)` eigenvalues of the *exact*
microscopic `H = H0 + V` it expands, with error `O(J_pm^(n+1))` when truncated at
order `n` -- provided the perturbation theory is complete to that order.

Two geometries are used, both reconstructed from a single rsqsi `Model` (so spin<->link
indexing is automatically consistent, no cross-codebase matching):

* the **even** conventional cubic cell (`CUBIC_S`): 16 spins, 8 order-4 tetrahedra,
  90 ice states, no parallel bonds -> `PVP = 0`. The series is complete at every order;
  convergence is clean (slope n+1).
* an **odd** complete-tetra + two-triangles cell (`build_tetra_triangles`, docs ch.4
  line 680): order-3 vertices carry a +-1/2 zero point, so a single hop can preserve
  the ground manifold -> `PVP != 0`, with a genuine first-order effective term.

Conventions (matching the drivers): spins on links (1-cells), tetrahedra on vertices
(0-cells); `sigma = 2 Sz in {+-1}`; constraint strength = J_zz/2, so
`H0 = sum_v (J_zz_v/2) (sum_l incidence[v,l] Sz_l)^2`, a hop `S+_i S-_j` carries J_pm.

`scipy` is used here only (sparse ED); it is a test-time dependency, not a package one.
"""
import itertools

import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import sympy as sp

from rsqsi import lattice as lat
from rsqsi import brillouin_wigner as bw

# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------

DIAMOND = lat.PrimitiveCell(
    sublattices=[sp.Matrix([0, 0, 0]), sp.Matrix([2, 2, 2])],
    bonds=[(0, sp.Matrix([2, 2, 2])), (0, sp.Matrix([2, -2, -2])),
           (0, sp.Matrix([-2, 2, -2])), (0, sp.Matrix([-2, -2, 2]))],
    lattice_vectors=sp.Matrix([[0, 4, 4], [4, 0, 4], [4, 4, 0]]),
)
CUBIC_S = sp.Matrix([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])      # |det| = 4 -> 16 spins


def _per_vertex(coupling, v):
    """`coupling[v]` if a dict, else the scalar `coupling` (uniform)."""
    return coupling[v] if isinstance(coupling, dict) else coupling


def _decorate(m, jzz, jpm):
    """Add the two hops `S+_iS-_j`, `S-_iS+_j` with amplitude `jpm[v]` for every pair
    of spins sharing a tetra `v`, and set each vertex constraint to `jzz[v]/2`."""
    nL = len(m.links)
    lpos = {lid: i for i, lid in enumerate(m.links)}
    for v in m.verts:
        for a, b in itertools.combinations(sorted(m.lattice.incident[v], key=str), 2):
            i, j = lpos[a], lpos[b]
            for si, sj in ((+1, -1), (-1, +1)):
                od = np.zeros(nL, int); od[i] = si; od[j] = sj
                m.add_operator(_per_vertex(jpm, v), od, np.zeros(nL, int))
    for idx, v in enumerate(m.verts):
        m.set_constraint(sp.Rational(1, 2) * _per_vertex(jzz, v), idx)
    return m


def cubic_vertices() -> list:
    """The vertex (tetrahedron) ids of the cubic cell, for building coupling dicts."""
    return list(lat.build(DIAMOND, CUBIC_S, periodic=True).verts)


def build_cubic_model(jzz, jpm) -> bw.Model:
    """Even cubic-cell XXZ `Model`; `jzz`/`jpm` are scalars (uniform) or per-vertex dicts."""
    return _decorate(bw.Model(lat.build(DIAMOND, CUBIC_S, periodic=True)), jzz, jpm)


def build_tetra_triangles(jpm=1.0, jzz=1.0) -> bw.Model:
    """A complete tetrahedron A (order 4) corner-sharing two triangles B, C (order 3),
    the docs ch.4 first-order example. Links 1 = B->A and 2 = C->A are the hopping pair
    (both head at A, so A stays in the ice manifold); A also carries s, t and the
    triangles carry 3,4 and 5,6. The dangling ends are order-1 (also odd)."""
    vids = ["A", "B", "C", "Ds", "Dt", "D3", "D4", "D5", "D6"]
    verts = {k: lat.Vertex(k, 0, sp.Matrix([0, 0, 0])) for k in vids}
    links = {1: lat.Link(1, "B", "A"), 2: lat.Link(2, "C", "A"),
             "s": lat.Link("s", "A", "Ds"), "t": lat.Link("t", "A", "Dt"),
             3: lat.Link(3, "B", "D3"), 4: lat.Link(4, "B", "D4"),
             5: lat.Link(5, "C", "D5"), 6: lat.Link(6, "C", "D6")}
    incident = {k: set() for k in verts}
    for lid, l in links.items():
        incident[l.tail].add(lid); incident[l.head].add(lid)
    return _decorate(bw.Model(lat.Lattice(verts, links, incident)), jzz, jpm)


# ---------------------------------------------------------------------------
# Ground manifold and the effective matrix
# ---------------------------------------------------------------------------

def ground_states(model: bw.Model):
    """The classical ground manifold as sigma vectors (sigma = 2Sz in {+-1}, link order)
    with a {sigma-tuple -> index} map. A state is a ground state iff every vertex sits at
    its minimum-|charge|: even vertices `incidence @ sigma = 0`, odd vertices `= +-1`."""
    inc = np.array(model.incidence.tolist(), dtype=int)
    odd = np.array(model.odd, dtype=bool)
    nL = len(model.links)
    states = []
    for t in itertools.product((-1, 1), repeat=nL):
        q = inc @ np.array(t, dtype=int)
        if not q[~odd].any() and np.all(np.abs(q[odd]) == 1):
            states.append(np.array(t, dtype=int))
    idx = {tuple(s): i for i, s in enumerate(states)}
    return states, idx


def effective_order_matrix(model: bw.Model, order: int, states, idx) -> np.ndarray:
    """The homogeneous order-`order` effective Hamiltonian as a numeric matrix on the
    ground basis (one `rs_effective_terms(model, order)` pass). The full truncated
    effective Hamiltonian is the sum over order = (1 or 2)..n; since an order-k term
    carries exactly k hop amplitudes, scaling the hops by `lam` scales this by `lam**k`,
    so each is built once (at lam = 1)."""
    nL = len(model.links)
    D = len(states)
    H = np.zeros((D, D))
    for key, poly in bw.rs_effective_terms(model, order).items():
        key = np.array(key, dtype=int)
        flips = np.nonzero(key)[0]
        for a, sa in enumerate(states):
            if not all(sa[l] == (-1 if key[l] == 1 else 1) for l in flips):
                continue                                     # ladder can't act here
            sb = sa.copy()
            for l in flips:
                sb[l] = -sb[l]
            b = idx.get(tuple(sb))
            if b is None:
                continue                                     # left the manifold
            subs = {bw.Sz(model.links[l]): sp.Rational(int(sa[l]), 2)
                    for l in range(nL)}
            H[b, a] += complex(poly.subs(subs)).real
    return H


def exact_pvp(model: bw.Model, states, idx) -> np.ndarray:
    """`P V P`: the microscopic hop `V` projected onto the ground manifold -- the exact
    first-order effective Hamiltonian. `H_eff^(1)` from the engine must equal this."""
    D = len(states)
    P = np.zeros((D, D))
    ops = [(next(k for k, v in o.od_rep.items() if v == 1),
            next(k for k, v in o.od_rep.items() if v == -1), complex(o.coeff).real)
           for o in model.operators]
    for a, sa in enumerate(states):
        for i, j, c in ops:                                  # S+_i S-_j: i down, j up
            if sa[i] == -1 and sa[j] == 1:
                sb = sa.copy(); sb[i] = 1; sb[j] = -1
                b = idx.get(tuple(sb))
                if b is not None:
                    P[b, a] += c
    return P


# ---------------------------------------------------------------------------
# Exact microscopic H = H0 + V (two regimes: sparse Sz=0 sector, or full dense)
# ---------------------------------------------------------------------------

def _diagonal_H0(model, sig):
    inc = np.array(model.incidence.tolist(), dtype=int)
    con = np.array([float(model.constraints[k]) for k in range(len(model.verts))])
    charge = sig @ inc.T / 2.0                               # physical charge per vertex
    return (charge ** 2) @ con


def exact_sector_parts(model: bw.Model):
    """`H0` (diagonal) and `V` (hops) as sparse matrices on the total-Sz=0 sector, which
    contains the whole ice manifold of an even model (e.g. the 16-spin cubic cell).
    Build once; diagonalise `H0 + lam*V` per sweep point."""
    nL = len(model.links)
    base = [b for b in range(1 << nL) if bin(b).count("1") == nL // 2]
    index = {b: i for i, b in enumerate(base)}
    dim = len(base)
    sig = np.array([[1 if (b >> l) & 1 else -1 for l in range(nL)] for b in base])
    H0 = ss.diags(_diagonal_H0(model, sig), format="csr")
    rows, cols, data = [], [], []
    for o in model.operators:
        i = next(k for k, v in o.od_rep.items() if v == 1)
        j = next(k for k, v in o.od_rep.items() if v == -1)
        c = complex(o.coeff).real
        for bi, b in enumerate(base):
            if ((b >> i) & 1) == 0 and ((b >> j) & 1) == 1:
                nb = (b | (1 << i)) & ~(1 << j)
                rows.append(index[nb]); cols.append(bi); data.append(c)
    V = ss.csr_matrix((data, (rows, cols)), shape=(dim, dim))
    return H0, V


def lowest_eigenvalues(H, k: int) -> np.ndarray:
    """The `k` smallest-algebraic eigenvalues of sparse Hermitian `H`, sorted."""
    ev = ssl.eigsh(H, k=min(k + 4, H.shape[0] - 2), which="SA",
                   return_eigenvectors=False)
    return np.sort(ev)[:k]


def exact_dense_parts(model: bw.Model):
    """`H0` and `V` as dense matrices on the *full* Hilbert space -- for small odd-order
    geometries whose ground manifold spans several total-Sz sectors."""
    nL = len(model.links)
    dim = 1 << nL
    sig = np.array([[1 if (b >> l) & 1 else -1 for l in range(nL)] for b in range(dim)])
    H0 = np.diag(_diagonal_H0(model, sig))
    V = np.zeros((dim, dim))
    for o in model.operators:
        i = next(k for k, v in o.od_rep.items() if v == 1)
        j = next(k for k, v in o.od_rep.items() if v == -1)
        c = complex(o.coeff).real
        for b in range(dim):
            if ((b >> i) & 1) == 0 and ((b >> j) & 1) == 1:
                nb = (b | (1 << i)) & ~(1 << j)
                V[nb, b] += c
    return H0, V


def ground_energy(model: bw.Model, states) -> float:
    """H0 on any ground state (constant across the manifold): the zeroth-order energy
    the effective-Hamiltonian spectrum is measured relative to. Zero for an even model;
    `sum_odd (J_zz/2)(1/2)^2` when odd vertices carry a +-1/2 zero point."""
    sig = np.array([states[0]])
    return float(_diagonal_H0(model, sig)[0])
