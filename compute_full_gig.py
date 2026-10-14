#!/usr/bin/env python3
r"""Full fourth-order complex ring flip g + i g' on the breathing pyrochlore:
enumerate ALL ortho/para/... H_±² H_{z±}² processes on a hexagon with the real
pyrochlore zeta phases, sum, and read off g (real) and g' (imag).

Pyrochlore spin sublattice mu in {0,1,2,3} = the four bond directions of the
diamond primitive cell; zeta_{ij} = zeta[mu_i, mu_j] with zeta = -conj(gamma),
gamma the standard symmetric cube-root-of-unity matrix (thesis Table 1.4,
-zeta*_{ij} = gamma_{ij}, w = e^{2 pi i/3}; this reproduces the standard
pyrochlore zeta of Ross 2011 / Savary-Balents).
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(__file__))
from rsqsi import lattice as lat          # noqa: E402
from rsqsi import brillouin_wigner as bw  # noqa: E402

DELTAS = [sp.Matrix([2, 2, 2]), sp.Matrix([2, -2, -2]),
          sp.Matrix([-2, 2, -2]), sp.Matrix([-2, -2, 2])]
DIAMOND = lat.PrimitiveCell(
    sublattices=[sp.Matrix([0, 0, 0]), sp.Matrix([2, 2, 2])],
    bonds=[(0, d) for d in DELTAS],
    lattice_vectors=sp.Matrix([[0, 4, 4], [4, 0, 4], [4, 4, 0]]),
)

w = sp.Rational(-1, 2) + sp.sqrt(3) / 2 * sp.I     # e^{2 pi i/3}, rectangular
GAMMA = sp.Matrix([[0, 1, w, w**2], [1, 0, w**2, w],
                   [w, w**2, 0, 1], [w**2, w, 1, 0]])
ZETA = sp.Matrix(4, 4, lambda i, j: sp.expand(-sp.conjugate(GAMMA[i, j])))

JpmA, JpmB = sp.symbols('J_pm_A J_pm_B', positive=True)
A, B = sp.symbols('A B', positive=True)                 # A=Jzz_A/2, B=Jzz_B/2
JzpmA, JzpmB = sp.symbols('J_zpm_A J_zpm_B', positive=True)
kap, Jp, Jz0, J0 = sp.symbols('kappa J_pm J_zpm J_zz', positive=True)
KSUBS = {A: (J0 / 2) * (1 + kap), B: (J0 / 2) * (1 - kap),
         JpmA: Jp * (1 + kap), JpmB: Jp * (1 - kap),
         JzpmA: Jz0 * (1 + kap), JzpmB: Jz0 * (1 - kap)}


def pyro_sublattice(L, lid):
    link = L.links[lid]
    delta = L.verts[link.head].position - L.verts[link.tail].position
    for mu, d in enumerate(DELTAS):
        if delta == d or delta == -d:
            return mu
    raise ValueError(f"link {lid} matches no bond direction")


def all_hexagons(m):
    """Distinct hexagons (frozenset of six ring links) with their O_p pattern."""
    seen, out = set(), []
    for loop in bw.find_closed_loops(m, 3):
        comp = sum(m.operators[a].od_rep for a in loop)
        if np.count_nonzero(comp) == 6 and np.abs(comp).max() == 1 \
                and bw.is_connected_loop(m, loop):
            ring = frozenset(m.links[i] for i in np.nonzero(comp)[0])
            if ring in seen:
                continue
            seen.add(ring)
            patt = {m.links[i]: int(comp[i]) for i in np.nonzero(comp)[0]}
            out.append((sorted(ring), patt))
    return out


def gprime_of_hexagon(L, ring, patt, verbose=False):
    mu = {lid: pyro_sublattice(L, lid) for lid in ring}
    tetra_links = {}
    for lid in ring:
        for v in (L.links[lid].tail, L.links[lid].head):
            tetra_links.setdefault(v, []).append(lid)
    ring_tetra = {v: ls for v, ls in tetra_links.items() if len(ls) == 2}

    m2 = bw.Model(L)
    for idx, v in enumerate(m2.verts):
        m2.set_constraint(A if v[1] == 0 else B, idx)
    pos = {lid: i for i, lid in enumerate(m2.links)}
    nL2 = len(m2.links)
    kinds, htet = {}, {}

    def mkop(coeff, flips, szs=()):
        od = np.zeros(nL2, int)
        for lid, s in flips.items():
            od[pos[lid]] = s
        d = np.zeros(nL2, int)
        for lid in szs:
            d[pos[lid]] = 1
        m2.add_operator(coeff, od, d)

    for v, ls in ring_tetra.items():
        a, b = ls
        Jpm_t = JpmA if v[1] == 0 else JpmB
        Jzpm_t = JzpmA if v[1] == 0 else JzpmB
        for i, j in ((a, b), (b, a)):
            htet[len(m2.operators)] = v[1]
            kinds[len(m2.operators)] = 'hop'
            mkop(Jpm_t, {i: +1, j: -1})
        for i, j in ((a, b), (b, a)):
            z = ZETA[mu[i], mu[j]]
            kinds[len(m2.operators)] = 'zpm'
            mkop(Jzpm_t * z, {i: +1}, (j,))
            kinds[len(m2.operators)] = 'zpm'
            mkop(Jzpm_t * sp.conjugate(z), {i: -1}, (j,))

    target = np.zeros(nL2, int)
    for lid, s in patt.items():
        target[pos[lid]] = s

    C_O = sp.Integer(0)
    nproc = 0
    byclass = {}
    for loop in bw.find_closed_loops(m2, 4):
        comp = sum(m2.operators[a].od_rep for a in loop)
        if not np.array_equal(comp, target):
            continue
        if sum(kinds[a] == 'zpm' for a in loop) != 2:
            continue
        if not bw.is_connected_loop(m2, loop):
            continue
        amp = bw.walsh_amplitude(m2, loop)
        C_O += amp
        nproc += 1
        hk = tuple(sorted(htet[a] for a in loop if kinds[a] == 'hop'))
        byclass[hk] = byclass.get(hk, 0) + 1
    C_O = sp.cancel(sp.expand(C_O))
    gp_k = sp.simplify(sp.im(C_O).subs(KSUBS))
    g_k = sp.simplify(sp.re(C_O).subs(KSUBS))
    if verbose:
        print(f"    {nproc} processes, classes {dict(byclass)}")
    return C_O, g_k, gp_k


def main():
    L = lat.build(DIAMOND, (4, 4, 4), periodic=False)
    m = bw.Model(L)
    nL = len(m.links)
    lpos = {lid: i for i, lid in enumerate(m.links)}
    Jpm = {0: JpmA, 1: JpmB}
    for v in m.verts:
        for a, b in itertools.combinations(sorted(m.lattice.incident[v]), 2):
            i, j = lpos[a], lpos[b]
            for si, sj in ((+1, -1), (-1, +1)):
                od = np.zeros(nL, int); od[i] = si; od[j] = sj
                m.add_operator(Jpm[v[1]], od, np.zeros(nL, int))

    hexes = all_hexagons(m)
    print(f"{len(hexes)} distinct hexagons in the cluster; checking g' on each:\n")
    results = []
    for n, (ring, patt) in enumerate(hexes[:6]):
        C_O, g_k, gp_k = gprime_of_hexagon(L, ring, patt, verbose=True)
        gp_lead = sp.series(gp_k, kap, 0, 2).removeO()
        g_lead = sp.series(g_k, kap, 0, 1).removeO()
        print(f"  hexagon {n} links {ring}")
        print(f"    g' = {gp_lead} + O(k^2)")
        print(f"    g  (H_pm^2 H_zpm^2 part) = {g_lead} + O(k)\n")
        results.append(sp.simplify(gp_lead))

    print("all hexagons give the same g'?",
          all(sp.simplify(r - results[0]) == 0 for r in results))
    print("\ng'  =", results[0], "  (thesis: -51*sqrt(3)/2 * J_pm^2 J_zpm^2/J_zz^3 * kappa)")


if __name__ == "__main__":
    main()
