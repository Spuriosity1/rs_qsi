#!/usr/bin/env python3
r"""Re-derive the fourth-order *complex* ring flip (H_pm^2 H_zpm^2) with the
homological BW engine, and check it against the thesis (Sanders 2026, Ch.2 +
Appendix 2 "Derivation of the complex ring flip").

The hexagon is the 6-cycle of tetrahedra

    v0(A) -L1- v1(B) -L2- v2(A) -L3- v3(B) -L4- v4(A) -L5- v5(B) -L6- v0 ,

six spins on links L1..L6 (= thesis sites 1..6), tetrahedra alternating A/B.
The ring flip is  O_p = S1+ S2- S3+ S4- S5+ S6- .

The thesis ortho/para alpha-coefficients are *pure propagator sums* fixed by the
ladder (od) structure and the A/B pattern -- the zeta phases and the dangling
S^z only set the overall g/g' split, not the energy denominators. So we can
check the energy tables and the alphas directly with the ladder parts alone.

Convention (matches the validated XXZ regression): constraint strength on a
tetra = Jzz/2, so a virtual state with charges Q_v has energy sum_v (Jzz_v/2)Q_v^2.
Write A := Jzz_A/2, B := Jzz_B/2 exactly as the thesis appendix does.
"""
from __future__ import annotations

import itertools
import sys
import os

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(__file__))
from rsqsi import lattice as lat          # noqa: E402
from rsqsi import brillouin_wigner as bw  # noqa: E402

A, B = sp.symbols('A B', positive=True)   # A = Jzz_A/2, B = Jzz_B/2


# ---------------------------------------------------------------------------
# Build the hexagon as a bare oriented 1-complex (6-cycle), orientations chosen
# so that O_p closes (is a 1-cycle).
# ---------------------------------------------------------------------------

def build_hexagon():
    sub = {0: 'A', 1: 'B', 2: 'A', 3: 'B', 4: 'A', 5: 'B'}
    verts = {i: lat.Vertex(i, sub[i], sp.Matrix([i, 0, 0])) for i in range(6)}
    # link k (k=1..6) joins v[k-1] and v[k mod 6]
    endpoints = {k: (k - 1, k % 6) for k in range(1, 7)}

    # O_p sign pattern on sites 1..6
    op = {1: +1, 2: -1, 3: +1, 4: -1, 5: +1, 6: -1}

    # brute-force each link's orientation so the O_p chain is a cycle
    for bits in itertools.product((0, 1), repeat=6):
        links, incident = {}, {i: set() for i in range(6)}
        for k in range(1, 7):
            a, b = endpoints[k]
            tail, head = (a, b) if bits[k - 1] == 0 else (b, a)
            links[k] = lat.Link(k, tail, head, sublattice=sub[tail])
            incident[tail].add(k)
            incident[head].add(k)
        L = lat.Lattice(verts, links, incident)
        m = bw.Model(L)
        pos = {lid: i for i, lid in enumerate(m.links)}
        chain = np.zeros(len(m.links), int)
        for k in range(1, 7):
            chain[pos[k]] = op[k]
        if m.is_cycle(chain):
            return m, pos
    raise RuntimeError("no orientation closes O_p")


def op_vec(pos, nL, flips):
    """od vector over links from {site: +-1}."""
    v = np.zeros(nL, int)
    for site, s in flips.items():
        v[pos[site]] = s
    return v


def energy_products(m, pos, named_ops):
    """For each ordering of the operators, the ORDERED tuple of intermediate
    spinon energies (product of the 3 propagator energies for a 4-op loop).

    Returns {ordering(tuple of names): (sympy energy product, [E1,E2,E3])}.
    Energies from charges Q (background independent): E_k = sum_v con_v Q_v^2.
    """
    nL = len(m.links)
    con = []                                  # constraint per vertex, A or B
    for v in m.verts:
        con.append(A if m.lattice.verts[v].sublattice == 'A' else B)
    inc = np.array(m.incidence.tolist(), dtype=int)   # |V| x |L|
    ods = {name: op_vec(pos, nL, flips) for name, flips in named_ops.items()}
    dq = {name: inc @ ods[name] for name in named_ops}

    out = {}
    for perm in itertools.permutations(named_ops):
        charge = np.zeros(inc.shape[0], int)
        chain = np.zeros(nL, int)
        Es, prod, ok = [], sp.Integer(1), True
        for k, name in enumerate(perm):
            chain = chain + ods[name]
            charge = charge + dq[name]
            if np.abs(chain).max(initial=0) > 1:          # spin-1/2 blocked
                ok = False
                break
            if k < len(perm) - 1:
                nz = np.nonzero(charge)[0]
                if nz.size == 0:                          # early vacuum return
                    ok = False
                    break
                Ek = sum(con[i] * int(charge[i])**2 for i in nz)
                Es.append(sp.expand(Ek))
                prod *= Ek
        if ok:
            out[perm] = (sp.expand(prod), Es)
    return out


def group_by_energy(table):
    groups = {}
    for perm, (prod, _) in table.items():
        groups.setdefault(prod, []).append(''.join(perm))
    return groups


def main():
    m, pos = build_hexagon()
    print(f"hexagon built: {len(m.verts)} tetrahedra, {len(m.links)} spins")
    print("sublattices:", {v: m.lattice.verts[v].sublattice for v in m.verts})

    # ---- ortho channel operators (ladder parts only) ----
    # E = Jpm,A S6- S1+ ;  F = Jpm,A S2- S3+ ;  e = S4- (Zpm) ;  f = S5+ (Zpm)
    ortho = {
        'E': {6: -1, 1: +1},
        'F': {2: -1, 3: +1},
        'e': {4: -1},
        'f': {5: +1},
    }
    print("\n=== ORTHO channel: spinon-energy products by ordering ===")
    tbl = energy_products(m, pos, ortho)
    print(f"{len(tbl)} / 24 orderings survive")
    for prod, perms in sorted(group_by_energy(tbl).items(), key=lambda kv: str(kv[0])):
        print(f"  {sp.factor(prod)!s:40s}  <- {', '.join(sorted(perms))}")

    # scalar BW propagator weight (sum over orderings of prod(-1/E_k))
    wortho = sum(sp.Integer(-1)**(len(ortho) - 1) / prod
                 for prod, _ in tbl.values())
    print("  sum_orderings prod(-1/E_k) =", sp.factor(sp.cancel(wortho)))

    # ---- para channel operators ----
    # p = Zpm S1+ ; m = Zpm* S4- ; T = Jpm,A S5+ S6- ; D = Jpm,B S2- S3+
    para = {
        'p': {1: +1},
        'q': {4: -1},          # 'm' in thesis (renamed to avoid clash)
        'T': {5: +1, 6: -1},
        'D': {2: -1, 3: +1},
    }
    print("\n=== PARA channel: spinon-energy products by ordering ===")
    tbl2 = energy_products(m, pos, para)
    print(f"{len(tbl2)} / 24 orderings survive")
    for prod, perms in sorted(group_by_energy(tbl2).items(), key=lambda kv: str(kv[0])):
        print(f"  {sp.factor(prod)!s:40s}  <- {', '.join(sorted(perms))}")
    wpara = sum(sp.Integer(-1)**(len(para) - 1) / prod
                for prod, _ in tbl2.values())
    print("  sum_orderings prod(-1/E_k) =", sp.factor(sp.cancel(wpara)))

    # -------------------------------------------------------------------
    # End-to-end: the main-text worked process (Fig. cplx_ringflips a),
    # operator set  { z16 S1+ Sz6, z23c S2- Sz3, S3+ S4-, S5+ S6- },
    # with the two S^z sitting on ring sites 3 and 6 (both ring-flipped).
    # walsh_amplitude applies every ladder honestly and reads each dangling
    # S^z at its moment of action -> the ordering-dependent +-1/2 the thesis
    # tracks by hand. Result is the (complex) scalar coefficient of O_p.
    # -------------------------------------------------------------------
    print("\n=== END-TO-END: coefficient of O_p (main-text process) ===")
    z16, z23c = sp.symbols('zeta16 zeta23c')
    JpmB, Jzpm = sp.symbols('J_pm_B J_zpm', positive=True)
    # constraints: A := Jzz_A/2 on A-tetras, B := Jzz_B/2 on B-tetras
    for i, v in enumerate(m.verts):
        m.set_constraint(A if m.lattice.verts[v].sublattice == 'A' else B, i)
    nL = len(m.links)

    def add(coeff, od, d):
        m.add_operator(coeff, op_vec(pos, nL, od), op_vec(pos, nL, d))

    base = len(m.operators)
    add(Jzpm * z16,  {1: +1}, {6: 1})          # a: z16 S1+ Sz6
    add(Jzpm * z23c, {2: -1}, {3: 1})          # b: z23c S2- Sz3
    add(JpmB,        {3: +1, 4: -1}, {})       # c: S3+ S4-
    add(JpmB,        {5: +1, 6: -1}, {})       # d: S5+ S6-
    loop = tuple(range(base, base + 4))

    w = bw.walsh_amplitude(m, loop)
    w = sp.cancel(w)
    print("  walsh_amplitude coeff of O_p =", w)
    print("  factored                     =", sp.factor(w))
    print("  value at A = B (no breathing):", sp.simplify(w.subs(B, A)))

    # -------------------------------------------------------------------
    # Surgical check of the thesis's explicit eq. (5)+(6): exactly the two
    # orderings m1 = (d,a,b,c) and m2 = (a,d,b,c) of the SAME four operators.
    # Reproduce the honest per-ordering (S^z scalar) x prod(-1/E_k).
    # -------------------------------------------------------------------
    print("\n--- thesis eq (5)+(6): the two orderings m1, m2 alone ---")
    ops = [m.operators[i] for i in loop]          # [a, b, c, d]
    idx = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    con = [A if m.lattice.verts[v].sublattice == 'A' else B for v in m.verts]
    inc = np.array(m.incidence.tolist(), dtype=int)
    start = {1: -1, 2: +1, 3: -1, 4: +1, 5: -1, 6: +1}   # ice start O acts on

    def contribution(order):
        state = dict(start)
        charge = np.zeros(inc.shape[0], int)
        factor = sp.Integer(1)
        Es = []
        for k, nm in enumerate(order):
            o = ops[idx[nm]]
            for p in np.nonzero(o.od_rep)[0]:
                site = m.links[p]
                state[site] = -state[site]
            charge = charge + inc @ o.od_rep
            for p in np.nonzero(o.d_rep)[0]:
                factor *= sp.Rational(state[m.links[p]], 2)   # dangling S^z
            if k < len(order) - 1:
                Ek = sum(con[i] * int(charge[i])**2 for i in np.nonzero(charge)[0])
                Es.append(sp.expand(Ek))
                factor *= sp.Integer(-1) / Ek
        return sp.cancel(factor * sp.prod([o.coeff for o in ops])), Es

    c1, e1 = contribution('dabc')
    c2, e2 = contribution('adbc')
    print("  m1=(d,a,b,c): props", [sp.factor(x) for x in e1], " contrib", sp.factor(c1))
    print("  m2=(a,d,b,c): props", [sp.factor(x) for x in e2], " contrib", sp.factor(c2))
    engine56 = sp.cancel(c1 + c2)
    print("  engine (5)+(6) =", sp.factor(engine56))

    thesis56 = (z16 * z23c * JpmB**2 * Jzpm**2
                * (sp.Integer(1) / (2 * A)) * (sp.Integer(1) / (A + B))
                * (sp.Integer(-1) / (2 * A) + sp.Integer(1) / (A + B)))
    print("  thesis (5)+(6) =", sp.factor(thesis56))
    print("  difference      =", sp.simplify(engine56 - thesis56))

    # ===================================================================
    # Is g' zero?  g' = Im(C_O), C_O = sum over processes of (real weight)x(phase).
    # Inversion swaps A<->B and conjugates the phases (zeta -> zeta*); at A=B it
    # is a symmetry, forcing C_O real (g'=0). With breathing it is broken.
    # Build the ortho process P1 and its inversion partner P1' and read off Im.
    # ===================================================================
    print("\n=== Is g' zero?  ortho process + its inversion partner ===")

    def coeff_of_O(ladder_ops, con_A, con_B):
        """walsh coeff of O_p for a bespoke operator list on a fresh model copy,
        with A-tetras at strength con_A and B-tetras at con_B."""
        mm = bw.Model(m.lattice)
        for i, v in enumerate(mm.verts):
            mm.set_constraint(con_A if mm.lattice.verts[v].sublattice == 'A' else con_B, i)
        nn = len(mm.links)
        base = len(mm.operators)
        for coeff, od, d in ladder_ops:
            mm.add_operator(coeff, op_vec(pos, nn, od), op_vec(pos, nn, d))
        return sp.cancel(bw.walsh_amplitude(mm, tuple(range(base, base + len(ladder_ops)))))

    JpmA = sp.symbols('J_pm_A', positive=True)
    phi = sp.symbols('phi')                      # the ortho relative phase zeta16*zeta23*
    phic = sp.conjugate(phi)

    # P1: ortho on B-hops, phase phi  (= the main-text process)
    P1 = [(Jzpm * phi, {1: +1}, {6: 1}),
          (Jzpm,       {2: -1}, {3: 1}),
          (JpmB, {3: +1, 4: -1}, {}),
          (JpmB, {5: +1, 6: -1}, {})]
    C1 = coeff_of_O(P1, A, B)
    # P1': inversion image -- A<->B on tetra strengths AND couplings, phase -> phi*
    P1p = [(Jzpm * phic, {1: +1}, {6: 1}),
           (Jzpm,        {2: -1}, {3: 1}),
           (JpmA, {3: +1, 4: -1}, {}),
           (JpmA, {5: +1, 6: -1}, {})]
    C1p = coeff_of_O(P1p, B, A)

    print("  C(P1)  =", sp.factor(C1))
    print("  C(P1') =", sp.factor(C1p))
    Co = sp.expand(C1 + C1p)
    # impose |phi|=1: phi*conj(phi)=1, and write phi = cos+ i sin
    th = sp.symbols('theta', real=True)
    Co_th = Co.subs({phi: sp.exp(sp.I * th), phic: sp.exp(-sp.I * th)})
    ImC = sp.simplify(sp.im(sp.expand_complex(Co_th)))
    print("  Im(C_O) for this ortho pair =", sp.simplify(ImC))
    print("    at A = B (no breathing)   =", sp.simplify(ImC.subs(B, A)))
    physical = sp.simplify(ImC.subs(th, 2 * sp.pi / 3))
    print("    at phi = w = exp(2 pi i/3):", physical)

    # thesis breathing parametrisation: Jzz_A/Jzz_B = Jpm_A/Jpm_B = (1+k)/(1-k)
    k, Jp, J0 = sp.symbols('kappa J_pm J_zz', positive=True)
    subs = {A: (J0 / 2) * (1 + k), B: (J0 / 2) * (1 - k),
            JpmA: Jp * (1 + k), JpmB: Jp * (1 - k)}
    gp = sp.simplify(physical.subs(subs))
    print("    g'(kappa) [this pair] =", gp)
    print("    series in kappa       =", sp.series(gp, k, 0, 2))
    print("    g'(kappa=0)           =", gp.subs(k, 0))


if __name__ == "__main__":
    main()
