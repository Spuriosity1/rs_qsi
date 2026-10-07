#!/bin/env python3
"""Driver: generate low-order terms of the QSI effective Hamiltonian (BW piece).

Reproduces the breathing-pyrochlore XXZ worked example from the thesis
(Sanders 2026, Ch.1 "Algorithm for generating terms"):

  * N = 2 : nearest-neighbour Ising renormalisation  ~ J_pm^2 / J_zz  S^z S^z
  * N = 3 : the ring-exchange  [3! J_pm,B^3/J_zz,A^2 + 3! J_pm,A^3/J_zz,B^2] O_p

Coefficients are symbolic in the (per-sublattice, hence disorder-ready) couplings.
"""
from __future__ import annotations

import sympy as sp

from rsqsi.lattice_model import build_pyrochlore
from rsqsi.terms import xxz_perturbation, transverse_field, StringTerm
from rsqsi.bw import generate_bw, Sz
from rsqsi.operators import as_latex


def main() -> None:
    model = build_pyrochlore(L=2)
    V = xxz_perturbation(model)

    print("=" * 70)
    print("Breathing-pyrochlore XXZ effective Hamiltonian (BW connected-loop)")
    print("=" * 70)

    # ---- N = 2 : Ising renormalisation on one A-bond -----------------------
    bond = next(b for b in model.bonds
                if model.tetras[model.bond_tetra[b]].kind == 'A')
    a, c = tuple(bond)
    res2 = generate_bw(model, V, N=2, support=bond)
    print(f"\n[N=2] A-bond ({a},{c}):")
    print("   diagonal operator:", sp.expand(res2.diag))
    print("   Ising coefficient  (Sz_%d Sz_%d):" % (a, c),
          sp.simplify(sp.expand(res2.diag).coeff(Sz(a) * Sz(c))))

    # ---- N = 3 : ring exchange on one hexagon ------------------------------
    hexagon = frozenset(model.rings[0].members)
    res3 = generate_bw(model, V, N=3, support=hexagon)
    print(f"\n[N=3] hexagon {sorted(hexagon)}:")
    print("   ring-exchange coefficient of O_p:",
          sp.simplify(res3.ring_coefficient()))
    print("   (thesis: 3! J_pm,B^3/J_zz,A^2 + 3! J_pm,A^3/J_zz,B^2)")
    print("\n   H_eff^(3) on this hexagon (LaTeX):")
    print("   ", as_latex(res3))

    # ---- arbitrary off-diagonal terms --------------------------------------
    # Any list[StringTerm] -- ladder factors plus optional S^z factors -- can be
    # fed to generate_bw.  Two illustrations:
    print("\n" + "-" * 70)
    print("Arbitrary off-diagonal perturbations")
    print("-" * 70)

    # (1) transverse field h S^x at second order on one site
    h = sp.Symbol('h', positive=True)
    j = model.sites[0]
    resh = generate_bw(model, transverse_field(model, h), N=2, support=frozenset((j,)))
    print(f"\n[N=2] transverse field on site {j}:")
    print("   diagonal response:", sp.expand(resh.diag),
          "  (= -2 h^2 / (J_zz,A + J_zz,B))")

    # (2) a custom S^z-dressed hop  g * S^z_k S^+_a S^-_c  (+ h.c.)
    k = next(x for ti in model.site_tetras[a] for x in model.tetras[ti].members
             if x not in (a, c))
    g = sp.Symbol('g', positive=True)
    T = StringTerm(factors=((a, +1), (c, -1)), coupling=g, sz_sites=(k,))
    resd = generate_bw(model, [T, T.dagger()], N=2, support=frozenset((a, c, k)))
    print(f"\n[N=2] custom term  g*S^z_{k} S^+_{a} S^-_{c} + h.c.:")
    print("   diagonal operator:", sp.expand(resd.diag),
          "  (= 1/4 of the plain hop, since (S^z)^2 = 1/4)")


if __name__ == "__main__":
    main()
