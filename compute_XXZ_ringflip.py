#!/bin/env python3
"""Driver: low-order BW terms of the QSI effective Hamiltonian (thesis Ch.1).

Reproduces the breathing-pyrochlore XXZ worked example from the thesis
(Sanders 2026, Ch.1 "Algorithm for generating terms"):

  * N = 2 : nearest-neighbour Ising renormalisation  ~ J_pm^2 / J_zz  S^z S^z
  * N = 3 : the ring-exchange  [3! J_pm,B^3/J_zz,A^2 + 3! J_pm,A^3/J_zz,B^2] O_p

Coefficients are symbolic in the (per-sublattice, hence disorder-ready) couplings.
"""
from __future__ import annotations

import sympy as sp
import rsqsi.lattice as lat
import rsqsi.brillouin_wigner as bw
import itertools
import numpy as np

# conventional-cube = 4; diamond = FCC + 2-atom basis (the two tetra centres).
# delta is the bond vector head - tail (from the sl-0 atom toward the sl-1 atom).
DIAMOND = lat.PrimitiveCell(
    sublattices=[sp.Matrix([0, 0, 0]), sp.Matrix([2, 2, 2])],
    bonds=[
        (0, sp.Matrix([ 2,  2,  2])),
        (0, sp.Matrix([ 2, -2, -2])),
        (0, sp.Matrix([-2,  2, -2])),
        (0, sp.Matrix([-2, -2,  2])),
    ],
    lattice_vectors=sp.Matrix([[0, 4, 4], [4, 0, 4], [4, 4, 0]]),
)


def build_diamond(L: int, periodic: bool = False) -> lat.Lattice:
    return lat.build(DIAMOND, (L, L, L), periodic)


JpmA, JpmB, JzzA, JzzB = sp.symbols('J_pm_A J_pm_B J_zz_A J_zz_B')

def main() -> None:
    dmnd_lat = build_diamond(L=5)
    print(len(dmnd_lat.verts), "vertices", len(dmnd_lat.links), "links")

    # implement the XXZ model
    m = bw.Model(dmnd_lat)

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

    for order in range(4):
        loops = bw.nonvanishing_strings(m, order)
        print(f"Order {order} -> found {len(loops)} unique loops")
    


if __name__ == "__main__":
    main()
