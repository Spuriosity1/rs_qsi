r"""Open transverse-field Ising chain as an ice model: the minimal odd-order
("half-integer zero point") worked example.

A 1D chain of N sites tiles a one-sublattice primitive cell with a single +x hop.
Built open (`periodic=False`) it has N 0-cells and N-1 1-cells (spins): the two end
0-cells are 1-member constrainers (odd order -> C = +-1/2, a free zero point), the
interior ones are 2-member (even -> C = 0). The ferromagnetic ground manifold is
two-fold (all up / all down); the transverse field V = Gamma * sum_i (S+_i + S-_i)
connects them by sweeping a domain wall (spinon) across the chain.

The leading connecting operator is an *open string* flipping every spin, at order
N-1. The engine returns, for the full-flip channel,

    (-1)^N * 4^(N-2) * Gamma^(N-1) / J^(N-2)

(J the diagonal coupling; constraint strength J/2). N=3 is the first nontrivial
case: two orderings, each with one J/2 domain-wall denominator g = -2/J, giving
-4 Gamma^2 / J.
"""
import numpy as np
import sympy as sp

from rsqsi import lattice as lat, brillouin_wigner as bw

Gamma, J = sp.symbols("Gamma J", positive=True)


def build_chain(N: int) -> bw.Model:
    """Open N-site Ising chain decorated with a transverse field."""
    cell = lat.PrimitiveCell(
        sublattices=[sp.Matrix([0])],
        bonds=[(0, sp.Matrix([1]))],
        lattice_vectors=sp.Matrix([[1]]),
    )
    m = bw.Model(lat.build(cell, (N,), periodic=False))
    nL = len(m.links)
    for i in range(nL):
        for s in (+1, -1):                                  # S+_i and its h.c. S-_i
            od = np.zeros(nL, int); od[i] = s
            m.add_operator(Gamma, od, np.zeros(nL, int))
    for idx, _ in enumerate(m.verts):
        m.set_constraint(J / 2, idx)                        # constraint strength = J/2
    return m


def main() -> None:
    for N in range(2, 6):
        m = build_chain(N)
        orders = [len(m.lattice.incident[v]) for v in m.verts]
        print(f"N={N}: orders {orders}, odd={m.odd}")

        order = N - 1
        loops = [lp for lp in bw.nonvanishing_strings(m, order)
                 if bw.is_connected_loop(m, lp)]
        channels: dict = {}
        for lp in loops:
            channels.setdefault(bw._net_flip(m, lp), []).append(lp)

        for key, lps in channels.items():
            if any(key):                                    # the connecting open strings
                amp = sum((bw.walsh_amplitude(m, lp) for lp in lps), sp.Integer(0))
                print(f"    full-flip {key}: {sp.simplify(amp)}")


if __name__ == "__main__":
    main()
