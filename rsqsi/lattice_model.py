"""Ice-model geometry: the general-ice core and its pyrochlore instance.

An :class:`IceModel` is the minimal geometric data the perturbative engine needs:

  * ``sites``      -- spin-1/2 sites (= links/1-cells of the parent lattice; for
                      pyrochlore these are the pyrochlore sites / diamond links),
                      each tagged with a sublattice index.
  * ``tetras``     -- the 0-cells (corner-sharing units). Each carries its member
                      sites and a ``kind`` label ('A' = up, 'B' = down for the
                      breathing pyrochlore) that selects its Ising coupling.
  * ``bonds``      -- unordered nearest-neighbour site pairs, each belonging to
                      exactly one tetra (used to place the XXZ hopping terms).
  * ``rings``      -- elementary plaquettes (hexagons on pyrochlore) as ordered
                      member lists with alternating signs, used to build the
                      ring-exchange reference operator.
  * ``ref``        -- a reference 2-in-2-out ice configuration (S^z eigenvalues in
                      units of 1/2, i.e. +-1 per site).
  * ``jzz``        -- sympy Ising coupling symbol per tetra kind.

The :func:`build_pyrochlore` factory wraps the vendored geometry code
(``rsqsi.geometry``) into this structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from .geometry import _lattice as lattice
from .geometry import _pyrochlore as pyro


@dataclass
class Tetra:
    members: tuple[int, ...]
    kind: str  # 'A' (up) or 'B' (down)


@dataclass
class Ring:
    members: tuple[int, ...]
    signs: tuple[int, ...]  # +-1, the sign of S^+/S^- in the ring-flip operator


@dataclass
class IceModel:
    sites: list[int]
    sublattice: dict[int, int]
    tetras: list[Tetra]
    bonds: list[frozenset]          # {a, b}
    rings: list[Ring]
    ref: dict[int, int]             # site -> +-1  (S^z in units of 1/2)
    jzz: dict[str, sp.Symbol]       # kind -> coupling symbol
    site_tetras: dict[int, list[int]] = field(default_factory=dict)
    bond_tetra: dict[frozenset, int] = field(default_factory=dict)

    def tetra_of_bond(self, a: int, b: int) -> Tetra:
        return self.tetras[self.bond_tetra[frozenset((a, b))]]

    def is_ice(self, cfg: dict[int, int]) -> bool:
        """True iff every tetra satisfies the 2-in-2-out (zero-charge) rule."""
        for t in self.tetras:
            if sum(cfg[i] for i in t.members) != 0:
                return False
        return True

    def spinon_energy(self, cfg: dict[int, int]) -> sp.Expr:
        """H0 energy of ``cfg`` relative to the ice ground state.

        H0 = sum_<ij> Jzz S^z_i S^z_j = sum_t Jzz_t [ Q_t^2/2 - 1/2 ], with
        Q_t = sum_{i in t} S^z_i.  Relative to an ice state (Q_t = 0) this is
        E = sum_t Jzz_t * Q_t^2 / 2, where S^z = cfg/2 so Q_t = (sum cfg)/2.
        A single spinon (Q_t = +-1) therefore costs Jzz_t/2.
        """
        E = sp.Integer(0)
        for t in self.tetras:
            q = sp.Rational(sum(cfg[i] for i in t.members), 2)  # Q_t
            if q != 0:
                E += self.jzz[t.kind] * q**2 / 2
        return sp.expand(E)


def build_pyrochlore(L: int = 2) -> IceModel:
    """Build a breathing-pyrochlore ice model on an L*L*L supercell.

    Up tetrahedra are labelled 'A', down tetrahedra 'B', matching the breathing
    anisotropy of the thesis (J_zz,A / J_zz,B, J_pm,A / J_pm,B).
    """
    lat = lattice.Lattice(pyro.primitive, [[L, 0, 0], [0, L, 0], [0, 0, L]])

    sites = list(range(lat.num_atoms))
    sublattice = {i: int(lat.atoms[i].sl_name) for i in sites}

    up, dn = pyro.get_tetras(lat)
    tetras: list[Tetra] = []
    for t in up:
        tetras.append(Tetra(members=tuple(t.members), kind='A'))
    for t in dn:
        tetras.append(Tetra(members=tuple(t.members), kind='B'))

    # site -> incident tetra indices, and bond -> tetra
    site_tetras: dict[int, list[int]] = {i: [] for i in sites}
    for ti, t in enumerate(tetras):
        for i in t.members:
            site_tetras[i].append(ti)

    bonds: list[frozenset] = []
    bond_tetra: dict[frozenset, int] = {}
    seen = set()
    for ti, t in enumerate(tetras):
        m = t.members
        for a_i in range(len(m)):
            for b_i in range(a_i + 1, len(m)):
                b = frozenset((m[a_i], m[b_i]))
                if b not in seen:
                    seen.add(b)
                    bonds.append(b)
                    bond_tetra[b] = ti

    rings = [Ring(members=tuple(r.members), signs=tuple(r.signs))
             for r in pyro.get_ringflips(lat)]

    # Reference ice state: sublattices {0,1} up, {2,3} down.  Each tetra holds
    # exactly one site of each sublattice, so every tetra is 2-in-2-out.
    ref = {i: (+1 if sublattice[i] in (0, 1) else -1) for i in sites}

    jzz = {'A': sp.Symbol('J_zz_A', positive=True),
           'B': sp.Symbol('J_zz_B', positive=True)}

    model = IceModel(sites=sites, sublattice=sublattice, tetras=tetras,
                     bonds=bonds, rings=rings, ref=ref, jzz=jzz,
                     site_tetras=site_tetras, bond_tetra=bond_tetra)
    assert model.is_ice(ref), "reference configuration is not an ice state"
    return model
