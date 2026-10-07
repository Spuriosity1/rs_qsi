"""String-operator terms of the perturbation V.

A :class:`StringTerm` is a single off-diagonal term of V: a product of spin
ladder operators on an ordered list of sites (the 1-chain gamma) together with a
symbolic coupling, and optionally some diagonal S^z factors (the ``sz_sites``).
``factors`` lists ``(site, delta)`` pairs with ``delta = +1`` for S^+ and
``delta = -1`` for S^-, read as an operator product left-to-right (the right-most
factor acts first on a ket).

This representation captures an *arbitrary* off-diagonal term built from spin
operators, e.g. every channel of the generic QSI Hamiltonian::

    h S^x_j                     -> field(j, h)        (S^+_j and S^-_j)
    J_pm   S^+_i S^-_j          -> hop(i, j, Jpm)
    J_pmpm S^+_i S^+_j          -> pmpm(i, j, Jpmpm)
    J_zpm  S^z_n S^+_o          -> zpm(n, o, +1, Jzpm)

The engine (:mod:`rsqsi.bw`) consumes any ``list[StringTerm]`` -- build them
directly for full generality, or use the constructors and the
``place_on_*`` replicators below to spread a template across the lattice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import sympy as sp

from .lattice_model import IceModel


@dataclass(frozen=True)
class StringTerm:
    factors: tuple[tuple[int, int], ...]   # ((site, +-1), ...) : S^+/S^-
    coupling: sp.Expr                      # symbolic coupling (may be a product)
    sz_sites: tuple[int, ...] = ()         # extra diagonal S^z factors

    def __post_init__(self):
        for _, d in self.factors:
            if d not in (+1, -1):
                raise ValueError("ladder delta must be +-1")
        ladder_sites = {s for s, _ in self.factors}
        if ladder_sites & set(self.sz_sites):
            raise ValueError("sz_sites must be disjoint from ladder sites "
                             "(mixed same-site S^z/S^± ordering is ambiguous)")

    @property
    def support(self) -> frozenset:
        return frozenset(s for s, _ in self.factors) | frozenset(self.sz_sites)

    @property
    def net_flip(self) -> dict[int, int]:
        """Net change of 2*S^z per site (the boundary d(gamma); spin-1/2)."""
        out: dict[int, int] = {}
        for s, d in self.factors:
            out[s] = out.get(s, 0) + 2 * d
        return {s: v for s, v in out.items() if v != 0}

    @property
    def is_diagonal(self) -> bool:
        return not self.factors

    def dagger(self) -> "StringTerm":
        """Hermitian conjugate: reverse order and flip every ladder."""
        return StringTerm(
            factors=tuple((s, -d) for s, d in reversed(self.factors)),
            coupling=self.coupling, sz_sites=self.sz_sites)

    def __repr__(self) -> str:
        lad = " ".join(f"S{'+' if d > 0 else '-'}_{s}" for s, d in self.factors)
        zz = " ".join(f"Sz_{s}" for s in self.sz_sites)
        body = (zz + " " + lad).strip()
        return f"{self.coupling}*{body}" if body else f"{self.coupling}"


# --------------------------------------------------------------------------
# convenience constructors for individual terms
# --------------------------------------------------------------------------

def hop(i: int, j: int, coupling: sp.Expr) -> StringTerm:
    """S^+_i S^-_j."""
    return StringTerm(factors=((i, +1), (j, -1)), coupling=coupling)


def pmpm(i: int, j: int, coupling: sp.Expr) -> StringTerm:
    """S^+_i S^+_j (the J_pmpm double-raise channel)."""
    return StringTerm(factors=((i, +1), (j, +1)), coupling=coupling)


def field(j: int, coupling: sp.Expr) -> list[StringTerm]:
    """Transverse field h S^x_j = h (S^+_j + S^-_j)/... -> two one-link strings
    (the user folds any 1/2 into ``coupling``)."""
    return [StringTerm(factors=((j, +1),), coupling=coupling),
            StringTerm(factors=((j, -1),), coupling=coupling)]


def zpm(zsite: int, j: int, direction: int, coupling: sp.Expr) -> StringTerm:
    """S^z_{zsite} S^{direction}_j  (the J_zpm channel)."""
    return StringTerm(factors=((j, direction),), coupling=coupling,
                      sz_sites=(zsite,))


# --------------------------------------------------------------------------
# generic replicators: spread a local template across the lattice
# --------------------------------------------------------------------------

BondFactory = Callable[[IceModel, int, int, str], list[StringTerm]]
SiteFactory = Callable[[IceModel, int, int], list[StringTerm]]


def place_on_bonds(model: IceModel, factory: BondFactory) -> list[StringTerm]:
    """Call ``factory(model, a, c, kind)`` on every nearest-neighbour bond
    (``kind`` = the tetra sublattice 'A'/'B') and collect the returned terms."""
    terms: list[StringTerm] = []
    for b in model.bonds:
        a, c = sorted(b)
        kind = model.tetras[model.bond_tetra[b]].kind
        terms.extend(factory(model, a, c, kind))
    return terms


def place_on_sites(model: IceModel, factory: SiteFactory) -> list[StringTerm]:
    """Call ``factory(model, site, sublattice)`` on every site."""
    terms: list[StringTerm] = []
    for s in model.sites:
        terms.extend(factory(model, s, model.sublattice[s]))
    return terms


# --------------------------------------------------------------------------
# ready-made channels of the generic QSI Hamiltonian
# --------------------------------------------------------------------------

def xxz_perturbation(model: IceModel,
                     jpm: dict[str, sp.Symbol] | None = None
                     ) -> list[StringTerm]:
    """V = sum_<ij> Jpm (S^+_i S^-_j + h.c.), coupling chosen by the bond's tetra."""
    if jpm is None:
        jpm = {'A': sp.Symbol('J_pm_A', positive=True),
               'B': sp.Symbol('J_pm_B', positive=True)}

    def factory(_m: IceModel, a: int, c: int, kind: str) -> list[StringTerm]:
        g = jpm[kind]
        return [hop(a, c, g), hop(c, a, g)]

    return place_on_bonds(model, factory)


def transverse_field(model: IceModel,
                     h: sp.Symbol | None = None) -> list[StringTerm]:
    """V = h sum_j S^x_j, as one-link string operators S^+_j, S^-_j."""
    if h is None:
        h = sp.Symbol('h', positive=True)
    return place_on_sites(model, lambda _m, s, _mu: field(s, h))
