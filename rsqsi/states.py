"""Classical S^z configurations and the action of string operators on them.

A configuration ``cfg`` maps every site to its S^z eigenvalue in units of 1/2
(i.e. +-1).  Applying a :class:`~rsqsi.terms.StringTerm` returns the evolved
configuration together with the *diagonal factor* contributed by any S^z
operators in the term, or ``None`` if a ladder operator annihilates the state
(spin-1/2: S^+ on an up spin, or S^- on a down spin).

The S^z factors are evaluated on the incoming configuration.  A term's S^z sites
must be disjoint from its ladder sites (enforced in :class:`StringTerm`), so the
ladder operators never shift a spin that an S^z of the same term measures, and
the ordering of commuting distinct-site factors is immaterial.
"""
from __future__ import annotations

import sympy as sp

from .terms import StringTerm


def apply_term(cfg: dict[int, int], term: StringTerm
               ) -> tuple[dict[int, int], sp.Expr] | None:
    """Apply ``term`` to ``cfg`` (right-most ladder factor first).

    Returns ``(new_cfg, factor)`` where ``factor`` is the product of the S^z
    eigenvalues (S^z = cfg/2) of the term's ``sz_sites``.  Returns ``None`` if a
    ladder operator annihilates the state.
    """
    factor = sp.Integer(1)
    for s in term.sz_sites:
        factor *= sp.Rational(cfg[s], 2)

    new = dict(cfg)
    for site, delta in reversed(term.factors):
        if delta > 0:           # S^+ : need spin down -> up
            if new[site] != -1:
                return None
            new[site] = +1
        else:                   # S^- : need spin up -> down
            if new[site] != +1:
                return None
            new[site] = -1
    return new, factor


def flipped_sites(a: dict[int, int], b: dict[int, int]) -> frozenset:
    """Sites whose S^z differs between two configurations."""
    return frozenset(s for s in a if a[s] != b[s])
