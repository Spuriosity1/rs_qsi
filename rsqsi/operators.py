"""Render generated effective-Hamiltonian terms as operator expressions.

The symbolic coupling coefficients live in sympy; the *operator structure* (the
monomial of S^z / S^+ / S^- on the support) is represented with the user's
``Commutation`` package, which also provides LaTeX output.  Each rendered term is
a pair ``(sympy coefficient, Commutation operator monomial)``.
"""
from __future__ import annotations

import commutation as C
import sympy as sp

from .bw import BWResult


def _sz(site: int) -> C.Operator:
    return C.Operator(f"Sz_{site}", f"S^z_{{{site}}}")


def _ladder(site: int, direction: int) -> C.Operator:
    sym = "+" if direction > 0 else "-"
    return C.Operator(f"S{sym}_{site}", f"S^{sym}_{{{site}}}")


def _sz_monomial(term: sp.Expr) -> tuple[sp.Expr, list[int]]:
    """Split an additive sympy term into (scalar coupling factor, [S^z sites]).

    The Walsh fit in :mod:`rsqsi.bw` yields square-free S^z monomials, so each
    S^z symbol appears to the first power.
    """
    sites = []
    scalar = term
    for sym in term.free_symbols:
        name = str(sym)
        if name.startswith("Sz_"):
            sites.append(int(name[3:]))
            scalar = scalar.subs(sym, 1)
    return sp.sympify(sp.simplify(scalar)), sorted(sites)


def render_terms(result: BWResult) -> list[tuple[sp.Expr, C.Expression]]:
    """Return ``(coefficient, operator)`` pairs for every generated term."""
    out: list[tuple[sp.Expr, C.Expression]] = []

    # diagonal (S^z) part
    diag = sp.expand(result.diag)
    if diag != 0:
        for term in diag.as_ordered_terms():
            scalar, sites = _sz_monomial(term)
            if scalar == 0:
                continue
            ops = [_sz(s) for s in sites]
            expr = C.Expression(C.Term(*ops)) if ops else C.Expression(C.Term())
            out.append((scalar, expr))

    # off-diagonal (ladder) channels; each may carry a spectator S^z dressing
    for flips, coeff in result.offdiag.items():
        coeff = sp.expand(coeff)
        if coeff == 0:
            continue
        ladder_ops = [_ladder(s, d) for s, d in sorted(flips)]
        for term in coeff.as_ordered_terms():
            scalar, sites = _sz_monomial(term)
            if scalar == 0:
                continue
            ops = [_sz(s) for s in sites] + ladder_ops
            out.append((scalar, C.Expression(C.Term(*ops))))
    return out


def as_latex(result: BWResult) -> str:
    """A human-readable LaTeX-ish sum of the generated terms."""
    pieces = []
    for coeff, expr in render_terms(result):
        op = expr.as_latex().strip()
        # strip Commutation's leading '+1 ' unit coefficient
        if op.startswith("+1 "):
            op = op[3:]
        pieces.append(f"\\left({sp.latex(coeff)}\\right)\\, {op}".strip())
    return "  +  ".join(pieces) if pieces else "0"
