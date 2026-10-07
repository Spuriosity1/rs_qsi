"""Brillouin-Wigner connected-loop term generator.

Implements the boxed "Procedure for calculating H_eff" from the thesis for the
BW piece  V (Q G0 V)^(n-1)  evaluated at E0, restricted to a connected local
cluster (linked-cluster theorem).

Method.  The effective operator is local, so we reconstruct it from its action
on the ``support`` sites alone.  For every S^z pattern ``sigma`` on the support
(each a would-be ice state), we sum over all length-N ordered sequences of the
available string terms that

  * start from ``sigma`` without any ladder operator annihilating,
  * keep every *intermediate* state out of the model space (some tetra charged),
  * return every support-incident tetra to zero charge at the end (closed loop),
  * and have connected support.

Each surviving ordering contributes  (prod couplings) * prod_k (-1 / E_k), where
E_k is the virtual spinon energy after k hops.  Because the starting ``sigma``
is (locally) an ice state, those energies depend only on the hop-induced charge
changes and are therefore background-independent.

The accumulated amplitudes are classified by their net spin flip and fitted (via
a Walsh/Fourier transform) back into operator form: a diagonal polynomial in the
S^z of the support, and one ladder monomial per off-diagonal flip channel
(carrying a possible S^z dressing on the spectator sites).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import sympy as sp

from .lattice_model import IceModel
from .states import apply_term
from .terms import StringTerm


def Sz(site: int) -> sp.Symbol:
    return sp.Symbol(f"Sz_{site}")


@dataclass
class BWResult:
    support: tuple[int, ...]
    diag: sp.Expr                                  # polynomial in Sz_i
    offdiag: dict[frozenset, sp.Expr] = field(default_factory=dict)
    # key: frozenset of (site, +-1) ladder factors; value: Sz-dressing coefficient

    def ring_coefficient(self) -> sp.Expr:
        """Coefficient of the ring-exchange operator O_p (flips every support
        site).  O_p and its conjugate O_p^dagger appear as two channels with
        equal coefficients; this returns that common single-channel value."""
        full = frozenset(self.support)
        coeffs = [sp.simplify(c) for f, c in self.offdiag.items()
                  if frozenset(s for s, _ in f) == full]
        if not coeffs:
            return sp.Integer(0)
        first = coeffs[0]
        for c in coeffs[1:]:
            assert sp.simplify(c - first) == 0, \
                "O_p and O_p^dagger channels disagree"
        return first


def _connected(adjacency: dict[int, set[int]], sites: frozenset) -> bool:
    """Are ``sites`` connected under the lattice interaction graph ``adjacency``
    (two sites adjacent iff they share a tetrahedron)?  This is the linked-cluster
    notion of connectedness, independent of which hops were chosen."""
    if not sites:
        return False
    target = set(sites)
    start = next(iter(target))
    seen = {start}
    stack = [start]
    while stack:
        x = stack.pop()
        for y in adjacency.get(x, ()):
            if y in target and y not in seen:
                seen.add(y)
                stack.append(y)
    return seen == target


def _delta_charge_energy(model: IceModel, start: dict[int, int],
                         cur: dict[int, int], incident_tetras: list[int]
                         ) -> sp.Expr:
    """Virtual spinon energy of ``cur`` relative to the ice state ``start``.

    E = sum_t Jzz_t (dQ_t)^2 / 2, dQ_t = sum_{i in t cap support} (cur_i-start_i)/2.
    Background-independent because ``start`` is an ice state (its charges vanish).
    """
    E = sp.Integer(0)
    for ti in incident_tetras:
        t = model.tetras[ti]
        dq = sum((cur[i] - start[i]) for i in t.members if i in cur)
        if dq != 0:
            q = sp.Rational(dq, 2)
            E += model.jzz[t.kind] * q**2 / 2
    return E


def _is_neutral(model: IceModel, start: dict[int, int], cur: dict[int, int],
                incident_tetras: list[int]) -> bool:
    for ti in incident_tetras:
        t = model.tetras[ti]
        if sum((cur[i] - start[i]) for i in t.members if i in cur) != 0:
            return False
    return True


def _walsh_fit(values: dict[tuple[int, ...], sp.Expr],
               sites: tuple[int, ...]) -> sp.Expr:
    """Expand a function on {+-1}^sites (keyed by the sigma tuple, aligned with
    ``sites``) into the S^z operator basis.  Uses sigma_i = 2 Sz_i."""
    n = len(sites)
    if n == 0:
        return list(values.values())[0] if values else sp.Integer(0)
    norm = sp.Rational(1, 2**n)
    out = sp.Integer(0)
    for subset in itertools.chain.from_iterable(
            itertools.combinations(range(n), r) for r in range(n + 1)):
        a = sp.Integer(0)
        for sigma, val in values.items():
            par = 1
            for j in subset:
                par *= sigma[j]
            a += val * par
        a *= norm
        a = sp.simplify(a)
        if a != 0:
            mono = sp.Integer(1)
            for j in subset:
                mono *= 2 * Sz(sites[j])     # sigma_j = 2 Sz_j
            out += a * mono
    return sp.expand(out)


def generate_bw(model: IceModel, terms: list[StringTerm], N: int,
                support: frozenset) -> BWResult:
    """Generate the order-N BW effective operator on ``support``."""
    support_t = tuple(sorted(support))
    avail = [t for t in terms if t.support <= support]

    incident = sorted({ti for s in support for ti in model.site_tetras[s]})

    # interaction-graph adjacency among support sites (share a tetrahedron)
    adjacency: dict[int, set[int]] = {s: set() for s in support_t}
    for ti in incident:
        mem = [i for i in model.tetras[ti].members if i in support]
        for i in mem:
            for j in mem:
                if i != j:
                    adjacency[i].add(j)

    # amplitudes[final_cfg_tuple][start_cfg_tuple] = sympy coefficient
    amp: dict[tuple, dict[tuple, sp.Expr]] = {}

    def cfg_tuple(cfg):
        return tuple(cfg[s] for s in support_t)

    def dfs(start, cur, depth, coeff, touched_sites):
        if depth == N:
            if _is_neutral(model, start, cur, incident) and \
                    _connected(adjacency, frozenset(touched_sites)):
                fkey = cfg_tuple(cur)
                skey = cfg_tuple(start)
                amp.setdefault(fkey, {})
                amp[fkey][skey] = amp[fkey].get(skey, sp.Integer(0)) + coeff
            return
        for term in avail:
            res = apply_term(cur, term)
            if res is None:
                continue
            nxt, diag_factor = res
            # intermediates (before the last hop) must be out of the model space
            if depth < N - 1 and _is_neutral(model, start, nxt, incident):
                continue
            E = _delta_charge_energy(model, start, nxt, incident)
            if depth < N - 1:
                if E == 0:
                    continue
                prop = -1 / E
            else:
                prop = sp.Integer(1)   # no propagator after the final hop
            dfs(start, nxt, depth + 1,
                coeff * term.coupling * prop * diag_factor,
                touched_sites | term.support)

    for sigma in itertools.product((+1, -1), repeat=len(support_t)):
        start = dict(zip(support_t, sigma))
        dfs(start, dict(start), 0, sp.Integer(1), set())

    # classify
    diag_vals: dict[tuple, sp.Expr] = {}
    offdiag: dict[frozenset, dict[tuple, sp.Expr]] = {}
    for fkey, starts in amp.items():
        fcfg = dict(zip(support_t, fkey))
        for skey, coeff in starts.items():
            scfg = dict(zip(support_t, skey))
            flips = frozenset((s, fcfg[s]) for s in support_t if fcfg[s] != scfg[s])
            if not flips:
                diag_vals[skey] = diag_vals.get(skey, sp.Integer(0)) + coeff
            else:
                offdiag.setdefault(flips, {})[skey] = coeff

    diag = _walsh_fit(diag_vals, support_t) if diag_vals else sp.Integer(0)

    # for each off-diagonal channel, fit the spectator-S^z dressing
    off_ops: dict[frozenset, sp.Expr] = {}
    for flips, starts in offdiag.items():
        flipped = {s for s, _ in flips}
        spectators = tuple(s for s in support_t if s not in flipped)
        # the flipped sites' initial values are forced; index remaining by spectators
        vals: dict[tuple[int, ...], sp.Expr] = {}
        for skey, coeff in starts.items():
            scfg = dict(zip(support_t, skey))
            vals[tuple(scfg[s] for s in spectators)] = coeff
        off_ops[flips] = sp.simplify(_walsh_fit(vals, spectators))

    return BWResult(support=support_t, diag=diag, offdiag=off_ops)
