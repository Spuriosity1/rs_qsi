import itertools

from .lattice import Lattice, dfs, link_neighbors
import numpy as np
import sympy as sp



# ---------------------------------------------------------------------------
# Homological scaffolding: the boundary operator d_1
# ---------------------------------------------------------------------------
#
# Spins live on 1-cells (links); an operator's od_rep is a 1-chain c in Z^L.
# Applying it deposits spinons (charge) on the 0-cells (tetrahedra), and the
# charge at each vertex is exactly the boundary d c. A process is a closed loop
# / ring iff it leaves no net spinon, i.e. d c = 0 -- so the closed loops are
# the integer cycle space Z_1 = ker d_1, not the chains that are themselves zero.


def incidence_matrix(lattice, verts=None, links=None) -> sp.SparseMatrix:
    r'''The boundary operator d_1 : C_1 -> C_0 as an integer sparse matrix.

    Shape (|V|, |L|), rows in `verts` order and columns in `links` order (both
    default to the lattice's own dict order). Entry d[v, l] = +1 if v is l's
    head, -1 if l's tail. A 1-chain c (column in `links` order) is a 1-cycle iff
    d * c == 0.
    '''
    verts = list(lattice.verts) if verts is None else verts
    links = list(lattice.links) if links is None else links
    vrow = {v: i for i, v in enumerate(verts)}
    data = {}
    for j, l in enumerate(links):
        for v, sign in lattice.links[l].boundary().items():
            data[(vrow[v], j)] = sign
    return sp.SparseMatrix(len(verts), len(links), data)


# --------------------------------------------------------------------------
# Error declaration boilerplate
# --------------------------------------------------------------------------


class OperatorSpecificationError(Exception):
    '''Exception raised when operator construction fails.'''
    def __init__(self, od, d):
        overlap = np.argwhere(od * d != 0)
        self.message = 'Off-diagonal and diagonal parts of spin spec overlap.\n'
        self.message += f'od={od}, d={d}, overlapping indices={overlap}'
        super().__init__(self.message)



# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------

class Operator:
    '''
    Representation of a string-type operator on an ice lattice.

    coeff: Generic SymPy coefficient.
    od_rep: Factors of +1 indicate S+, -1 indicate S-. 0 indicates no operator.
    d_rep: Dangling Sz's. 1 indicates Sz present, 0 otherwise.

    Note that d_rep and od_rep must be disjoint -- an error is thrown on
    instantiation if they overlap.
    '''
    def __init__(self, coeff: sp.Expr, od_rep, d_rep):
        self.coeff: sp.Expr  = coeff
        self.od_rep: np.ndarray = np.array(od_rep, dtype=int)
        self.d_rep: np.ndarray = np.array(d_rep, dtype=int)

        if np.any(self.od_rep*self.d_rep):
            raise OperatorSpecificationError(self.od_rep, self.d_rep)


class Model:
    '''Hamiltonian terms decorating a frozen lattice.

    The lattice must be fully built and pruned *before* the Model is created:
    from here on `self.links`/`self.verts` fix a positional index for every
    1-chain, and the boundary operator `self.incidence` (d_1) is built once.

    `self.constraints` give the prefactor to the base Hamiltonian, H_0 =
    sum_{v = 1...V} self.constraints[v] [sum_l incidence_matrix_{v, l} Sz[l]]^2

    `self.operators` lists every term explicitly: each Hermitian-conjugate
    partner (coeff -> conj, S+ <-> S-) is stored as its own entry.
    '''
    def __init__(self, lattice: Lattice):
        self.lattice = lattice
        self.operators = []
        self.links = list(lattice.links)                 # canonical column order
        self.verts = list(lattice.verts)                 # canonical row order
        self.constraints = [1 for _ in self.verts]     # constraint strengths
        self.incidence = incidence_matrix(lattice, self.verts, self.links)

    def set_constraint(self, coeff: sp.Expr, vert_id: int):
        '''Set the H0 strength on vertex index `vert_id` (position in self.verts).'''
        self.constraints[vert_id] = coeff

    def add_operator(self, coeff: sp.Expr, od_rep, d_rep):
        '''coeff: SymPy coefficient. od_rep/d_rep: vectors over self.links,
        +1 per S+, -1 per S-, 0 otherwise (d_rep marks dangling Sz).'''
        assert len(od_rep) == len(self.links)
        assert len(d_rep) == len(self.links)
        self.operators.append(Operator(coeff, od_rep, d_rep))

    def chain_matrix(self) -> sp.Matrix:
        '''od_reps stacked as the columns of an |L| x T integer matrix, so
        `self.incidence * self.chain_matrix()` has a zero column exactly where
        that operator's 1-chain is already closed.'''
        return sp.Matrix([list(o.od_rep) for o in self.operators]).T

    def boundary(self, chain) -> sp.MatrixBase:
        '''The 0-chain d c: charge deposited on each vertex (in self.verts order).'''
        return self.incidence * sp.Matrix(chain)

    def is_cycle(self, chain) -> bool:
        '''True iff d c == 0: `chain` is a closed loop (no net spinons).'''
        return self.boundary(chain) == sp.zeros(len(self.verts), 1)


def find_closed_loops(model: Model, order: int) -> list[tuple[int]]:
    '''
    All closed loops built from exactly `order` operator applications.

    Write |L| = # links, |V| = # vertices, T = # terms.
    Let G = chain_matrix (|L| x T, operator od_reps as columns) and iota = d_1.
    Then Q = iota @ G (|V| x T) is the charge each operator deposits on the
    tetrahedra. A multiset of operators with count vector x (in Z^T_>=0) is a
    loop iff it leaves no net spinon, i.e. Q x = 0, with |x|_1 = order.

    So the loops are the non-negative integer points of ker Q on the order-N
    shell. That is a bounded lattice search, not a linear solve: we never touch
    floats (an integer null space through ℝ would need exact rescaling *and*
    saturation to be trustworthy). We enumerate x directly, operator counts in
    non-decreasing index order (canonical -> each multiset once), pruning any
    branch whose running boundary cannot reach 0 within the remaining budget.

    Returns a list of sorted operator-index tuples (an operator may repeat); the
    composite 1-chain is sum(model.operators[a].od_rep for a in loop). Assumes
    model.operators already contains the Hermitian conjugates (S+<->S-), so x
    stays non-negative. Connectivity is NOT imposed -- a disjoint pair of small
    rings is also a valid Q x = 0 point; filter on support adjacency downstream
    if you want only linked clusters.
    '''
    if order < 1:
        return []

    Q = np.array((model.incidence * model.chain_matrix()).tolist(), dtype=int)   # |V| x T
    active = [a for a, o in enumerate(model.operators) if o.od_rep.any()]         # drop pure-Sz ops
    step = max((int(np.abs(Q[:, a]).sum()) for a in active), default=0)           # max |db|_1

    loops = []

    def grow(used, q, start):
        if len(used) == order:
            if not q.any():                                      # Q x == 0: closed
                loops.append(tuple(used))
            return
        if step and int(np.abs(q).sum()) > (order - len(used)) * step:
            return                                               # cannot close in time
        for i in range(start, len(active)):                      # non-decreasing -> unique multisets
            a = active[i]
            grow(used + [a], q + Q[:, a], i)

    grow([], np.zeros(Q.shape[0], dtype=int), 0)
    return loops


def Sz(link) -> sp.Symbol:
    '''Symbol for a dangling S^z on a link (spectator factor from a d_rep).'''
    return sp.Symbol(f"Sz_{link}")


# --------------------------------------------------------------------------
# Rayleigh-Schroedinger counterterm helpers
# --------------------------------------------------------------------------
#
# BW evaluates G0 = (E - H0)^-1 at E0 = 0 (the bare scalar resolvent g = -1/E_k).
# RS expands G0(E) in the energy corrections E^(n):
#   G0^(a) = sum_{j=1..a} sum_{compositions (n_1..n_j) of a} (prod E^(n_i)) g^{j+1}
#   H_eff^(n) = sum over {m factors of V, m + sum a_i = n} of V Q G0^(a_1) V ... V
# So a counterterm is just higher powers of the SAME scalar resolvent, weighted
# by products of lower-order E^(n): a finite polynomial, no derivatives.


def _integer_compositions(a: int):
    '''All ordered compositions of a >= 1 into positive parts; () for a == 0.'''
    if a == 0:
        yield ()
        return
    for first in range(1, a + 1):
        for rest in _integer_compositions(a - first):
            yield (first,) + rest


def _g0_dressing(a: int, Ecorr: dict) -> list:
    '''The resolvent counterterm G0^(a) on a scalar resolvent g, as a list of
    (weight, power) pairs meaning sum_i weight_i * g**power_i. a == 0 is the bare
    G0 = g -> [(1, 1)]. `Ecorr` maps order k -> energy correction E^(k).'''
    if a == 0:
        return [(sp.Integer(1), 1)]
    by_power: dict = {}
    for comp in _integer_compositions(a):
        power = len(comp) + 1
        weight = sp.prod([Ecorr[p] for p in comp])
        by_power[power] = by_power.get(power, sp.Integer(0)) + weight
    return [(w, p) for p, w in sorted(by_power.items())]


def _slot_distributions(num_slots: int, budget: int):
    '''All (a_1..a_{num_slots}) with a_i >= 0 summing to `budget` (weak
    compositions) -- how to spread a counterterm budget over a loop's slots.'''
    if num_slots == 0:
        if budget == 0:
            yield ()
        return
    for first in range(budget + 1):
        for rest in _slot_distributions(num_slots - 1, budget - first):
            yield (first,) + rest


def _walsh_fit(values: dict, sites: tuple) -> sp.Expr:
    '''Expand a function on {+-1}^sites (keyed by the sigma tuple, aligned with
    `sites`) into the S^z operator basis, using sigma_i = 2 Sz_i. `sites` are
    link ids. Ported from the validated bw.py.'''
    n = len(sites)
    if n == 0:
        return next(iter(values.values())) if values else sp.Integer(0)
    norm = sp.Rational(1, 2**n)
    out = sp.Integer(0)
    for r in range(n + 1):
        for subset in itertools.combinations(range(n), r):
            coeff = sp.Integer(0)
            for sigma, val in values.items():
                par = 1
                for j in subset:
                    par *= sigma[j]
                coeff += val * par
            coeff = sp.cancel(norm * coeff)
            if coeff != 0:
                mono = sp.Integer(1)
                for j in subset:
                    mono *= 2 * Sz(sites[j])          # sigma_j = 2 Sz_j
                out += coeff * mono
    return sp.expand(out)


def calc_coeff(model: Model, loop: tuple[int]) -> sp.Expr:
    '''
    Scalar BW amplitude of one `loop` (a multiset of operator indices from
    find_closed_loops): the sum over the T! orderings of its operators of

        prod(op.coeff)  *  prod_{k=1}^{T-1} ( -1 / E_k ),

    where E_k = sum_v constraints[v] Q_v^2 is the virtual spinon energy of the
    k-th intermediate (Q = charge after the first k ops, background-independent).

    An ordering is dropped if it drives a spin past its spin-1/2 range (a link's
    running |od| > 1, the background-free annihilation condition) or if an
    intermediate returns to the ground manifold (E_k = 0) before the last step.
    The composite ladder operator is implied by the loop's net flip; dangling S^z
    (d_rep) factors ride along as symbolic Sz spectators. (A spin appearing in
    both a d_rep and an od_rep would make that factor ordering-dependent -- not
    handled here.)

    All per-ordering work is restricted to the loop's own support -- the links it
    flips and the tetrahedra those links touch (the only ones that can carry a
    spinon). The full lattice is never materialised in the hot loop.
    '''
    ops = [model.operators[a] for a in loop]
    T = len(ops)
    coupling = sp.prod([o.coeff for o in ops])
    spectators = sp.prod([Sz(model.links[i]) for o in ops
                          for i in np.nonzero(o.d_rep)[0]])

    # support: the links the loop flips, and the vertices those links touch
    supp = sorted({i for o in ops for i in np.nonzero(o.od_rep)[0]})
    vloc: dict = {}
    for p in supp:
        for vid in model.lattice.links[model.links[p]].boundary():
            vloc.setdefault(vid, len(vloc))
    Dr = np.zeros((len(vloc), len(supp)), dtype=int)        # incidence on the support
    for col, p in enumerate(supp):
        for vid, sign in model.lattice.links[model.links[p]].boundary().items():
            Dr[vloc[vid], col] = sign
    vpos = {vid: i for i, vid in enumerate(model.verts)}
    con = [model.constraints[vpos[vid]] for vid in vloc]    # H0 strength per support vert
    od_r = [o.od_rep[supp] for o in ops]                    # each op's flips on the support
    dq = [Dr @ r for r in od_r]                             # charge each op deposits

    total = sp.Integer(0)
    for perm in itertools.permutations(range(T)):
        chain = np.zeros(len(supp), dtype=int)
        charge = np.zeros(len(vloc), dtype=int)
        denom = sp.Integer(1)
        ok = True
        for k, a in enumerate(perm):
            chain = chain + od_r[a]
            charge = charge + dq[a]
            if np.abs(chain).max(initial=0) > 1:            # spin-1/2 annihilation
                ok = False
                break
            if k < T - 1:                                   # propagator on intermediates only
                nz = np.nonzero(charge)[0]
                if nz.size == 0:                            # back in the ground manifold
                    ok = False
                    break
                denom *= -sum(con[i] * int(charge[i])**2 for i in nz)
        if ok:
            total += 1 / denom

    return sp.cancel(coupling * spectators * total)


def walsh_amplitude(model: Model, loop: tuple[int],
                    slot_orders: tuple | None = None,
                    Ecorr: dict | None = None) -> sp.Expr:
    '''Operator-valued amplitude of one `loop`, with its T-1 propagator slots
    dressed by counterterms of orders `slot_orders` (default all bare).

    Unlike calc_coeff (a scalar), this resolves the S^z dependence: it brute-force
    enumerates the starting S^z of every touched spin, applies the ladder operators
    honestly (S+ needs a down spin, S- an up spin -- orderings that annihilate for
    a given background drop out), carries dangling-S^z (d_rep) factors as the
    current spin eigenvalue, and Walsh-fits the result back to Sz symbols.

    The loop's net flip `composite` defines its ladder channel O (links with
    composite != 0); those spins are forced to the unique start O acts on, so the
    returned S^z polynomial is the *dressing* of O (and for a diagonal loop,
    composite == 0, the whole diagonal operator -- its constant term is E^(n)).
    Intermediate energies are the background-independent hop charges; the Q
    projector (no model-space return mid-process) is kept. Support-restricted.
    '''
    ops = [model.operators[a] for a in loop]
    T = len(ops)
    if slot_orders is None:
        slot_orders = (0,) * (T - 1)
    if Ecorr is None:
        Ecorr = {}
    coupling = sp.prod([o.coeff for o in ops])

    composite = sum((o.od_rep for o in ops),                # net flip = ladder channel
                    np.zeros(len(model.links), dtype=int))
    touched = sorted({i for o in ops
                      for i in (*np.nonzero(o.od_rep)[0], *np.nonzero(o.d_rep)[0])})
    tpos = {i: k for k, i in enumerate(touched)}            # link index -> state slot
    ladder = [i for i in touched if composite[i] != 0]      # net-flipped -> part of O
    fit = [i for i in touched if composite[i] == 0]         # Sz good quantum numbers
    forced = {tpos[i]: -int(np.sign(composite[i])) for i in ladder}  # start O acts on

    # charge / energy data on the od-support (background-independent)
    supp = sorted({i for o in ops for i in np.nonzero(o.od_rep)[0]})
    vloc: dict = {}
    for p in supp:
        for vid in model.lattice.links[model.links[p]].boundary():
            vloc.setdefault(vid, len(vloc))
    Dr = np.zeros((len(vloc), len(supp)), dtype=int)
    for col, p in enumerate(supp):
        for vid, sign in model.lattice.links[model.links[p]].boundary().items():
            Dr[vloc[vid], col] = sign
    vpos = {vid: i for i, vid in enumerate(model.verts)}
    con = [model.constraints[vpos[vid]] for vid in vloc]
    dq = [Dr @ o.od_rep[supp] for o in ops]

    values = {}
    for fit_sigma in itertools.product((1, -1), repeat=len(fit)):
        start = [0] * len(touched)
        for pos, val in forced.items():
            start[pos] = val
        for k, i in enumerate(fit):
            start[tpos[i]] = fit_sigma[k]
        amp = sp.Integer(0)
        for perm in itertools.permutations(range(T)):
            state = list(start)
            charge = np.zeros(len(vloc), dtype=int)
            factor = sp.Integer(1)
            ok = True
            for k, a in enumerate(perm):
                op = ops[a]
                for i in np.nonzero(op.od_rep)[0]:          # apply S+/S- honestly
                    s = int(op.od_rep[i])
                    w = tpos[i]
                    if state[w] != -s:                      # S+ needs down, S- up
                        ok = False
                        break
                    state[w] = -state[w]
                if not ok:
                    break
                charge = charge + dq[a]
                for i in np.nonzero(op.d_rep)[0]:           # dangling Sz eigenvalue
                    factor *= sp.Rational(state[tpos[i]], 2)
                if k < T - 1:                               # dressed propagator
                    nz = np.nonzero(charge)[0]
                    if nz.size == 0:                        # model-space return (Q)
                        ok = False
                        break
                    E_k = sum(con[j] * int(charge[j])**2 for j in nz)
                    g = sp.Integer(-1) / E_k
                    factor *= sum(w * g**p for w, p in _g0_dressing(slot_orders[k], Ecorr))
            if ok:
                amp += factor
        values[fit_sigma] = amp

    coeff = _walsh_fit(values, tuple(model.links[p] for p in fit))
    return sp.cancel(coupling * coeff)


# --------------------------------------------------------------------------
# Rayleigh-Schroedinger assembly
# --------------------------------------------------------------------------


def _net_flip(model: Model, loop: tuple[int]) -> tuple:
    '''The loop's composite 1-chain (its ladder channel) as a hashable key.'''
    comp = sum((model.operators[a].od_rep for a in loop),
               np.zeros(len(model.links), dtype=int))
    return tuple(int(x) for x in comp)


def is_connected_loop(model: Model, loop: tuple[int]) -> bool:
    '''True iff the loop's flipped links form one connected component in the
    line graph (links adjacent iff they share a tetrahedron). Linked-cluster
    filter -- find_closed_loops also returns disconnected multisets.'''
    supp = {model.links[i] for a in loop
            for i in np.nonzero(model.operators[a].od_rep)[0]}
    if len(supp) <= 1:
        return True
    start = next(iter(supp))
    seen = dfs(start, lambda l: link_neighbors(model.lattice, l) & supp)
    return seen == supp


def _identity_channel(expr: sp.Expr) -> sp.Expr:
    '''The structure-free (Sz-independent) part of a Walsh-fitted operator.'''
    expr = sp.expand(expr)
    zero = {s: 0 for s in expr.free_symbols if str(s).startswith("Sz_")}
    return sp.cancel(expr.subs(zero))


def energy_corrections(model: Model, max_order: int,
                       connectivity: str = 'linked') -> dict:
    '''The scalar RS energy corrections E^(1..max_order).

    E^(n) is the identity (Sz-free) channel of the order-n connected *diagonal*
    operator -- the sum over connected, net-chain-zero loops of order m in 2..n,
    with their T-1 propagator slots dressed by G0^(a) carrying a budget n-m,
    using the already-computed lower E^(k). Triangular: E^(n) needs only E^(k<n),
    and E^(1) = PVP = 0. (Off-diagonal / ring loops are operators, not energy.)

    This sums over the connected loops the enumeration returns; for a
    translationally-invariant cell divide by the number of reference cells for
    the per-site density. See [resolvent convention: scalar, bg-independent].
    '''
    Ecorr: dict = {1: sp.Integer(0)}
    for n in range(2, max_order + 1):
        diag = sp.Integer(0)
        for m in range(2, n + 1):
            for loop in find_closed_loops(model, m):
                if connectivity == 'linked' and not is_connected_loop(model, loop):
                    continue
                if any(_net_flip(model, loop)):          # off-diagonal -> not energy
                    continue
                for dist in _slot_distributions(m - 1, n - m):
                    diag += walsh_amplitude(model, loop, dist, Ecorr)
        Ecorr[n] = _identity_channel(diag)
    return Ecorr


def rs_effective_terms(model: Model, order: int,
                       connectivity: str = 'linked') -> dict:
    '''RS-corrected coefficient of each operator structure at total `order`.

    Returns {net-flip key -> Sz-polynomial coefficient}. For every connected loop
    L of order m in 2..order, add sum over slot distributions of (order-m) of
    walsh_amplitude(L, dist, Ecorr) to L's ladder channel (its net flip). RS
    introduces no new structures -- only coefficient corrections -- so the keys
    are exactly the channels the BW loops already produce; the Sz-free part of
    the diagonal (all-zero) key is the ground-state correction E^(order).
    '''
    Ecorr = energy_corrections(model, order, connectivity)
    terms: dict = {}
    for m in range(2, order + 1):
        for loop in find_closed_loops(model, m):
            if connectivity == 'linked' and not is_connected_loop(model, loop):
                continue
            key = _net_flip(model, loop)
            for dist in _slot_distributions(m - 1, order - m):
                terms[key] = terms.get(key, sp.Integer(0)) \
                    + walsh_amplitude(model, loop, dist, Ecorr)
    return {k: sp.cancel(v) for k, v in terms.items()}





