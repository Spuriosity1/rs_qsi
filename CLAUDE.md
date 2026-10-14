# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`rsqsi` generates terms in the Rayleigh–Schrödinger / Brillouin–Wigner perturbation
series for the **effective Hamiltonian of (disordered, breathing) quantum spin ice**,
following the algorithm in the PhD thesis (Sanders 2026). Coefficients are symbolic
(sympy) in per-sublattice couplings, so disorder/breathing comes out automatically.
Validated against the thesis worked examples: N=2 Ising renormalisation, N=3 ring
exchange, and the 4th-order complex ring flip `g + i g'`.

> Note: `README.md` is **stale** — it documents an earlier architecture
> (`rsqsi/geometry/`, `terms.py`, `states.py`, `bw.py`, `operators.py`,
> `compute_RS_series.py`, a `Commutation` dependency). None of those exist now.
> Trust the source and this file, not the README.

## Commands

```bash
pip install -e .                      # sympy + numpy (Commutation in pyproject is unused)
python compute_XXZ_ringflip.py        # loop counts on a breathing-pyrochlore cell
python compute_complex_ringflip.py    # 4th-order complex ring flip, single hexagon
python compute_full_gig.py            # full g + i g' with real pyrochlore zeta phases
```

The driver scripts are the real entry points and all run clean. They import the
package form: `from rsqsi import lattice as lat, brillouin_wigner as bw`.

### Tests

```bash
pytest -q        # 9 tests
```

Tests and drivers both import the package form (`from rsqsi import …`); `conftest.py`
just puts the repo root on `sys.path` so `import rsqsi` resolves without an install.
Do **not** reintroduce bare module imports (`import lattice`) — `brillouin_wigner.py`
uses relative imports internally and breaks when imported outside the package. The
test file encodes the authoritative regression targets (thesis N=2/N=3 coeffs,
`_g0_dressing` counterterm weights, Walsh / energy-correction checks).

## Architecture

The whole thing is built on a **chain-complex (homological) view of the lattice**:
spins live on 1-cells, spinons (charge) on 0-cells, and "closed loop / ring" means
"integer 1-cycle" (zero boundary). Understanding that one idea unlocks the codebase.

### `rsqsi/lattice.py` — the oriented 1-complex

- `Vertex` = 0-cell (a tetrahedron / constrainer site); `Link` = 1-cell (a spin),
  oriented `tail → head`, with `boundary() = {tail:-1, head:+1}` (the operator ∂).
- `Lattice` holds `verts`, `links`, and `incident` (vertex→link-ids). IDs are stable
  hashable keys, never list indices, so deleting a link (`remove_link`, a *defect*)
  renumbers nothing.
- `PrimitiveCell` + `build(cell, shape, periodic)` is a generic tiler: `shape` is a
  diagonal box tuple or an integer sympy `Matrix` S whose columns are supercell
  vectors (lets you cut a cubic cell from the FCC/diamond primitive cell). Cells are
  the `|det S|` reps of Zᵈ/SZᵈ; `periodic` folds a link's head back in, otherwise
  drops boundary-crossing links. Bonds are `(from_sublattice, delta)` with `delta`
  the real-space bond vector; `_resolve_bond` finds the destination sublattice/cell.
- Topology (incidence) and geometry (positions) are deliberately separate: `dfs` and
  `link_neighbors` (line-graph adjacency) touch only incidence.

### `rsqsi/brillouin_wigner.py` — the engine (everything else)

Data model:
- `Operator(coeff, od_rep, d_rep)`: `coeff` is any sympy expr; `od_rep` is a vector
  over links (+1 per S⁺, −1 per S⁻, 0 else) — an element of the 1-chain group;
  `d_rep` marks *dangling* Sᶻ spectators. The two must be disjoint.
- `Model(lattice)`: freezes the lattice into a canonical link column order + vertex
  row order, builds the incidence matrix `d_1` once (`incidence_matrix`), and holds
  `operators` (**you must add both Hermitian conjugates** — S⁺↔S⁻) and per-vertex
  `constraints`. H₀ = Σ_v constraints[v]·Q_v², where Q_v = (∂c)_v. The physics
  convention throughout is **constraint strength = J_zz/2** so a virtual state with
  charges Q has energy Σ_v (J_zz,v/2)·Q_v².

Pipeline:
1. `find_closed_loops(model, order)` — all order-N multisets of operators whose
   composite chain is a 1-cycle (`Q x = 0`, no net spinon). Done as a **bounded
   integer lattice search** (never a float null-space), pruning branches that can't
   reach zero charge in the remaining budget. Connectivity is *not* imposed —
   disjoint ring pairs also satisfy `Q x = 0`; filter with `is_connected_loop`.
2. `calc_coeff(model, loop)` — the **scalar** BW amplitude: Σ over the T! orderings of
   `∏coeff · ∏(−1/E_k)`, with two guards — spin-½ annihilation (running |od| on a
   link > 1) and the Q-projector (no return to the ground manifold, E_k=0, before the
   last step). Support-restricted: only the loop's own links/tetrahedra are touched.
3. `walsh_amplitude(model, loop, slot_orders, Ecorr)` — the **operator-valued**
   amplitude. Brute-forces every background Sᶻ on touched spins, applies ladders
   honestly (S⁺ needs a down spin, orderings that annihilate drop out), reads dangling
   Sᶻ eigenvalues at their moment of action, and Walsh-fits the result back to `Sz_…`
   symbols (`_walsh_fit`, σ=2Sᶻ). `slot_orders`/`Ecorr` dress each propagator slot
   with an RS counterterm (`_g0_dressing`: G0^(a) as a polynomial in the scalar
   resolvent g weighted by products of lower E^(n); `_slot_distributions` spreads a
   budget over slots).
4. RS assembly: `energy_corrections(model, n)` builds E^(1..n) triangularly
   (E^(1)=PVP=0; E^(n) is the identity/Sᶻ-free channel of the order-n connected
   *diagonal* loops); `rs_effective_terms(model, order)` returns `{net-flip key →
   Sᶻ-polynomial coefficient}` — RS adds no new operator structures, only corrects the
   coefficients of the channels the BW loops already produce.

A loop's **net flip** (`_net_flip`, the composite 1-chain) is its ladder channel O_p
and the dict key under which contributions accumulate.

### `rsqsi/hamiltonian.py`

Essentially an empty stub (a planned higher-level "decompose a Hamiltonian into
diagonal/off-diagonal strings" layer). It uses a bare `import lattice`; ignore it as
a usage example.

## How a driver wires up a model (the usual pattern)

Build a lattice, make a `Model`, then for every pair of spins sharing a tetra add the
two hop operators `S⁺_i S⁻_j` and its h.c. with the sublattice's `J_pm`, and set each
vertex constraint to `J_zz,v/2`. See `compute_XXZ_ringflip.py` / the test fixture for
the canonical XXZ setup, and `compute_full_gig.py` for adding `zpm` operators with
dangling Sᶻ and complex zeta phases.
