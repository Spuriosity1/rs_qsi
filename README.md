# rsqsi — RS effective-Hamiltonian terms for disordered quantum spin ice

Computes terms in the Rayleigh–Schrödinger (RS) series for the effective
Hamiltonian of quantum spin ice, following the "Algorithm for generating terms"
in the PhD thesis (Sanders 2026, Ch.1 / Appendix B).

## Status — Increment 1

Implements the **Brillouin–Wigner connected-loop term** `V (Q G0 V)^(n-1)` at
`E0` (the part generating all new operator structures), with **symbolic**
coefficients in the per-sublattice (disorder-ready) couplings. Validated against
the breathing-pyrochlore XXZ worked example:

| Order | Result | Thesis |
|-------|--------|--------|
| N=2 | A-bond Ising `2 J_pm,A²/J_zz,B · SᶻSᶻ` (B-bond: A↔B) | `∝ J_pm²/J_zz SᶻSᶻ` |
| N=3 | ring exchange `3!·(J_pm,A³/J_zz,B² + J_pm,B³/J_zz,A²) · O_p` | identical |

Note the virtual spinons sit on the sublattice *opposite* to the active bond,
hence the A↔B crossing in the denominators — reproduced automatically from the
geometry.

## Arbitrary off-diagonal terms

The engine consumes any `list[StringTerm]`. A `StringTerm` is a product of ladder
operators (`factors`, each `(site, ±1)` for `S±`) with an optional set of diagonal
`S^z` factors (`sz_sites`) and a symbolic `coupling` — enough to express any
off-diagonal spin term. The `S^z` factors contribute their (configuration-dependent)
eigenvalues to the amplitude, so `J_{z±}`-type and decorated operators come out
automatically.

```python
from rsqsi.terms import StringTerm, hop, pmpm, field, zpm, place_on_bonds

hop(i, j, Jpm)                               # S⁺_i S⁻_j
pmpm(i, j, Jpmpm)                            # S⁺_i S⁺_j
field(j, h)                                  # [S⁺_j, S⁻_j]
zpm(n, o, +1, Jzpm)                          # S^z_n S⁺_o
StringTerm(((a,+1),(c,-1)), g, sz_sites=(k,))  # g · S^z_k S⁺_a S⁻_c   (+ .dagger())
```

Spread a local template across the lattice with `place_on_bonds(model, factory)` /
`place_on_sites(model, factory)` (factories receive the model, so they can reach
neighbours for dressings); `xxz_perturbation` and `transverse_field` are thin
wrappers. For anything the replicators don't cover, build the `StringTerm`s directly.

## Layout

- `rsqsi/geometry/` — vendored pyrochlore/diamond geometry (from
  `010_qsi_subspace_ED`): sites, tetrahedra (0-cells, A=up/B=down), hexagonal
  plaquettes, bonds.
- `rsqsi/lattice_model.py` — `IceModel` (general-ice core) + `build_pyrochlore`.
- `rsqsi/terms.py` — `StringTerm` (1-chain + coupling); `xxz_perturbation`.
- `rsqsi/states.py` — S^z configurations and ladder-operator action (spin-1/2).
- `rsqsi/bw.py` — the BW generator (loop enumeration + permutation sum →
  operator fit). Core engine; `generate_bw(model, V, N, support)`.
- `rsqsi/operators.py` — renders results as a single `Commutation` expression +
  LaTeX (the symbolic couplings ride along as each term's sympy multiplier,
  using `Commutation` 2.0.0's SymPy-scalar support).
- `compute_RS_series.py` — driver reproducing the N=2 and N=3 results.
- `tests/test_xxz.py` — validation suite.

## Run

```bash
pip install -e .          # or: pip install sympy numpy Commutation
python compute_RS_series.py
pytest -q
```

## Deferred to later increments

RS reducible counter-terms (`G0^(n)`, `E^(n)` self-consistency); the `J_{z±}`
complex ring flips (ortho/para → `g+ig'`) and decorated `ξ` terms; a numeric
disorder-realization evaluator; and further `IceModel` instances (1D TFIM
chain, …) via the general-ice core.
