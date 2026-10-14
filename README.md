# rsqsi

`rsqsi` is a Python engine that generates terms of the Rayleigh–Schrödinger / Brillouin–Wigner series for the effective Hamiltonian of quantum spin ice, treating generalised ("constrainer") ice models in a chain-complex formalism and producing symbolic effective operators to arbitrary order in an off-diagonal perturbation; its couplings and virtual energies are kept fully symbolic (per-sublattice and, in general, per-bond).

## Installation

From the repository root:

```bash
pip install -e .
```

## Dependencies

- Python >= 3.10
- [`sympy`](https://pypi.org/project/sympy/)
- [`numpy`](https://pypi.org/project/numpy/)

These are declared in `pyproject.toml`. The optional `test` extra pulls in `pytest`.

## Documentation

All other information — theory, formalism, derivations, the generating procedure, and usage details — lives on the project website:

**https://spuriosity1.github.io/rs_qsi/**
