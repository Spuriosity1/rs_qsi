"""Make the repo root importable so `import rsqsi` resolves without an install.

The package uses relative imports internally (e.g. `from .lattice import ...`),
so tests import it as `from rsqsi import ...` -- the repo root must be on
sys.path for that to work when the package isn't pip-installed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
