"""Make the rsqsi/ source directory importable under pytest.

The modules use bare intra-package imports (e.g. `import lattice`), so the
source dir itself must be on sys.path -- they are not imported as `rsqsi.*`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rsqsi"))
