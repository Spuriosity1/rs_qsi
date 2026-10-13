r'''
Parent-lattice data structure for ice models: an *oriented 1-complex*.

Spins live on the 1-cells (links); the classical/parent Hamiltonian's
constrainers $C_t$ live on the 0-cells (vertices). The central object is the
boundary operator $\partial$: every link knows its boundary as a signed pair of
vertices, $\partial\ell = (+\mathrm{head}) - (-\mathrm{tail})$. The constrainer
at a vertex is the transpose (coboundary): the signed sum of the spins on the
links touching it.

Topology (incidence) and geometry (positions) are kept separate: `dfs`,
`link_neighbors` and `constrainer` touch only incidence; positions ride along on
`Vertex` for building and plotting only.

Stable, hashable IDs (dict keys, never list indices) mean deleting a link never
renumbers anything else.
'''
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Hashable

from sympy import Matrix, MatrixBase, diag, floor


# ---------------------------------------------------------------------------
# Cell-complex containers
# ---------------------------------------------------------------------------

@dataclass
class Vertex:
    """A 0-cell: a constrainer site. Carries whatever the physics needs."""
    id: Hashable
    sublattice: Hashable
    position: Matrix              # sympy column vector


@dataclass
class Link:
    """A 1-cell: a spin lives here. Oriented tail -> head."""
    id: Hashable
    tail: Hashable                  # 0-cell id
    head: Hashable                  # 0-cell id
    sublattice: Hashable = None

    def boundary(self) -> dict[Hashable, int]:
        r"""$\partial\ell$ as {vertex id: sign}."""
        return {self.tail: -1, self.head: +1}


# ---------------------------------------------------------------------------
# The lattice
# ---------------------------------------------------------------------------

@dataclass
class Lattice:
    verts: dict[Hashable, Vertex]
    links: dict[Hashable, Link]
    incident: dict[Hashable, set]   # vertex id -> set of link ids

    def constrainer(self, v: Hashable) -> dict[Hashable, int]:
        r"""Coefficients of $C_v$: {link id: sign} over links touching `v`."""
        return {l: self.links[l].boundary()[v] for l in self.incident[v]}


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

@dataclass
class PrimitiveCell:
    """All the lattice-specific data; the tiler (`build`) is generic.

    `sublattices` are the 0-cell positions within one cell. Each bond template
    is (from_sublattice, delta): an oriented link leaving the `from_sl` atom
    along the real-space vector `delta`; the destination sublattice and cell are
    detected from `delta` (see _resolve_bond). `lattice_vectors` embed the cells
    for geometry.
    """
    sublattices: list                       # positions, sympy column vectors
    bonds: list[tuple[int, MatrixBase]]                             # (from_sl, delta); see _resolve_bond
    lattice_vectors: Matrix | None = None   # columns are the basis vectors


def _supercell_matrix(shape) -> Matrix:
    """The supercell as an integer matrix S (columns = supercell vectors in
    primitive-cell units). A plain tuple/list means the diagonal box diag(shape);
    a Matrix is taken as-is (allowing non-diagonal supercells)."""
    return Matrix(shape) if isinstance(shape, MatrixBase) else diag(*shape)


def _fold(S, Sinv, n) -> tuple:
    """Canonical representative of cell `n` in Z^d / S Z^d: the integer point of
    the half-open parallelepiped S[0,1)^d that is congruent to n."""
    shift = S @ Matrix([floor(x) for x in Sinv @ Matrix(n)])
    return tuple(int(x) for x in (Matrix(n) - shift))


def _supercell_cells(S, Sinv) -> list:
    """The |det S| cells of Z^d / S Z^d: the integer points of S[0,1)^d, found by
    scanning the bounding box of that parallelepiped and keeping the interior."""
    dim = S.shape[0]
    lo = [sum(min(int(S[i, j]), 0) for j in range(dim)) for i in range(dim)]
    hi = [sum(max(int(S[i, j]), 0) for j in range(dim)) for i in range(dim)]
    box = itertools.product(*(range(lo[i], hi[i] + 1) for i in range(dim)))
    return [n for n in box if all(0 <= x < 1 for x in Sinv @ Matrix(n))]


def _position(cell, sl_pos, lattice_vectors):
    if lattice_vectors is None:
        raise TypeError("Cannot get _position when lattice_vectors is None")
    return lattice_vectors @ Matrix(cell) + sl_pos


def _resolve_bond(cell: PrimitiveCell, bond) -> tuple:
    r"""Turn a bond `(from_sl, delta)` into `(from_sl, to_sl, disp)`.

    `delta` is the real-space bond vector $p_\mathrm{head} - p_\mathrm{tail}$,
    pointing along the bond away from the `from_sl` atom. The head atom sits at
    $p_\mathrm{from} + \delta$; we find the sublattice `to_sl` and integer cell
    displacement `disp` it lands on, i.e. the `to_sl` for which
    $\mathrm{disp} = R^{-1}(p_\mathrm{from} + \delta - p_\mathrm{to})$ is an
    integer vector. Raises if `delta` lands on no sublattice image.
    """
    from_sl, delta = bond
    R = cell.lattice_vectors
    if R is None:
        raise TypeError("Cannot resolve a bond when lattice_vectors is None")
    Rinv = R.inv()
    head = cell.sublattices[from_sl] + delta
    for to_sl, p_to in enumerate(cell.sublattices):
        disp = Rinv @ (head - p_to)
        if all(x.is_integer for x in disp):
            return from_sl, to_sl, tuple(int(x) for x in disp)
    raise ValueError(
        f"bond delta {list(delta)} from sublattice {from_sl} lands on no "
        f"sublattice image (head position {list(head)})"
    )


def build(cell: PrimitiveCell, shape, periodic: bool = True) -> Lattice:
    """Tile `cell` over a supercell of primitive cells.

    `shape` is either a length-d tuple -- the diagonal box diag(shape) -- or an
    integer sympy Matrix S whose columns are the supercell vectors in primitive-
    cell units, allowing non-diagonal supercells (e.g. a cubic cell cut from an
    FCC primitive cell). The cells are the |det S| representatives of Z^d / S Z^d.

    Vertex id = (cell index tuple, sublattice index). Links get fresh integer
    ids. With `periodic`, a link's head cell is folded back into the supercell;
    otherwise a link whose head leaves the supercell is dropped (open boundary).
    """
    S = _supercell_matrix(shape)
    Sinv = S.inv()
    cells = _supercell_cells(S, Sinv)

    verts: dict[Hashable, Vertex] = {}
    for n in cells:
        for sl, sl_pos in enumerate(cell.sublattices):
            vid = (n, sl)
            verts[vid] = Vertex(vid, sl, _position(n, sl_pos, cell.lattice_vectors))

    links: dict[Hashable, Link] = {}
    incident: dict[Hashable, set] = {vid: set() for vid in verts}

    resolved = [_resolve_bond(cell, b) for b in cell.bonds]

    next_id = 0
    for n in cells:
        for from_sl, to_sl, disp in resolved:
            raw = tuple(n[k] + disp[k] for k in range(len(n)))
            m = _fold(S, Sinv, raw)
            if not periodic and m != raw:              # head left the supercell
                continue
            tail, head = (n, from_sl), (m, to_sl)
            links[next_id] = Link(next_id, tail, head, sublattice=from_sl)
            incident[tail].add(next_id)
            incident[head].add(next_id)
            next_id += 1

    return Lattice(verts, links, incident)


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def remove_link(lat: Lattice, lid: Hashable) -> Link:
    """Delete a link (a defect). Vertices stay; incidence is updated in place."""
    link = lat.links.pop(lid)
    lat.incident[link.tail].discard(lid)
    lat.incident[link.head].discard(lid)
    return link


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------

def dfs(start: Hashable,
        neighbors: Callable[[Hashable], set],
        visit: Callable[[Hashable], None] = lambda _: None) -> set:
    """Generic depth-first search; `neighbors` chooses what graph you walk."""
    seen, stack = set(), [start]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        visit(x)
        stack.extend(n for n in neighbors(x) if n not in seen)
    return seen


def link_neighbors(lat: Lattice, lid: Hashable) -> set:
    """Line-graph adjacency: links sharing a 0-cell with `lid`."""
    l = lat.links[lid]
    return {k for v in (l.tail, l.head) for k in lat.incident[v]} - {lid}
