"""Helpers to rebuild the Drogon fidelity-level inputs of the book experiments.

The coarse levels were produced by NORCE's legacy upscaler (``us.py``), which
is not public. Its outputs are shipped in ``data/levels``. This module holds
the facies-probability rule of the book and a reimplementation of the legacy
fine/coarse volume map (``map_fine_volumes_parfor``), used to check and
document the shipped maps.

The legacy map deliberately differs from an exact geometric overlap: fine
cells are candidates only when their *centre* lies inside the coarse cell's
axis-aligned bounding box, and the weights are volumes of intersecting convex
hulls. Inactive and collapsed cells take part as well. upscaling_v2's exact
tetrahedral overlap therefore yields different (and, for the book results,
wrong) averaging matrices.
"""

from __future__ import annotations

import multiprocessing
import os
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import KDTree

# Ordering of the eight corners of a cell in the legacy cornerpoint helper
# (PET ``utils.grid.cornerpoint.elem_vtcs_ndcs``).
HEX_PERM = np.array([4, 5, 7, 6, 0, 1, 3, 2])
CORNER_BFR = np.array([(k, j, i) for k in (0, 1) for j in (0, 1) for i in (0, 1)])[HEX_PERM]


def read_grdecl_keywords(path: Path, keywords: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Read numeric GRDECL keywords as float64, expanding ``n*value`` repeats."""
    wanted = {keyword.upper() for keyword in keywords}
    result: dict[str, list[float]] = {}
    current = None
    with open(path, errors="replace") as stream:
        for line in stream:
            code = line.split("--", 1)[0].strip()
            if not code:
                continue
            if current is None:
                name = code.split()[0].upper()
                if re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                    current = name if name in wanted else "_skip"
                    if current != "_skip":
                        result[current] = []
                    code = code[len(name):]
                else:
                    continue
            for token in code.replace("/", " / ").split():
                if token == "/":
                    current = None
                    break
                if current == "_skip":
                    continue
                if "*" in token:
                    count, value = token.split("*", 1)
                    result[current].extend([float(value)] * int(count))
                elif re.fullmatch(r"[-+0-9.eEdD]+", token):
                    result[current].append(float(token.replace("D", "E").replace("d", "e")))
                # non-numeric items (SPECGRID's radial flag) carry no geometry
    missing = wanted - result.keys()
    if missing:
        raise KeyError(f"{', '.join(sorted(missing))} not found in {path}")
    return {name: np.asarray(values) for name, values in result.items()}


def read_geometry(path: Path) -> dict[str, np.ndarray]:
    """Return SPECGRID dimensions and COORD/ZCORN shaped like the legacy code."""
    data = read_grdecl_keywords(path, ("SPECGRID", "COORD", "ZCORN"))
    ni, nj, nk = (int(value) for value in data["SPECGRID"][:3])
    return {
        "dims": (ni, nj, nk),
        "coord": data["COORD"].reshape(nj + 1, ni + 1, 2, 3),
        "zcorn": data["ZCORN"].reshape(nk, 2, nj, 2, ni, 2),
    }


def legacy_fine_geometry(path: Path) -> dict[str, np.ndarray]:
    """Fine geometry as the legacy upscaler held it in memory.

    It read COORD/ZCORN from EGRID (float32), shifted the pillars by the
    MAPAXES origin in float32, and replaced vertically collapsed pillars by
    the mean top and bottom depth.
    """
    data = read_grdecl_keywords(path, ("SPECGRID", "MAPAXES", "COORD", "ZCORN"))
    ni, nj, nk = (int(value) for value in data["SPECGRID"][:3])
    coord = data["COORD"].astype(np.float32).reshape(nj + 1, ni + 1, 2, 3)
    origin = data["MAPAXES"].astype(np.float32)[2:4]
    for end in (0, 1):
        coord[:, :, end, :2] += origin
    dz = coord[:, :, 1, 2] - coord[:, :, 0, 2]
    collapsed = np.nonzero(dz == 0)
    top_mean, bottom_mean = np.mean(coord[:, :, 0, 2]), np.mean(coord[:, :, 1, 2])
    coord[collapsed[0], collapsed[1], 0, 2] = top_mean
    coord[collapsed[0], collapsed[1], 1, 2] = bottom_mean
    zcorn = data["ZCORN"].astype(np.float32).reshape(nk, 2, nj, 2, ni, 2)
    return {"dims": (ni, nj, nk), "coord": coord, "zcorn": zcorn}


def cell_corners(coord: np.ndarray, zcorn: np.ndarray) -> np.ndarray:
    """Corner coordinates per cell, shape ``(ncell, 8, 3)``, in legacy order.

    Each corner lies on its pillar at the ZCORN depth (PET's
    ``corner_coordinates``); arithmetic stays in the input precision.
    """
    nk, _, nj, _, ni, _ = zcorn.shape
    k, j, i = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij")
    k, j, i = (axis.reshape(-1, 1) for axis in (k, j, i))
    bk, bj, bi = (CORNER_BFR[:, axis][None, :] for axis in range(3))
    z = zcorn[k, bk, j, bj, i, bi]
    pj, pi = j + bj, i + bi
    top, bottom = coord[pj, pi, 0], coord[pj, pi, 1]
    dz = bottom[..., 2] - top[..., 2]
    x = (z - top[..., 2]) * ((bottom[..., 0] - top[..., 0]) / dz) + top[..., 0]
    y = (z - top[..., 2]) * ((bottom[..., 1] - top[..., 1]) / dz) + top[..., 1]
    return np.stack([x, y, z], axis=-1).astype(np.float64)


def grandy_volume(cell: np.ndarray) -> float:
    """Hexahedron volume with the legacy corner order (Grandy 1997)."""
    x = cell.T
    x3mx0, x5mx0, x6mx0 = x[:, 6] - x[:, 4], x[:, 1] - x[:, 4], x[:, 3] - x[:, 4]
    x7mx1, x7mx2, x7mx4 = x[:, 2] - x[:, 5], x[:, 2] - x[:, 7], x[:, 2] - x[:, 0]
    det1 = np.dot(x7mx1 + x6mx0, np.cross(x7mx2, x3mx0))
    det2 = np.dot(x6mx0, np.cross(x7mx2 + x5mx0, x7mx4))
    det3 = np.dot(x7mx1, np.cross(x5mx0, x7mx4 + x3mx0))
    return abs((det1 + det2 + det3) / 12.0)


def _intersection_volume(fine_cell: np.ndarray, coarse_cell: np.ndarray, coarse_mesh) -> float:
    import trimesh

    if np.any(fine_cell.max(axis=0) < coarse_cell.min(axis=0)) or np.any(
        coarse_cell.max(axis=0) < fine_cell.min(axis=0)
    ):
        return 0.0
    try:
        fine_mesh = trimesh.convex.convex_hull(fine_cell)
        intersection = trimesh.boolean.intersection([fine_mesh, coarse_mesh])
    except Exception:  # degenerate hulls; the legacy run counted them as zero
        return 0.0
    return 0.0 if intersection is None else float(intersection.volume)


def _map_coarse_cells(indices, coarse_cells, fine_cells, fine_centers):
    import trimesh

    tree = KDTree(fine_centers)
    rows, columns, weights = [], [], []
    coarse_volumes = np.zeros(len(indices))
    for position, n in enumerate(indices):
        coarse_cell = coarse_cells[n]
        coarse_mesh = trimesh.convex.convex_hull(coarse_cell)
        coarse_volume = grandy_volume(coarse_cell)
        coarse_volumes[position] = coarse_volume
        center = coarse_cell.mean(axis=0)
        half = (coarse_cell.max(axis=0) - coarse_cell.min(axis=0)) / 2
        candidates = [
            idx for idx in tree.query_ball_point(center, r=np.linalg.norm(half))
            if np.all(np.abs(fine_centers[idx] - center) <= half)
        ]
        cell_columns, cell_weights = [], []
        for idx in candidates:
            volume = _intersection_volume(fine_cells[idx], coarse_cell, coarse_mesh)
            if volume > 0:
                cell_columns.append(idx)
                cell_weights.append(volume)
        total = sum(cell_weights)
        if coarse_volume > 0 and total > coarse_volume:
            cell_weights = [w * coarse_volume / total for w in cell_weights]
        rows.extend([n] * len(cell_columns))
        columns.extend(cell_columns)
        weights.extend(cell_weights)
    return rows, columns, weights, coarse_volumes


_WORKER_DATA = {}


def _init_worker(cpus, data):
    # Pin each worker to one CPU: manifold3d's TBB pool otherwise starts a
    # thread per core in every worker process.
    os.sched_setaffinity(0, {cpus.get()})
    _WORKER_DATA.update(data)


def _map_block(block):
    return _map_coarse_cells(block, _WORKER_DATA["coarse"], _WORKER_DATA["fine"], _WORKER_DATA["centers"])


def legacy_volume_map(fine: dict, coarse: dict, n_jobs: int | None = None, chunks: int = 512):
    """Return the legacy (volume, row-normalized mean, coarse volume) maps."""
    fine_cells = cell_corners(fine["coord"], fine["zcorn"])
    coarse_cells = cell_corners(coarse["coord"], coarse["zcorn"])
    fine_centers = fine_cells.mean(axis=1)
    n_coarse, n_fine = len(coarse_cells), len(fine_cells)
    available = sorted(os.sched_getaffinity(0))
    n_jobs = min(n_jobs or len(available), len(available))
    context = multiprocessing.get_context("fork")
    cpus = context.Queue()
    for cpu in available[:n_jobs]:
        cpus.put(cpu)
    data = {"coarse": coarse_cells, "fine": fine_cells, "centers": fine_centers}
    blocks = np.array_split(np.arange(n_coarse), min(chunks, n_coarse))
    with context.Pool(n_jobs, initializer=_init_worker, initargs=(cpus, data)) as pool:
        parts = pool.map(_map_block, blocks)
    rows = np.concatenate([np.asarray(part[0], dtype=np.int64) for part in parts])
    columns = np.concatenate([np.asarray(part[1], dtype=np.int64) for part in parts])
    weights = np.concatenate([np.asarray(part[2], dtype=float) for part in parts])
    coarse_volume = np.concatenate([part[3] for part in parts])
    volume = sparse.csr_matrix((weights, (rows, columns)), shape=(n_coarse, n_fine))
    row_sum = np.asarray(volume.sum(axis=1)).ravel()
    scale = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
    mean = (sparse.diags(scale) @ volume).tocsr()
    return volume, mean, coarse_volume


def facies_probabilities(satnum: np.ndarray, active: np.ndarray) -> dict[str, np.ndarray]:
    """Fine-grid SATNUM probability fields of the book's truncation scheme.

    Each active cell gets probability 0.8 for its reference SATNUM and shares
    0.2 equally among the other four SATNUM values of its group; inactive
    cells are flagged with -1.
    """
    groups = ((1, 2, 7, 8, 12), (3, 4, 9, 10, 11))
    fields = {}
    for value in range(1, 13):
        field = np.full(satnum.size, -1.0)
        field[active] = 0.0
        for group in groups:
            if value in group:
                cells = active & np.isin(satnum, group)
                field[cells] = np.where(satnum[cells] == value, 0.8, 0.2 / 4)
        fields[f"p{value}"] = field
    return fields
