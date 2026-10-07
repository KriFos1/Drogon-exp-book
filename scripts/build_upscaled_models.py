"""Regenerate Drogon fidelity grids and mapped inputs with upscaling-v2.

The Jupiter2 run supplied four fidelity resolutions. This builder keeps the
same cell counts (so the published ensemble/resource allocations remain
usable) but reconstructs the three coarse geometries with the v2 fault-aware
grid builder, derives fine/coarse volume intersections with v2, and remaps
the categorical regions, facies probabilities, schedules, and legacy fault
records onto the new grids.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
import shutil

import cwrap
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from resdata.grid import Grid
from resdata.resfile import ResdataFile
from upscaling import calculate_volume_overlap, fault_blocks, load_grid, upscale_schedule_file
from upscaling.fault_aware_upscale import process_hybrid_dims

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Levels"
OUTPUT = ROOT / "LevelsV2"
INCLUDE = ROOT / "include"
FINE_GRDECL = SOURCE / "Level104098" / "Grid.grdecl"
LEVELS = {
    300: (10, 15, 2),
    9000: (20, 30, 15),
    17500: (25, 35, 20),
    104098: (46, 73, 31),
}
REGION_INPUTS = {
    "EQLNUM": INCLUDE / "regions" / "drogon.eqlnum",
    "FIPNUM": INCLUDE / "regions" / "drogon.fipnum",
    "FIPZON": INCLUDE / "regions" / "drogon.fipzon",
    "PVTNUM": INCLUDE / "regions" / "drogon.pvtnum",
    "MULTNUM": INCLUDE / "grid" / "drogon.multnum",
}


def read_keyword(path: Path, keyword: str, size: int) -> np.ndarray:
    """Read one numeric GRDECL keyword, including Eclipse repeat syntax."""
    lines = path.read_text(errors="replace").splitlines()
    start = None
    for i, line in enumerate(lines):
        code = line.split("--", 1)[0].strip()
        if code.upper() == keyword.upper():
            start = i + 1
            break
    if start is None:
        raise KeyError(f"{keyword} not found in {path}")

    values: list[float] = []
    for line in lines[start:]:
        code = line.split("--", 1)[0]
        for token in code.replace("/", " / ").split():
            if token == "/":
                result = np.asarray(values, dtype=float)
                if result.size != size:
                    raise ValueError(f"{keyword} in {path} has {result.size} values; expected {size}")
                return result
            if "*" in token:
                count_text, value_text = token.split("*", 1)
                count = int(count_text) if count_text else 1
                value = float(value_text.replace("D", "E")) if value_text else 0.0
                values.extend([value] * count)
            else:
                values.append(float(token.replace("D", "E")))
    raise ValueError(f"Missing '/' terminator for {keyword} in {path}")


def write_grdecl(path: Path, keyword: str, values: np.ndarray, *, integer=False) -> None:
    data = np.asarray(values).ravel()
    with path.open("w") as stream:
        stream.write(keyword + "\n")
        fmt = "%d" if integer else "%.9g"
        np.savetxt(stream, data, fmt=fmt, delimiter=" ")
        stream.write("/\n")


def write_grid_grdecl(egrid: Path, output: Path) -> None:
    grid = Grid(str(egrid))
    with output.open("w") as stream:
        stream.write("SPECGRID\n")
        stream.write(f"{grid.get_nx()} {grid.get_ny()} {grid.get_nz()} 1 F /\n")
    ecl = ResdataFile(str(egrid))
    with cwrap.open(str(output), "a") as stream:
        for name in ("COORD", "ZCORN", "ACTNUM"):
            if ecl.has_kw(name):
                ecl[name][0].write_grdecl(stream)


def upscaled_mode(matrix: sparse.csr_matrix, values: np.ndarray) -> np.ndarray:
    """Volume-weighted categorical mode for each coarse cell."""
    source = np.asarray(values).ravel()
    output = np.zeros(matrix.shape[0], dtype=source.dtype)
    for row in range(matrix.shape[0]):
        begin, end = matrix.indptr[row], matrix.indptr[row + 1]
        if begin == end:
            continue
        source_ids = matrix.indices[begin:end]
        weights = matrix.data[begin:end]
        categories, inverse = np.unique(source[source_ids], return_inverse=True)
        output[row] = categories[np.argmax(np.bincount(inverse, weights=weights))]
    return output


def logical_volume_overlap(fine_grid, coarse_grid, boundaries) -> sparse.csr_matrix:
    """Fast volume-weighted map for v2's nested logical block boundaries.

    The hybrid-grid builder selects boundaries on fine-cell faces. Each active
    fine cell therefore belongs to exactly one coarse block; its geometric
    volume supplies the conservative weight. ``--exact-intersections`` remains
    available to build tetrahedral geometric intersections instead.
    """
    i_bounds, j_bounds, k_bounds = (np.asarray(axis, dtype=int) for axis in boundaries)
    nx, ny, nz = fine_grid.dims
    cnx, cny, cnz = len(i_bounds) - 1, len(j_bounds) - 1, len(k_bounds) - 1
    k, j, i = np.indices((nz, ny, nx), dtype=np.int64)
    ci = np.searchsorted(i_bounds, i, side="right") - 1
    cj = np.searchsorted(j_bounds, j, side="right") - 1
    ck = np.searchsorted(k_bounds, k, side="right") - 1
    np.clip(ci, 0, cnx - 1, out=ci)
    np.clip(cj, 0, cny - 1, out=cj)
    np.clip(ck, 0, cnz - 1, out=ck)

    fine_indices = np.arange(nx * ny * nz, dtype=np.int64).reshape(nz, ny, nx)
    rows = (ci + cnx * (cj + cny * ck)).ravel()
    columns = fine_indices.ravel()
    weights = np.asarray(fine_grid.volumes, dtype=float).ravel()
    active = np.asarray(fine_grid.active, dtype=bool).ravel()
    coarse_active = np.asarray(coarse_grid.active, dtype=bool).ravel()
    use = active & (weights > 0) & coarse_active[rows]
    return sparse.coo_matrix(
        (weights[use], (rows[use], columns[use])),
        shape=(cnx * cny * cnz, nx * ny * nz),
    ).tocsr()


def cell_mapper(old_grid, new_grid) -> np.ndarray:
    """Map old-level cells to nearest new-level centers in the same resolution."""
    old_centers = old_grid.centers.reshape(-1, 3)
    new_centers = new_grid.centers.reshape(-1, 3)
    finite = np.all(np.isfinite(new_centers), axis=1)
    valid_ids = np.flatnonzero(finite)
    tree = cKDTree(new_centers[valid_ids])
    result = np.arange(len(old_centers), dtype=np.int64)
    old_finite = np.all(np.isfinite(old_centers), axis=1)
    _, nearest = tree.query(old_centers[old_finite])
    result[old_finite] = valid_ids[np.asarray(nearest, dtype=int)]
    return result


def remap_fault_include(source: Path, destination: Path, old_grid, new_grid) -> None:
    mapper = cell_mapper(old_grid, new_grid)
    nx, ny, nz = old_grid.dims
    new_nx, new_ny, new_nz = new_grid.dims
    records = set()
    for line in source.read_text(errors="replace").splitlines():
        code = line.split("--", 1)[0].replace("/", " ").split()
        if len(code) < 8 or code[0].upper() == "FAULTS":
            continue
        name = code[0].strip("'\"")
        if not re.fullmatch(r"F\d+", name, flags=re.IGNORECASE):
            continue
        i1, i2, j1, j2, k1, k2 = (int(value) for value in code[1:7])
        face = code[7].strip("'\"")
        for k in range(k1, k2 + 1):
            for j in range(j1, j2 + 1):
                for i in range(i1, i2 + 1):
                    old_index = (i - 1) + nx * ((j - 1) + ny * (k - 1))
                    mapped = int(mapper[old_index])
                    ni = mapped % new_nx + 1
                    nj = (mapped // new_nx) % new_ny + 1
                    nk = mapped // (new_nx * new_ny) + 1
                    if 1 <= ni <= new_nx and 1 <= nj <= new_ny and 1 <= nk <= new_nz:
                        records.add((name, ni, nj, nk, face))
    with destination.open("w") as stream:
        stream.write("FAULTS\n")
        for name, i, j, k, face in sorted(records):
            stream.write(f"'{name}' {i} {i} {j} {j} {k} {k} '{face}' /\n")
        stream.write("/\n")


def remap_transmissibility(source: Path, destination: Path, old_grid, new_grid) -> None:
    mapper = cell_mapper(old_grid, new_grid)
    old_nx, old_ny, _ = old_grid.dims
    new_nx, new_ny, _ = new_grid.dims
    multiply_records = defaultdict(list)
    nnc_records = defaultdict(list)
    section = None
    for line in source.read_text(errors="replace").splitlines():
        code = line.split("--", 1)[0].replace("/", " ").split()
        if not code:
            continue
        key = code[0].upper()
        if key in {"MULTIPLY", "EDITNNC"}:
            section = key
            continue
        if key in {"END", "ENDEDIT"}:
            section = None
            continue
        if section == "MULTIPLY" and key in {"TRANX", "TRANY", "TRANZ"} and len(code) >= 8:
            value = float(code[1])
            i1, i2, j1, j2, k1, k2 = map(int, code[2:8])
            for k in range(k1, k2 + 1):
                for j in range(j1, j2 + 1):
                    for i in range(i1, i2 + 1):
                        old_index = (i - 1) + old_nx * ((j - 1) + old_ny * (k - 1))
                        mapped = int(mapper[old_index])
                        ni = mapped % new_nx + 1
                        nj = (mapped // new_nx) % new_ny + 1
                        nk = mapped // (new_nx * new_ny) + 1
                        multiply_records[(key, ni, nj, nk)].append(value)
        elif section == "EDITNNC" and len(code) >= 7:
            vals = list(map(int, code[:6]))
            factor = float(code[6])
            mapped_cells = []
            for i, j, k in ((vals[0], vals[1], vals[2]), (vals[3], vals[4], vals[5])):
                old_index = (i - 1) + old_nx * ((j - 1) + old_ny * (k - 1))
                mapped = int(mapper[old_index])
                mapped_cells.extend((mapped % new_nx + 1, (mapped // new_nx) % new_ny + 1,
                                     mapped // (new_nx * new_ny) + 1))
            if tuple(mapped_cells[:3]) != tuple(mapped_cells[3:]):
                nnc_records[tuple(mapped_cells)].append(factor)

    with destination.open("w") as stream:
        stream.write("MULTIPLY\n")
        for (keyword, i, j, k), values in sorted(multiply_records.items()):
            value = float(np.mean(values))
            stream.write(f"{keyword} {value:.12g} {i} {i} {j} {j} {k} {k} /\n")
        stream.write("/\nEDITNNC\n")
        for cells, factors in sorted(nnc_records.items()):
            record = (*cells, float(np.mean(factors)))
            stream.write(" ".join(f"{x:.12g}" if isinstance(x, float) else str(x) for x in record) + " /\n")
        stream.write("/\n")


def build_level(fine_grid, source_egrid: Path, cell_count: int,
                dimensions: tuple[int, int, int], build_root: Path,
                exact_intersections: bool) -> None:
    print(f"Building Level{cell_count} with upscaling-v2, dimensions={dimensions}", flush=True)
    source_dir = SOURCE / f"Level{cell_count}"
    target_dir = OUTPUT / f"Level{cell_count}"
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("FAULT.INC", "drogon_US.trans"):
        shutil.copy2(source_dir / filename, target_dir / filename)
    if cell_count == 104098:
        # The fine grid is the fixed FMU computational grid.
        for filename in ("Grid.grdecl", "probfaction.npz", "US_schdl.sch"):
            shutil.copy2(source_dir / filename, target_dir / filename)
        shutil.copy2(source_egrid, target_dir / "Grid.EGRID")
        fine_size = int(np.prod(fine_grid.dims))
        for keyword, path in REGION_INPUTS.items():
            values = read_keyword(path, keyword, fine_size)
            write_grdecl(target_dir / f"{keyword}.grdecl", keyword, values, integer=True)
        identity = sparse.identity(cell_count, format="csr")
        sparse.save_npz(target_dir / "TransformMatVolume.npz", identity)
        sparse.save_npz(target_dir / "TransformMatMean.npz", identity)
        return

    old_grid = load_grid(source_dir / "Grid.EGRID", relative_tolerance=1e-7)
    old_faults = source_dir / "FAULT.INC"
    old_trans = source_dir / "drogon_US.trans"
    blocks, _ = fault_blocks(fine_grid)
    summary = process_hybrid_dims(build_root, source_egrid, fine_grid, blocks, dimensions, fault_weight=1.0)
    generated_egrid = build_root / f"hybrid_{'x'.join(map(str, dimensions))}" / summary["egrid_file"]
    coarse_grid = load_grid(generated_egrid, relative_tolerance=1e-7)
    if coarse_grid.dims != dimensions:
        raise ValueError(f"Expected dimensions {dimensions}, got {coarse_grid.dims}")

    boundaries = (
        summary["i_boundaries"], summary["j_boundaries"], summary["k_boundaries"]
    )
    if exact_intersections:
        print(f"Level{cell_count}: calculating geometric volume intersections (this may take a while)", flush=True)
        volume = calculate_volume_overlap(fine_grid, coarse_grid).overlap.tocsr()
    else:
        volume = logical_volume_overlap(fine_grid, coarse_grid, boundaries)
    row_sum = np.asarray(volume.sum(axis=1)).ravel()
    inverse = np.zeros_like(row_sum)
    inverse[row_sum > 0] = 1.0 / row_sum[row_sum > 0]
    transform = (sparse.diags(inverse) @ volume).tocsr()

    shutil.copy2(generated_egrid, target_dir / "Grid.EGRID")
    write_grid_grdecl(target_dir / "Grid.EGRID", target_dir / "Grid.grdecl")
    sparse.save_npz(target_dir / "TransformMatVolume.npz", volume)
    sparse.save_npz(target_dir / "TransformMatMean.npz", transform)

    fine_size = int(np.prod(fine_grid.dims))
    for keyword, path in REGION_INPUTS.items():
        fine_values = read_keyword(path, keyword, fine_size)
        coarse_values = upscaled_mode(volume, fine_values)
        write_grdecl(target_dir / f"{keyword}.grdecl", keyword, coarse_values, integer=True)

    source_probabilities = np.load(SOURCE / "Level104098" / "probfaction.npz")
    probability_fields = {
        name: np.asarray(transform @ np.asarray(source_probabilities[name]).ravel(), dtype=np.float32)
        for name in source_probabilities.files
    }
    np.savez_compressed(target_dir / "probfaction.npz", **probability_fields)

    remap_fault_include(old_faults, target_dir / "FAULT.INC", old_grid, coarse_grid)
    remap_transmissibility(old_trans, target_dir / "drogon_US.trans", old_grid, coarse_grid)
    schedule = upscale_schedule_file(
        INCLUDE / "schedule" / "drogon_hist.sch", target_dir / "US_schdl.sch",
        fine_grid, coarse_grid, overlap=volume,
    )
    print(f"Level{cell_count}: {dimensions}, active={np.count_nonzero(coarse_grid.active)}, "
          f"overlap nnz={volume.nnz}, schedule changes={schedule.transformed_records}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, default=ROOT / "_upscaling_build")
    parser.add_argument("--levels", type=int, nargs="*", choices=LEVELS, default=list(LEVELS))
    parser.add_argument(
        "--exact-intersections", action="store_true",
        help="Use v2's geometric tetrahedron intersections (slower) rather than the exact logical-block volume map.",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    args.build_root.mkdir(parents=True, exist_ok=True)
    fine_egrid = args.build_root / "DROGON_FINE.EGRID"
    Grid.load_from_grdecl(str(FINE_GRDECL)).save_EGRID(str(fine_egrid))
    fine_grid = load_grid(fine_egrid, relative_tolerance=1e-7)
    if fine_grid.dims != LEVELS[104098]:
        raise ValueError(f"Unexpected Drogon fine-grid dimensions: {fine_grid.dims}")
    for count in args.levels:
        build_level(fine_grid, fine_egrid, count, LEVELS[count], args.build_root, args.exact_intersections)
    print(f"Built Drogon levels in {OUTPUT}")


if __name__ == "__main__":
    main()
