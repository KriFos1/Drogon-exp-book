#!/bin/bash
# Make a relocatable copy of OPM Flow 2024.10 as built on the NORCE HPC
# (EasyBuild opm-simulators/2024.10-foss-2022a), the simulator of the book runs.
# Run on the HPC login node: bash make_hpc_bundle.sh /path/to/opm-flow-2024.10-hpc
set -uo pipefail
B=${1:?usage: make_hpc_bundle.sh OUTPUT_DIR}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S=/cluster/software
F=$S/opm-simulators/2024.10-foss-2022a/bin/flow
O=$S/OpenMPI/4.1.4-GCC-11.3.0
FB=$S/FlexiBLAS/3.2.0-GCC-11.3.0
# Library path of the module and its dependencies, as Lmod would set it.
LP=$(python3 "$HERE/module_libpath.py" opm-simulators/2024.10-foss-2022a)

rm -rf "$B"; mkdir -p "$B/bin" "$B/lib/flexiblas" "$B/libexec" "$B/python/lib"
cp "$F" "$B/libexec/flow"
cp -a "$O" "$B/openmpi"
cp -L "$FB"/lib/flexiblas/*.so "$B/lib/flexiblas/"
deps() { LD_LIBRARY_PATH=$LP ldd "$1" | awk '/=> \/cluster\/software/ {print $3}'; }
for f in "$F" "$O"/bin/orterun "$O"/lib/*.so "$O"/lib/openmpi/*.so "$FB"/lib/flexiblas/*.so; do
  deps "$f"
done | sort -u | grep -v "^$O/" | while read -r lib; do cp -L "$lib" "$B/lib/"; done
PYLIB=$(LD_LIBRARY_PATH=$LP ldd "$F" | awk '/libpython3.10/ {print $3}')
cp -a "$(dirname "$(readlink -f "$PYLIB")")/python3.10" "$B/python/lib/"

cat > "$B/bin/flow" <<'W'
#!/usr/bin/env bash
# OPM Flow 2024.10 as built on the NORCE HPC (EasyBuild, foss-2022a), relocated.
D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="$D/lib:$D/openmpi/lib:${LD_LIBRARY_PATH:-}"
export OPAL_PREFIX="$D/openmpi" PYTHONHOME="$D/python"
export FLEXIBLAS_LIBRARY_PATH="$D/lib/flexiblas" FLEXIBLAS="$D/lib/flexiblas/libflexiblas_openblas.so"
exec "$D/libexec/flow" "$@"
W
cat > "$B/bin/mpirun" <<'W'
#!/usr/bin/env bash
# OpenMPI 4.1.4 of the HPC build, relocated; puts the bundled flow first on PATH.
D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPAL_PREFIX="$D/openmpi" PATH="$D/bin:$PATH"
export LD_LIBRARY_PATH="$D/lib:$D/openmpi/lib:${LD_LIBRARY_PATH:-}"
exec "$D/openmpi/bin/mpirun" "$@"
W
chmod +x "$B/bin/flow" "$B/bin/mpirun"
"$B/bin/flow" --version
du -sh "$B"
