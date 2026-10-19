#!/usr/bin/env bash
#
# Clone, patch and build TGIF-DB (Roy lab) for this machine.
#
# TGIF ships a Makefile written for Linux/gcc and a bundled GSL that is not
# present in the repo (gsl/lib is empty). Two things have to change before it
# builds and runs on macOS/arm64 with Apple clang; both are applied here
# idempotently, so re-running this script is safe.
#
#   1. modules/random_svd/matrix_vector_functions_gsl.c calls gsl_linalg_QR_*
#      without including <gsl/gsl_linalg.h>. That compiled under C89's implicit
#      declarations; clang 17 defaults to C23, where it is an error. Fixed with
#      a -include flag, not a source edit.
#
#   2. TGIF::clean() does `delete tree[i]` on a vector<Node*> where Node is
#      ABSTRACT and its destructor is NON-VIRTUAL. That is undefined behaviour,
#      and clang emits a trap instruction for it, so every run died with
#      SIGTRAP inside clean() -- on the FIRST window, before any output file was
#      written. `virtual ~Node()` is the one-word fix, applied by sed below.
#      (It also makes ~Leaf() run, which frees the residual matrix R.)
#
#   3. io::write_sigDB() does `for (i = 0; i < sigDbs.size()-1; i++)` and then
#      `sigDbs.back()`. When a pair of tasks has NO significantly differential
#      boundary -- which is the common case on thinned matrices, where nothing
#      is called at all -- sigDbs is empty, size()-1 wraps around as size_t, and
#      it walks off the end. That is a SIGSEGV at every noisy rung. Guarded with
#      an early return, so the pair gets a header-only file.
#
# GSL comes from the snakemake-tutorial conda env: it is the only env here with
# GSL, and it is 2.7, which is what TGIF's README requires.
#
# Usage:  ./build_tgif.sh          (from simulation/tgif_benchmark/)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/tgif"
GSL_PREFIX="${GSL_PREFIX:-/Users/PSLabMonisha/miniforge3/envs/snakemake-tutorial}"

if [ ! -d "$SRC/.git" ]; then
    echo "==> cloning TGIF"
    git clone https://github.com/Roy-lab/tgif.git "$SRC"
fi

if [ ! -f "$GSL_PREFIX/include/gsl/gsl_version.h" ]; then
    echo "ERROR: no GSL headers under $GSL_PREFIX" >&2
    echo "       set GSL_PREFIX to a prefix holding GSL 2.7 (conda: conda install -c conda-forge gsl)" >&2
    exit 1
fi
echo "==> GSL: $(sed -n 's/.*GSL_VERSION "\(.*\)".*/\1/p' "$GSL_PREFIX/include/gsl/gsl_version.h") at $GSL_PREFIX"

# --- patch 2: virtual destructor -------------------------------------------
if grep -q '^		~Node() {};' "$SRC/modules/node.h"; then
    echo "==> patching modules/node.h: ~Node -> virtual ~Node"
    sed -i '' 's/^		~Node() {};/		virtual ~Node() {};/' "$SRC/modules/node.h"
else
    echo "==> modules/node.h already patched"
fi
grep -q 'virtual ~Node' "$SRC/modules/node.h" || { echo "ERROR: node.h patch failed" >&2; exit 1; }

# --- patch 3: empty sigDbs in write_sigDB -----------------------------------
if ! grep -q 'PATCHED: see build_tgif.sh' "$SRC/modules/io.cpp"; then
    echo "==> patching modules/io.cpp: guard empty sigDbs in write_sigDB"
    python3 - "$SRC/modules/io.cpp" <<'PY'
import sys
path = sys.argv[1]
src = open(path).read()
anchor = "\tint shifted = 0;\n\tfor (int i = 0; i < (sigDbs.size()-1); i++) {"
if src.count(anchor) != 1:
    sys.exit("ERROR: write_sigDB anchor not found exactly once; patch by hand")
guard = "\tif (sigDbs.empty()) { ofs.close(); return 0; } // PATCHED: see build_tgif.sh\n"
open(path, "w").write(src.replace(anchor, guard + anchor))
PY
else
    echo "==> modules/io.cpp already patched"
fi

# --- compile ----------------------------------------------------------------
cd "$SRC"
echo "==> compiling RSVDPACK objects"
gcc -O2 -include gsl/gsl_linalg.h -c -o modules/random_svd/matrix_funcs.o \
    modules/random_svd/matrix_vector_functions_gsl.c -I"$GSL_PREFIX/include" 2>/dev/null
gcc -O2 -include gsl/gsl_linalg.h -c -o modules/random_svd/rsvd.o \
    modules/random_svd/low_rank_svd_algorithms_gsl.c -I"$GSL_PREFIX/include" 2>/dev/null

echo "==> compiling tgif-db"
g++ -O2 tgif-db.cpp modules/*.cpp modules/random_svd/*.o -o tgif-db \
    -L"$GSL_PREFIX/lib" -lgsl -lgslcblas -I"$GSL_PREFIX/include"

echo "==> compiling tgif-dc"
g++ -O2 tgif-dc.cpp modules/*.cpp modules/random_svd/*.o -o tgif-dc \
    -L"$GSL_PREFIX/lib" -lgsl -lgslcblas -I"$GSL_PREFIX/include"

echo
echo "built: $SRC/tgif-db"
echo "run it with DYLD_LIBRARY_PATH=$GSL_PREFIX/lib (run_tgif_bench.py sets this itself)."
echo
echo "NOTE: TGIF's getopt does not permute argv on BSD/macOS, so OPTIONS MUST COME"
echo "      BEFORE the two positional arguments, unlike the order in TGIF's README."
