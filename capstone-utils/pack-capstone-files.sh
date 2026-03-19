#!/usr/bin/env bash
# pack-capstone-files.sh
# Copies Capstone-related source files into a timestamped folder and archives it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${REPO_ROOT}/capstone-export_${TIMESTAMP}"

# List of files to collect, relative to the repo root
FILES=(
    "llvm/lib/Target/Capstone/CapstoneInstrInfo.td"
    "llvm/lib/Target/Capstone/CapstoneISelDAGToDAG.cpp"
    "llvm/lib/Target/Capstone/CapstoneISelDAGToDAG.h"
    "llvm/lib/Target/Capstone/CapstoneISelLowering.cpp"
    "llvm/lib/Target/Capstone/CapstoneISelLowering.h"
    "llvm/include/llvm/IR/IntrinsicsCapstone.td"
    "clang/include/clang/Basic/BuiltinsCapstone.td"
    "clang/lib/CodeGen/TargetBuiltins/Capstone.cpp"
)

mkdir -p "${OUTPUT_DIR}"
echo "==> Collecting files into: ${OUTPUT_DIR}"

missing=0
for rel_path in "${FILES[@]}"; do
    src="${REPO_ROOT}/${rel_path}"
    if [[ ! -f "${src}" ]]; then
        echo "  [MISSING] ${rel_path}"
        missing=$((missing + 1))
        continue
    fi

    dst="${OUTPUT_DIR}/$(basename "${rel_path}")"
    cp "${src}" "${dst}"
    echo "  [OK]      ${rel_path}"
done

if [[ ${missing} -gt 0 ]]; then
    echo ""
    echo "WARNING: ${missing} file(s) were not found and were skipped."
fi

ARCHIVE="${REPO_ROOT}/capstone-export_${TIMESTAMP}.tar.gz"
echo ""
echo "==> Creating archive: ${ARCHIVE}"
tar -czf "${ARCHIVE}" -C "${REPO_ROOT}" "capstone-export_${TIMESTAMP}"

echo "==> Removing temporary directory: ${OUTPUT_DIR}"
rm -rf "${OUTPUT_DIR}"

echo ""
echo "Done! Archive created at:"
echo "  ${ARCHIVE}"




