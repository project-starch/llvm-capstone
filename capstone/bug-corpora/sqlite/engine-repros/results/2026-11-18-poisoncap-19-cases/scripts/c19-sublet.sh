#!/usr/bin/env bash
# Cases 02, 11 and 13 on Sublet's block port, built from the corpus sources of dev fe1d2dad2ebe,
# in the configuration of v6-sublet-block of 2026-10-26: 256 MiB heap from malloc, minimum
# request 16, lookaside on, the platform of sublet-platform.sha256. Twice. The build carries the
# two standing controls and the four realloc and capacity controls, as block-port.sh did.
# sdk-v6-restored is sdk-v6 with its libc path repointed: the compiler and the libc archive that
# sdk-v6 names were deleted from this host after 2026-10-26. The compiler is rebuilt from the
# recorded source archive (same sha256 as toolchain-record.txt), and with this libc archive the
# unchanged case 03 links to the byte-identical image of builds/v6-sublet-block (README).
set -uo pipefail
P=$HOME/poisoncap-platform
K=$P/repo-19cases
CTL=$K/capstone/bug-corpora/sqlite/engine-repros/controls
cd "$P"
export CAPSTONE_QEMU_CPU_EXTRA="svadu=false,x-capstone-supervisor-slots=32"
export CAPSTONE_QEMU_LOCK=$HOME/.capstone-locks/qemu.lock
mkdir -p "$(dirname "$CAPSTONE_QEMU_LOCK")"
CFG=(--heap-bytes 268435456 --heap-min 16 --heap-malloc)
SEL=02,11,13
OBS=()
for c in control_uaf_mem5 control_bounds_mem5 control_realloc_same_mem5 control_realloc_shrink_mem5 \
         control_realloc_fail_mem5 control_capacity_mem5; do OBS+=(--observe "$CTL/$c.c"); done
rm -f logs/c19-sublet.done

rm -rf builds/c19-sublet-block
python3 "$K/capstone/ports/sqlite/repro322/build-virtual.py" --sdk "$P/sdk-v6-restored" \
  --amalgamation "$HOME/pg-virtual-kit/inputs/sqlite-amalg" --ext-src "$HOME/pg-virtual-kit/inputs/sqlite-src-3220000" \
  --sublet --block-port "${CFG[@]}" --only $SEL "${OBS[@]}" --out builds/c19-sublet-block > logs/build-c19-sublet-block.out 2>&1
echo "rc=$?" >> logs/build-c19-sublet-block.out
sl() {  # $1 run name
  rm -rf "runs/$1"
  CAPSTONE_QEMU_DOMAIN_LOCK=$HOME/.capstone-locks/block-a.lock \
  python3 "$K/capstone/ports/sqlite/repro322/run-virtual-staged.py" --build builds/c19-sublet-block \
    --platform platform-v6-heap256 --work "runs/$1" --llvm-bin "$HOME/spec-kit/toolchain/build/bin" --cases $SEL \
    > "logs/run-$1.out" 2>&1; echo "rc=$?" >> "logs/run-$1.out"
}
sl c19-sublet-block
sl c19-sublet-block-rep2
echo done > logs/c19-sublet.done
