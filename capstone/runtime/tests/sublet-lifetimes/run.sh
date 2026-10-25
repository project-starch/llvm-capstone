#!/usr/bin/env bash
# Compile the public wrappers with the installed compiler and execute its bytes.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../tests/capstone-test-env.sh"
if [[ ${CAPSTONE_QEMU_LOCK_HELD:-0} != 1 ]]; then
    exec env CAPSTONE_QEMU_LOCK_HELD=1 flock -w "${CAPSTONE_QEMU_LOCK_WAIT:-3600}" \
        "$CAPSTONE_QEMU_LOCK" bash "$here/run.sh" "$@"
fi
qroot=$CAPSTONE_REPO_ROOT/capstone/capstone-qemu
qemu=${VIRTUAL_CAPSTONE_QEMU_BINARY:-$qroot/build/qemu-system-riscv64}
work=$(mktemp -d "$CAPSTONE_TMP_ROOT/sublet-wrappers.XXXXXX")
# variant: define, QEMU gate case id (selects the expected cause), allocated
# node count, and which emitted instruction must be the fault PC.
#   good           36  ebreak          11  (8 bootstrap + 3 derives)
#   bad_offset     27  cause 29         8  zero-size CDERIVE in derive
#   linear_parent  11  cause 26         8  zero-size CDERIVE in derive
#   no_manage       7  cause 27        10  the CREVOKE
#   stale_alias     1  cause 25        10  entry.inc's load of the child
variants=(good:-:36:11:none bad_offset:BAD_OFFSET:27:8:derive
          linear_parent:LINEAR_PARENT:11:8:derive no_manage:NO_MANAGE:7:10:revoke
          stale_alias:STALE_ALIAS:1:10:none)
total=0
for opt in 1 2; do
    for spec in "${variants[@]}"; do
        IFS=: read -r variant define case_id count site <<< "$spec"
        extra=()
        [[ $define == - ]] || extra+=(-D"$define"=1)
        obj=$work/$opt-$variant.o
        binary=$work/$opt-$variant.bin
        "$CAPSTONE_CLANG" -target capstone64-unknown-elf -ffreestanding -fno-builtin \
            -fno-jump-tables -mllvm -capstone-gp-free -O"$opt" -Wall -Wextra -Werror \
            -I"$here/../../include" "${extra[@]}" -c "$here/wrappers.c" -o "$obj"
        "$CAPSTONE_LLVM_BIN/llvm-readobj" --relocations "$obj" > "$obj.relocations"
        # A code-only blob is safe to embed only when it has no relocations.
        python3 - "$obj.relocations" <<'PY'
import pathlib, sys
text = pathlib.Path(sys.argv[1]).read_text()
assert text.split('Relocations [', 1)[1].strip() == ']', text
PY
        "$CAPSTONE_LLVM_BIN/llvm-objcopy" --only-section=.text -O binary "$obj" "$binary"
        offset=$(python3 - "$binary" "$site" <<'PY'
import pathlib, struct, sys
code = pathlib.Path(sys.argv[1]).read_bytes()
assert len(code) % 4 == 0
words = struct.unpack('<' + 'I' * (len(code) // 4), code)
# derive's refusal path is CDERIVE rd, parent, x0 into a real destination.
# without_manage's type guard is CDERIVE x0, cap, x0; rd=0 tells them apart.
derive = [4*i for i, w in enumerate(words)
          if w & 0xfff0707f == 0xa200105b and (w >> 7) & 31]
revoke = [4*i for i, w in enumerate(words) if w & 0xfe007fff == 0xa400105b]
site = sys.argv[2]
if site == 'none':
    print(0)
else:
    sites = derive if site == 'derive' else revoke
    assert len(sites) == 1, (site, sites)
    print(sites[0])
PY
)
        for layout in flat paged; do
            flags=("${extra[@]}")
            [[ $layout != paged ]] || flags+=(-DPAGED_NODES=1)
            elf=$work/$opt-$variant-$layout.elf
            clang --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d \
                -nostdlib -fuse-ld=lld "${flags[@]}" -DSUBLET_CASE="$case_id" \
                -DSUBLET_NODE_COUNT="$count" -DWRAPPER_FAULT_OFFSET="$offset" \
                '-DSUBLET_PROGRAM="entry.inc"' "-DWRAPPER_BINARY=\"$binary\"" \
                -I"$here" -Wl,-T,"$qroot/tests/virtual-capstone-runtime/link.ld" \
                "$qroot/tests/sublet-lifetimes/probe.S" -o "$elf"
            timeout 10s "$qemu" -M virt -cpu rv64,x-capstone-u-mode=true \
                -bios none -smp 1 -nographic -semihosting \
                -device "loader,file=$elf" > "$elf.log" 2>&1
            rg -qx 'VIRTUAL-CAPSTONE-M1:PASS' "$elf.log"
            echo "PASS wrapper O$opt $variant $layout"
            total=$((total+1))
        done
    done
done
echo "$total/20 passed; logs: $work"
