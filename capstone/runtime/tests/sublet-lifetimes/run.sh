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
total=0
for opt in 1 2; do
    for bad in 0 1; do
        extra=(); case_id=36; count=11
        if [[ $bad == 1 ]]; then extra+=(-DBAD_OFFSET=1); case_id=27; count=8; fi
        obj=$work/$opt-$bad.o
        binary=$work/$opt-$bad.bin
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
        offset=$(python3 - "$binary" "$bad" <<'PY'
import pathlib, struct, sys
code = pathlib.Path(sys.argv[1]).read_bytes()
assert len(code) % 4 == 0
words = struct.unpack('<' + 'I' * (len(code) // 4), code)
# The out-of-range-offset path invokes CDERIVE with size=x0 to fault atomically.
sites = [4*i for i, w in enumerate(words) if w & 0xfff0707f == 0xa200105b]
assert sites, 'no zero-size CDERIVE in emitted code'
if sys.argv[2] == '1':
    assert len(sites) == 1, sites
print(sites[0])
PY
)
        for layout in flat paged; do
            flags=("${extra[@]}")
            [[ $layout != paged ]] || flags+=(-DPAGED_NODES=1)
            elf=$work/$opt-$bad-$layout.elf
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
            echo "PASS wrapper O$opt bad_offset=$bad $layout"
            total=$((total+1))
        done
    done
done
echo "$total/8 passed; logs: $work"
