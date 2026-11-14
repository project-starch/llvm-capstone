#!/usr/bin/env bash
# Build the static trap-route smoke image and retain its disassembly.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../tests/capstone-test-env.sh"
out=${1:-$CAPSTONE_TMP_ROOT/trap-smoke}
mkdir -p "$out/obj"
if [[ -z ${TRAP_LIBC_ARCHIVE:-} ]]; then
    OUT_DIR="$out/libc" bash "$here/build-musl-capstone.sh"
    TRAP_LIBC_ARCHIVE="$out/libc/libc-capstone.a"
fi
musl=$(bash "$here/prepare-musl-capstone.sh" | tail -1)
flags=(-target capstone64-unknown-elf -O1 -ffreestanding -fno-builtin
    -ffunction-sections -fdata-sections -nostdinc -Wno-int-conversion
    -Wno-int-to-void-pointer-cast -Wno-void-pointer-to-int-cast
    -Wno-capstone-pointer-roundtrip -D_XOPEN_SOURCE=700
    -DCAPSTONE_RUNTIME_TRAP=1 -DCAPSTONE_RUNTIME_VIRTUAL=1 -DCAPSTONE_MUSL_MALLOC=1
    -mllvm -capstone-gp-free -mllvm -capstone-image-gp
    -Xclang -target-abi -Xclang lp64)
for feature in +m +a +f +d +movc-keeps-integer-source; do
    flags+=(-Xclang -target-feature -Xclang "$feature")
done
for inc in arch/capstone64 arch/generic obj/src/internal src/include src/internal obj/include include src/multibyte; do
    flags+=(-I"$musl/$inc")
done
flags+=(-I"$CAPSTONE_REPO_ROOT/capstone/runtime/include")
rt="$CAPSTONE_REPO_ROOT/capstone/runtime"
objects=()
for src in "$here/runtime/"{linux-start.c,tls.c,delegate.c,native-syscall.S,set_thread_area.S,setjmp.S,atomic_libcalls.c,lock.c,string_bounds_safe.c,atexit_capability_safe.c} \
    "$rt/virtual/"{start.S,mapping.c,malloc-mapping.c,thread.c,pthread.c} "$rt/domain/gct-section-end.S"; do
    obj="$out/obj/$(basename "${src%.*}").o"
    "$CAPSTONE_CLANG" "${flags[@]}" -c "$src" -o "$obj"
    objects+=("$obj")
done
"$CAPSTONE_LLVM_BIN/llvm-ar" rcs "$out/libtrap-runtime.a" "${objects[@]}"
for test in smoke malloc-smoke; do
    "$CAPSTONE_CLANG" "${flags[@]}" -c "$here/tests/trap/$test.c" -o "$out/$test.o"
    "$CAPSTONE_LD_LLD" --gc-sections -T "$rt/virtual/link.ld" -o "$out/$test.dom" \
        "$out/$test.o" --whole-archive "$out/libtrap-runtime.a" --no-whole-archive \
        "$TRAP_LIBC_ARCHIVE"
    "$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$out/$test.dom" > "$out/$test.dis"
    "$CAPSTONE_LLVM_BIN/llvm-readelf" -SW "$out/$test.dom" > "$out/$test.sections"
    if grep -q '\.capstone_virtual' "$out/$test.sections"; then
        echo 'FAIL: launcher image handshake survived' >&2
        exit 1
    fi
    "$CAPSTONE_LLVM_BIN/llvm-nm" -u "$out/$test.dom" > "$out/$test.undefined"
    test ! -s "$out/$test.undefined"
done
"$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$out/obj/native-syscall.o" > "$out/syscall.dis"
"$CAPSTONE_CLANG" "${flags[@]}" -c "$here/tests/trap/syscall-shape.c" -o "$out/syscall-shape.o"
"$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$out/syscall-shape.o" > "$out/syscall-shape.dis"
python3 "$here/tests/trap/check-disassembly.py" "$out"
# Supply the baseline start.S and its vm-abi.h together to build the control.
if [[ -n ${TRAP_CONTROL_START:-} ]]; then
    "$CAPSTONE_CLANG" "${flags[@]}" -c "$TRAP_CONTROL_START" -o "$out/control-start.o"
    "$CAPSTONE_LLVM_BIN/llvm-readelf" -x .capstone_virtual "$out/control-start.o" > "$out/control-handshake.txt"
    grep -q '43504f4e 56564d37' "$out/control-handshake.txt"
    echo 'PASS: baseline startup contains the launcher image handshake'
fi
printf 'Built %s and %s. Execution requires the trap-route kernel loader.\n' "$out/smoke.dom" "$out/malloc-smoke.dom"
