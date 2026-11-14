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
"$CAPSTONE_CLANG" "${flags[@]}" -c "$here/tests/trap/jump-register.S" -o "$out/jump-register.o"
for test in smoke malloc-smoke threads-smoke threads-disabled futex-stale setjmp-smoke signal-smoke stale-free; do
    source_test=$test
    test_flags=()
    extra=()
    if [[ $test == threads-disabled ]]; then
        source_test=threads-smoke
        test_flags+=(-DTHREADS_DISABLED_CONTROL)
    fi
    if [[ $test == setjmp-smoke ]]; then extra+=("$out/jump-register.o"); fi
    "$CAPSTONE_CLANG" "${flags[@]}" "${test_flags[@]}" -c "$here/tests/trap/$source_test.c" -o "$out/$test.o"
    "$CAPSTONE_LD_LLD" --gc-sections -T "$rt/virtual/link.ld" -o "$out/$test.dom" \
        "$out/$test.o" "${extra[@]}" --whole-archive "$out/libtrap-runtime.a" --no-whole-archive \
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
# The CDERIVE failure cases link against a libc built with
# MUSL_CAPSTONE_EXTRA_CFLAGS=-DCAPSTONE_HEAP_DERIVE_INJECTION=1.
if [[ -n ${TRAP_LIBC_INJECTION_ARCHIVE:-} ]]; then
    for case in 1 2 3 4 5 6; do
        "$CAPSTONE_CLANG" "${flags[@]}" -DCASE=$case -c "$here/tests/trap/malloc-fail.c" -o "$out/mf-$case.o"
        "$CAPSTONE_LD_LLD" --gc-sections -T "$rt/virtual/link.ld" -o "$out/mf-$case.dom" \
            "$out/mf-$case.o" --whole-archive "$out/libtrap-runtime.a" --no-whole-archive \
            "$TRAP_LIBC_INJECTION_ARCHIVE"
    done
fi
printf 'Built %s\n' "$out/"*.dom

if [[ -n ${TRAP_CONTROL_SETJMP:-} ]]; then
    "$CAPSTONE_CLANG" "${flags[@]}" -c "$TRAP_CONTROL_SETJMP" -o "$out/control-setjmp.o"
    control_objects=()
    for obj in "${objects[@]}"; do
        [[ $obj == "$out/obj/setjmp.o" ]] || control_objects+=("$obj")
    done
    if "$CAPSTONE_LD_LLD" --gc-sections -T "$rt/virtual/link.ld" -o "$out/control-setjmp.dom" \
        "$out/setjmp-smoke.o" "$out/jump-register.o" "$out/control-setjmp.o" \
        "${control_objects[@]}" "$TRAP_LIBC_ARCHIVE" > "$out/control-setjmp.log" 2>&1; then
        echo 'FAIL: baseline setjmp unexpectedly linked' >&2
        exit 1
    fi
    grep -q 'undefined symbol: __capstone_sigsetjmp_save' "$out/control-setjmp.log"
    grep -q 'undefined symbol: __capstone_longjmp_reap' "$out/control-setjmp.log"
    echo 'PASS: baseline setjmp fails to link with both missing launcher hooks'
fi
