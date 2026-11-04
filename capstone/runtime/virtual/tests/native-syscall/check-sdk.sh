#!/bin/bash
# Cross-compile only on the build host using an existing matching compiler.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../../../.." && pwd)
source "$repo/capstone/tests/capstone-test-env.sh"
out=${1:?fresh output directory}
cc=${2:?existing Capstone clang}
objdump=${3:?matching llvm-objdump}
nm=${4:?matching llvm-nm}
[[ ! -e $out ]] || { echo 'output already exists' >&2; exit 2; }
free_kb=$(df -Pk "$(dirname "$out")" | awk 'NR==2 {print $4}')
((free_kb >= 12582912)) || { echo 'less than 12 GiB free' >&2; exit 2; }
mkdir -p "$out/build"
arch=$repo/capstone/ports/musl-capstone/arch-capstone64
asm=$repo/capstone/ports/musl-capstone/runtime/native-syscall.S
flags=(-target capstone64-unknown-elf -O2 -ffreestanding -DCAPSTONE_RUNTIME_VIRTUAL
       -I"$repo/capstone/runtime/include" -I"$arch")
"$cc" "${flags[@]}" -c "$here/sdk-probe.c" -o "$out/build/default.o"
"$cc" "${flags[@]}" -DCAPSTONE_NATIVE_SYSCALL_ABI -c "$here/sdk-probe.c" -o "$out/build/native.o"
"$cc" "${flags[@]}" -DCAPSTONE_NATIVE_SYSCALL_ABI -c "$asm" -o "$out/build/stub.o"
"$nm" -u "$out/build/default.o" > "$out/default-symbols.txt"
"$nm" -u "$out/build/native.o" > "$out/native-symbols.txt"
"$objdump" -d "$out/build/stub.o" > "$out/stub-disassembly.txt"
grep -q ' U __capstone_hostcall$' "$out/default-symbols.txt"
! grep -q '__capstone_native_syscall' "$out/default-symbols.txt"
grep -q ' U __capstone_native_syscall$' "$out/native-symbols.txt"
! grep -q '__capstone_hostcall' "$out/native-symbols.txt"
[[ $(grep -c 'movc' "$out/stub-disassembly.txt") == 6 ]]
grep -q 'ecall' "$out/stub-disassembly.txt"
rc=0
"$cc" "${flags[@]}" -c "$asm" -o "$out/build/refused.o" > "$out/default-stub-refused.log" 2>&1 || rc=$?
[[ $rc != 0 ]] && grep -q 'require explicit virtual native ABI selection' "$out/default-stub-refused.log"
sha256sum "$cc" "$asm" "$arch/syscall_arch.h" "$here/sdk-probe.c" > "$out/inputs.sha256"
echo 'PASS SDK opt-in: default delegates, native calls stub, six tagged moves, unselected stub refuses build'
