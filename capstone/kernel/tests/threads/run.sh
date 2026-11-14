#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../tests/capstone-test-env.sh"
kernel=${1:?usage: run.sh APPLIED_LINUX_TREE}
out=${2:-$CAPSTONE_TMP_ROOT/thread-unit}
mkdir -p "$out/include/asm"
cat > "$out/include/asm/unistd.h" <<'HEADER'
#define __ARCH_WANT_NEW_STAT
#define __ARCH_WANT_SYS_CLONE
#include <asm-generic/unistd.h>
HEADER
bind="$kernel/arch/riscv/kernel/sublet"
cc -std=c11 -Wall -Wextra -I"$out/include" -I"$bind" \
    "$here/bind-test.c" "$bind/bind-core.c" "$bind/syscall-pointers.c" -o "$out/bind-test"
"$out/bind-test"
# Control: treat requeue's val2 count as the generic timeout pointer.
python3 - "$bind/bind-core.c" "$out/control.c" <<'PY'
import pathlib, sys
s = pathlib.Path(sys.argv[1]).read_text()
assert '\t\t*pointers = 0x11;' in s
pathlib.Path(sys.argv[2]).write_text(s.replace('\t\t*pointers = 0x11;', '\t\t*pointers = 0x19;'))
PY
cc -std=c11 -Wall -Wextra -I"$out/include" -I"$bind" \
    "$here/bind-test.c" "$out/control.c" "$bind/syscall-pointers.c" -o "$out/bind-control"
if "$out/bind-control" > "$out/control.log" 2>&1; then
    echo 'FAIL: incorrect requeue pointer mask passed' >&2
    exit 1
fi
cat "$out/control.log"
echo 'PASS: negative control rejected'
