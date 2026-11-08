#!/usr/bin/env bash
# Apply once to a disposable, unmodified Linux 6.4.14 tree (the boards' buildroot kernel).
# The 6.4 port of ../install-native-task.sh. Unlike the 6.1 recipe, the tag-aware memory patch
# (../unpinned/linux-6.1-unpinned.patch, ported here) and the mprotect/gcc helper patches are part of
# the series, so one run turns a pristine tree into the native-task kernel. See README.md for the
# hunks that are not mechanical.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
virtual=$(cd -- "$here/.." && pwd)
source "$virtual/../../tests/capstone-test-env.sh"
kernel=${1:?Prepared Linux 6.4.14 source/build tree}
[[ -f "$kernel/Makefile" && -f "$kernel/arch/riscv/kernel/entry.S" ]]
version=$(make -s -C "$kernel" kernelversion)
[[ $version == 6.4.14 ]] || { echo "Linux $version, not 6.4.14: this series is ported to 6.4.14 only." >&2; exit 1; }
[[ ! -e "$kernel/drivers/misc/capstone" ]] || {
    echo 'Use a fresh disposable kernel tree; capstone driver already exists.' >&2
    exit 1
}
stage=$(mktemp -d "${CAPSTONE_TMP_ROOT:-/tmp/capstone}/native-task-source.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT
mapfile -t patches < <(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$here/series")
patches=("${patches[@]/#/$here/}")
python3 - "$kernel" "$stage/preflight" "${patches[@]}" <<'PYTHON'
from pathlib import Path
import shutil, sys
source, target = map(Path, sys.argv[1:3])
target.mkdir()
for name in sys.argv[3:]:
    for line in Path(name).read_text().splitlines():
        if not line.startswith('--- a/'):
            continue
        relative = Path(line[6:])
        original, copy = source / relative, target / relative
        if original.is_file() and not copy.exists():
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, copy)
PYTHON
# Exact application only: a ported series that needs fuzz is not the series that was reviewed.
for change in "${patches[@]}"; do
    patch --dry-run -F0 -d "$stage/preflight" -p1 < "$change"
    patch -F0 -d "$stage/preflight" -p1 < "$change"
done
bash "$virtual/stage-adapter.sh" "$stage/adapter"
for change in "${patches[@]}"; do
    patch -F0 -d "$kernel" -p1 < "$change"
done
install -m 0644 "$stage/adapter/capstone_native_helpers.h" \
    "$kernel/include/linux/capstone_native_helpers.h"
mv "$stage/adapter" "$kernel/drivers/misc/capstone"
printf '\nobj-$(CONFIG_RISCV_CAPSTONE_TASK) += capstone/\n' >> "$kernel/drivers/misc/Makefile"
# The minimal delta (README.md, "Configuration"): SMP, NR_CPUS and RCU stay as the tree's .config has them (one
# POSSIBLE CPU is checked at run time: a one-hart device tree, or nr_cpus=1). Kept from the 6.1 recipe: generic virtual
# CPU accounting with forced user context tracking, which task-driver's "compute charged to user" and "compute
# resumes" checks require (without them the gate fails exactly those two). NATIVE_TASK_6_1_CONFIG=1 also turns SMP off,
# as the 6.1 recipe did.
[[ ${NATIVE_TASK_6_1_CONFIG:-0} == 1 ]] && "$kernel/scripts/config" --file "$kernel/.config" --disable SMP
"$kernel/scripts/config" --file "$kernel/.config" \
    --disable VIRT_CPU_ACCOUNTING_NATIVE --disable TICK_CPU_ACCOUNTING \
    --enable VIRT_CPU_ACCOUNTING_GEN --enable CONTEXT_TRACKING_USER_FORCE \
    --enable RISCV_CAPSTONE_TASK
echo 'Sources installed. Run make ARCH=riscv olddefconfig, then check that CONFIG_RISCV_CAPSTONE_TASK=y survived it.'
