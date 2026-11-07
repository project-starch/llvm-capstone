#!/usr/bin/env bash
# Apply once to a disposable Linux 6.1 tree with the tag-aware memory patch.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../tests/capstone-test-env.sh"
kernel=${1:?Prepared Linux source/build tree}
[[ -f "$kernel/Makefile" && -f "$kernel/arch/riscv/kernel/entry.S" ]]
[[ ! -e "$kernel/drivers/misc/capstone" ]] || {
    echo 'Use a fresh disposable kernel tree; capstone driver already exists.' >&2
    exit 1
}
# The spawn patch builds on the task patch. Preflight the complete sequence on
# disposable copies, rather than testing dependent patches against old files.
stage=$(mktemp -d "${CAPSTONE_TMP_ROOT:-/tmp/capstone}/native-task-source.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT
patches=("$here/kernel-native-task.patch" "$here/kernel-native-files.patch"
         "$here/kernel-native-process.patch" "$here/kernel-native-spawn.patch"
         "$here/kernel-native-directory.patch" "$here/kernel-native-time.patch")
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
for change in "${patches[@]}"; do
    patch --dry-run -d "$stage/preflight" -p1 < "$change"
    patch -d "$stage/preflight" -p1 < "$change"
done
# Stage before mutating Linux so missing frontend inputs also fail atomically.
bash "$here/stage-adapter.sh" "$stage/adapter"
for change in "${patches[@]}"; do
    patch -d "$kernel" -p1 < "$change"
done
mv "$stage/adapter" "$kernel/drivers/misc/capstone"
printf '\nobj-$(CONFIG_RISCV_CAPSTONE_TASK) += capstone/\n' >> "$kernel/drivers/misc/Makefile"
"$kernel/scripts/config" --file "$kernel/.config" --disable SMP \
    --disable VIRT_CPU_ACCOUNTING_NATIVE --disable TICK_CPU_ACCOUNTING \
    --enable VIRT_CPU_ACCOUNTING_GEN --enable CONTEXT_TRACKING_USER_FORCE \
    --enable RISCV_CAPSTONE_TASK
echo 'Sources installed. Run make ARCH=riscv olddefconfig, then build Image.'
