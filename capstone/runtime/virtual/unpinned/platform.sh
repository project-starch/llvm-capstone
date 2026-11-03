#!/usr/bin/env bash
# Make a fresh, real-file platform; never update the input platform in place.
set -euo pipefail
base=${1:?platform-dev4 directory}
qemu=${2:?raw-moves QEMU binary}
kernel=${3:?prepared patched kernel directory}
adapter=${4:?build-adapter.sh output directory}
out=${5:?new platform directory}
[[ ! -e $out ]] || { echo "refusing existing platform: $out" >&2; exit 2; }
parent=$(dirname "$out")
mkdir -p "$parent"
df -h "$parent"
avail=$(df -B1 --output=avail "$parent" | tail -1)
(( avail >= 12*1024*1024*1024 )) || { echo 'UNRESOLVED: less than 12 GiB free' >&2; exit 2; }
cp -rL --sparse=always "$base" "$out"
install -m 755 "$qemu" "$out/qemu-system-riscv64"
install -m 644 "$kernel/arch/riscv/boot/Image" "$out/images/Image"
install -m 644 "$kernel/System.map" "$out/kernel/System.map"
cp -rL "$adapter"/. "$out/adapter/"
python3 - "$out" <<'PY'
from pathlib import Path
import hashlib
import json
import sys
root = Path(sys.argv[1])
assert not any(p.is_symlink() for p in root.rglob('*')), 'platform contains symlinks'
record = json.loads((root / 'platform.json').read_text())
record['files'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in root.rglob('*') if p.is_file() and p.name != 'platform.json'}
record['unpinned'] = {'cpu_props': ['x-capstone-raw-moves=true', 'x-capstone-test-csrs=false'],
                      'module_params': 'unpinned=1', 'kernel_contract': 'capstone,unpinned-v1'}
record['provenance']['unpinned'] = 'Local raw-transport model, patched Linux 6.1 and matching adapter; file hashes above are authoritative.'
(root / 'platform.json').write_text(json.dumps(record, indent=2) + '\n')
print('PLATFORM_READY; enable the recorded CPU properties and module parameter in the runner')
PY
