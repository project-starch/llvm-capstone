#!/bin/bash
# Run and validate one benchmark at one size with runcpu, in a freshly booted guest.
# usage: run.sh KIT SPEC_ROOT TOOLCHAIN PLATFORM BUILD_RECORD SIZE BENCH OUT [LABEL]
#   LABEL         the arm (manifest `arms`, default capstone-virtual); its SDK comes from BUILD_RECORD/arms.json
#   TOOLCHAIN     recipes/toolchain.sh's directory (sdk/, src/llvm-capstone/capstone/runtime/host)
#   PLATFORM      qemu-system-riscv64, images/{Image,fw_jump.elf,rootfs.ext2},
#                 adapter/{capstone-vexec,capstone-job,module/capstone_vm.ko}, ssh-server/dropbearmulti; checked against
#                 manifest.lock.json
#   BUILD_RECORD  tools/build.sh's record (ldecod-host, arms.json)
#   OUT           new directory: result.json, runcpu.log, SPEC's report and log, the run
#                 directory's command files and output hashes, invocations/, vm/, qemu-usage.json
# The guest is booted for this run only and discarded afterwards (capstone_vm, disk snapshot).
set -euo pipefail
KIT=$(cd "$1" && pwd) SPEC_ROOT=$(cd "$2" && pwd) TC=$(cd "$3" && pwd) PL=$(cd "$4" && pwd)
BREC=$(cd "$5" && pwd) SIZE=$6 BENCH=$7 OUT=$8 LABEL=${9:-capstone-virtual}
[ -e "$OUT" ] && { echo "run: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
M=$KIT/manifest.lock.json
SDK=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]]['sdk'])" "$BREC/arms.json" "$LABEL" 2>/dev/null) \
  || { echo "run: $BREC/arms.json has no arm $LABEL" >&2; exit 2; }
echo "$LABEL" > "$OUT/label.txt"
VMHOST=$TC/src/llvm-capstone/capstone/runtime/host
python3 "$KIT/tools/check-platform.py" "$M" "$PL" > "$OUT/platform-check.txt"
export CAPSTONE_QEMU_LOCK=$OUT/qemu.lock
# The guest's state directory must be short: capstone_vm puts an SSH socket in it, and a Unix socket
# path is limited to 107 bytes (a run under rep1/capstone-spatial-scrub/538.imagick_r/ overflowed).
# OUT/vm is a symlink to a short directory named by a hash of OUT, which capstone_vm resolves.
STATES=$(mkdir -p "${CAPSTONE_SPEC_VM_STATES:-$HOME/.capstone-spec-vm}" && cd "${CAPSTONE_SPEC_VM_STATES:-$HOME/.capstone-spec-vm}" && pwd -P)
VMSTATE=$STATES/$(printf '%s' "$OUT" | sha256sum | cut -c1-12)
mkdir -m 700 "$VMSTATE"; ln -s "$VMSTATE" "$OUT/vm"   # capstone_vm insists on 0700; the path is canonical (QEMU's argv carries it)
vm() { env PYTHONPATH="$VMHOST" python3 -m capstone_vm --state "$OUT/vm" "$@"; }
mem=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['platform']['vm']['memory'])" "$M")
share=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['platform']['vm']['share_model'])" "$M")
vm up --profile virtual --exact-bounds --memory "$mem" --share-model "$share" \
  --qemu "$PL/qemu-system-riscv64" --kernel "$PL/images/Image" --firmware "$PL/images/fw_jump.elf" \
  --rootfs "$PL/images/rootfs.ext2" --share "$SPEC_ROOT" \
  --launcher "$PL/adapter/capstone-vexec" --job-helper "$PL/adapter/capstone-job" \
  --module "$PL/adapter/module/capstone_vm.ko" --ssh-server "$PL/ssh-server/dropbearmulti" > "$OUT/vm-up.log" 2>&1
trap 'vm down > /dev/null 2>&1 || true' EXIT
# The guest's QEMU process: its CPU time and peak memory are read before the guest goes down.
qemu_pid=$(python3 - "$PL/qemu-system-riscv64" "$VMSTATE" <<'PY'
import os, sys
qemu, state = sys.argv[1], sys.argv[2]
for pid in os.listdir('/proc'):
    if not pid.isdigit(): continue
    try: argv = open(f'/proc/{pid}/cmdline', 'rb').read().split(b'\0')
    except OSError: continue
    words = [a.decode(errors='replace') for a in argv]
    if words and os.path.basename(words[0]) == os.path.basename(qemu) and any(state in w for w in words):
        print(pid); break
PY
)
usage() { python3 - "${qemu_pid:-0}" "$1" "$OUT/qemu-usage.json" <<'PY'
import json, os, sys, time
pid, when, out = sys.argv[1], sys.argv[2], sys.argv[3]
rec = json.load(open(out)) if os.path.exists(out) else {"pid": int(pid), "clk_tck": os.sysconf('SC_CLK_TCK')}
try:
    f = open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()
    rec[when] = {"utime_ticks": int(f[11]), "stime_ticks": int(f[12]), "time": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    for line in open(f'/proc/{pid}/status'):
        if line.startswith(('VmHWM', 'VmRSS')): rec[when][line.split(':')[0]] = line.split()[1] + ' kB'
except OSError as e:
    rec[when] = {"error": str(e)}
json.dump(rec, open(out, 'w'), indent=1)
PY
}
usage before
cd "$SPEC_ROOT"
set +u; source ./shrc > /dev/null; set -u
start=$(date -u +%FT%TZ)
rc=0
runcpu --config=capstone-virtual --action=validate --nobuild --size="$SIZE" --tune=base \
  --define sdk="$SDK" --define kit="$KIT" --define arm_label="$LABEL" --define vm_state="$OUT/vm" --define vm_share="$SPEC_ROOT" \
  --define vm_host="$VMHOST" --define ldecod_host="$BREC/ldecod-host" --define record="$OUT" \
  "$BENCH" > "$OUT/runcpu.log" 2>&1 || rc=$?
end=$(date -u +%FT%TZ)
usage after
vm down > /dev/null 2>&1 || true
trap - EXIT
# SPEC's own records of this run: the log and reports runcpu names, and the run directory's
# command files, miscompare files and output hashes (outputs can be large; they stay in place).
mkdir -p "$OUT/spec"
for f in $(grep -o "result/CPU2017\.[0-9]*\.[A-Za-z0-9._-]*" "$OUT/runcpu.log" | sort -u); do
  [ -f "$f" ] && cp "$f" "$OUT/spec/"
done
# The run directory: the working directory of the benchmark's guest invocations (runcpu's log
# does not name it).
rundir=$(python3 - "$OUT" <<'PY'
import glob, json, os, sys
for f in sorted(glob.glob(os.path.join(sys.argv[1], "invocations", "[0-9][0-9][0-9][0-9].json"))):
    d = json.load(open(f))
    if d.get("route") == "guest": print(d["cwd"]); break
PY
)
if [ -n "$rundir" ] && [ -d "$rundir" ]; then
  echo "$rundir" > "$OUT/rundir.txt"
  # SPEC's command files and the .err files, which carry the launcher's CAPSTONE_VM_STATS lines.
  for f in "$rundir"/*.cmd "$rundir"/*.mis "$rundir"/*.out "$rundir"/*.err; do
    [ -f "$f" ] && [ "$(stat -c %s "$f")" -lt 16777216 ] && cp "$f" "$OUT/spec/"
  done
  ( cd "$rundir" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum ) > "$OUT/rundir.sha256"
fi
python3 "$KIT/tools/verdict.py" "$OUT" "$BENCH" "$SIZE" "$rc" "$start" "$end" "$LABEL" > "$OUT/result.json"
cat "$OUT/result.json"
