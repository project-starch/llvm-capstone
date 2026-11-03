#!/bin/bash
# Profile PHP scripts in one arm with the capprof kit: a freshly booted guest whose QEMU carries
# the capprof plugin, one guest invocation `php -n SCRIPT` per script, in order, each under the
# window helper (profrun) and so each its own capprof window, then
# the kit's attribution (prof-report.py). It is the kit's tools/profile.sh with runcpu replaced by
# direct calls of the kit's own invocation wrapper (tools/capstone-run.sh), so the boot, the
# guest setup, the staging to tmpfs, the window and the attribution are the kit's code, unchanged.
#
# usage: profile-php.sh ARM IMAGE OUT SCRIPT [SCRIPT...]
#   ARM     capstone | plain (the kit's two kinds; capstone = the launcher, exact bounds, mapped-xattr
#           share; plain = a static rv64 program run as the guest's own process, share model none,
#           both as ~/prof-dev/arms-census3.json's cap-dev3 and rv64 arms)
#   IMAGE   OUT/<arm>/php from build-arms.sh; SCRIPT: a PHP script (adapt-scripts.py's output);
#           all below SHARE, and OUT new and below SHARE
# env: SHARE (default: the parent of OUT's parent), KIT (~/prof-dev/kit3), PL (~/prof-dev/platform-dev3),
#      VMHOST (~/prof-dev/host), NM (~/prof-dev/cap3/build/bin/llvm-nm), LIMIT (wall seconds for all
#      the invocations together, default 3600: past it the guest is shut down and the run is a FAIL)
# Writes OUT/stdout-K.txt, OUT/stderr-K.txt (script K's output, copied back after its window),
# OUT/invocations/, OUT/capprof/window-K.tsv, OUT/profile/window-K.json, OUT/profile/summary.txt,
# OUT/profile.json (the sum) and OUT/result.json (judge.py: every invocation exit 0, one complete
# window per script, every test's line and the Total line in each).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd -P)
ARM=${1:?usage} IMAGE=$(readlink -f "${2:?usage}") OUT=${3:?usage}; shift 3
[ $# -ge 1 ] || { echo "profile-php: no script" >&2; exit 2; }
SCRIPTS=(); for s in "$@"; do SCRIPTS+=("$(readlink -f "$s")"); done
[ -e "$OUT" ] && { echo "profile-php: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd -P)
SHARE=$(cd "${SHARE:-$OUT/../..}" && pwd -P)
KIT=${KIT:-$HOME/prof-dev/kit3} PL=${PL:-$HOME/prof-dev/platform-dev3} VMHOST=${VMHOST:-$HOME/prof-dev/host}
NM=${NM:-$HOME/prof-dev/cap3/build/bin/llvm-nm} LIMIT=${LIMIT:-3600}
for p in "$OUT" "$IMAGE" "${SCRIPTS[@]}"; do case "$p" in "$SHARE"/*) ;; *) echo "profile-php: $p is not below $SHARE" >&2; exit 2 ;; esac; done
case $ARM in
  capstone) UP=(--share-model mapped-xattr --exact-bounds); PLAINENV=() ;;
  plain) UP=(--share-model none); PLAINENV=(CAPSTONE_SPEC_PLAIN=1) ;;
  *) echo "profile-php: arm $ARM?" >&2; exit 2 ;;
esac
R=$OUT/record.txt
{ echo "arm $ARM image $IMAGE scripts ${#SCRIPTS[@]} limit $LIMIT"; echo "platform $PL kit $KIT share $SHARE"
  sha256sum "$IMAGE" "${SCRIPTS[@]}"; } > "$R"
python3 - "$PL" >> "$R" <<'PY'
import hashlib, json, os, sys
pl = sys.argv[1]; want = json.load(open(os.path.join(pl, "platform.json"))).get("files", {})
bad = [p for p, h in want.items() if hashlib.sha256(open(os.path.join(pl, p), "rb").read()).hexdigest() != h]
print("platform.json files", len(want), "mismatch", bad)
sys.exit(1 if bad else 0)
PY
sha256sum "$PL/qemu-system-riscv64" "$PL/images/Image" "$PL/images/fw_jump.elf" "$PL/images/rootfs.ext2" \
  "$PL/adapter/capstone-vexec" "$PL/adapter/capstone-job" "$PL/adapter/module/capstone_vm.ko" \
  "$PL/plugins/libcapprof.so" "$PL/profrun" "$PL/kernel/System.map" > "$OUT/inputs.sha256"
export CAPSTONE_QEMU_LOCK=$OUT/qemu.lock
STATES=$(mkdir -p "$HOME/.capstone-spec-vm" && cd "$HOME/.capstone-spec-vm" && pwd -P)
VMSTATE=$STATES/$(printf '%s' "$OUT" | sha256sum | cut -c1-12)
rm -rf "$VMSTATE"; mkdir -m 700 "$VMSTATE"; ln -s "$VMSTATE" "$OUT/vm"
vm() { env PYTHONPATH="$VMHOST" python3 -m capstone_vm --state "$OUT/vm" "$@"; }
mkdir -p "$OUT/capprof"
vm up --profile virtual --memory 8G "${UP[@]}" --share "$SHARE" --cpu-prop x-capstone-test-csrs=true \
  --plugin "$PL/plugins/libcapprof.so" --plugin-arg out="$OUT/capprof" --no-pcrel --icount 0 \
  --qemu "$PL/qemu-system-riscv64" --kernel "$PL/images/Image" --firmware "$PL/images/fw_jump.elf" \
  --rootfs "$PL/images/rootfs.ext2" \
  --launcher "$PL/adapter/capstone-vexec" --job-helper "$PL/adapter/capstone-job" \
  --module "$PL/adapter/module/capstone_vm.ko" --ssh-server "$PL/ssh-server/dropbearmulti" > "$OUT/vm-up.log" 2>&1
trap 'vm down > /dev/null 2>&1 || true' EXIT
python3 -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1]))['command']))" "$VMSTATE/config.json" > "$OUT/qemu-command.json"
# The kit's guest setup (tools/profile.sh): profrun on tmpfs, the module's sections, no syslogd/klogd, no ASLR.
cp "$PL/profrun" "$OUT/profrun"
vm exec -- sh -c "echo 0 > /proc/sys/kernel/randomize_va_space && cp /mnt/host/${OUT#"$SHARE"/}/profrun /tmp/profrun && chmod 0755 /tmp/profrun && cd /sys/module/capstone_vm/sections && for f in .* *; do [ -f \"\$f\" ] && echo \"\$f \$(cat \"\$f\")\"; done; killall syslogd klogd 2>/dev/null; echo randomize_va_space=\$(cat /proc/sys/kernel/randomize_va_space); uname -a; cat /proc/cmdline; for p in /sys/module/capstone_vm/parameters/*; do echo \"param \$(basename \$p)=\$(cat \$p)\"; done" > "$OUT/module-sections.txt"
# capstone-run.sh needs a $SPEC/bin to tell SPEC's own tools apart; there are none here.
mkdir -p "$OUT/nospec/bin"
start=$(date -u +%FT%TZ); t0=$(date +%s); rc=0
for k in "${!SCRIPTS[@]}"; do
  left=$(( LIMIT - ($(date +%s) - t0) )); [ "$left" -gt 0 ] || { rc=124; break; }
  ( cd "$(dirname "${SCRIPTS[$k]}")" && env SPEC="$OUT/nospec" CAPSTONE_SPEC_RECORD="$OUT" CAPSTONE_SPEC_KIT="$KIT" \
      CAPSTONE_VM_STATE="$OUT/vm" CAPSTONE_VM_SHARE="$SHARE" CAPSTONE_VM_HOST="$VMHOST" \
      CAPSTONE_SPEC_PROFRUN=/tmp/profrun CAPSTONE_SPEC_STAGE=tmpfs "${PLAINENV[@]}" \
      timeout "$left" bash "$KIT/tools/capstone-run.sh" "$IMAGE" -n "$(basename "${SCRIPTS[$k]}")" \
      > "$OUT/stdout-$k.txt" 2> "$OUT/stderr-$k.txt" ) || { rc=$?; break; }
done
end=$(date -u +%FT%TZ); wall=$(( $(date +%s) - t0 ))
vm exec -- sh -c "dmesg | tail -n 200; echo ---; dmesg | wc -l" > "$OUT/dmesg.txt" 2>&1 || true
vm down > /dev/null 2>&1 || true
trap - EXIT
python3 -I "$HERE/judge.py" "$OUT" "$rc" "$start" "$end" "$wall" "${SCRIPTS[@]}"
mkdir -p "$OUT/profile"
python3 "$KIT/tools/prof-report.py" --out "$OUT" --platform "$PL" --kind "$ARM" --nm "$NM" \
  > "$OUT/profile/summary.txt" 2> "$OUT/profile/report.err" || { echo "prof-report failed: $(tail -1 "$OUT/profile/report.err")" >&2; exit 1; }
head -14 "$OUT/profile/summary.txt"
grep -q '"verdict": "PASS"' "$OUT/result.json"
