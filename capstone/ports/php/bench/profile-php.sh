#!/bin/bash
# Profile one PHP script in one arm with the capprof kit: a freshly booted guest whose QEMU carries
# the capprof plugin, one guest invocation `php -n SCRIPT` under the window helper (profrun), then
# the kit's attribution (prof-report.py). It is the kit's tools/profile.sh with runcpu replaced by
# one direct call of the kit's own invocation wrapper (tools/capstone-run.sh), so the boot, the
# guest setup, the staging to tmpfs, the window and the attribution are the kit's code, unchanged.
#
# usage: profile-php.sh ARM IMAGE SCRIPT OUT
#   ARM     capstone | plain (the kit's two kinds; capstone = the launcher, exact bounds, mapped-xattr
#           share; plain = a static rv64 program run as the guest's own process, share model none,
#           both as ~/prof-dev/arms-census3.json's cap-dev3 and rv64 arms)
#   IMAGE   OUT/<arm>/php from build-arms.sh; SCRIPT: the PHP script (adapt-scripts.py's output);
#           both below SHARE, and OUT new and below SHARE
# env: SHARE (default: the parent of OUT's parent), KIT (~/prof-dev/kit3), PL (~/prof-dev/platform-dev3),
#      VMHOST (~/prof-dev/host), NM (~/prof-dev/cap3/build/bin/llvm-nm), LIMIT (wall seconds for the
#      invocation, default 3600: past it the guest is shut down and the run is a FAIL)
# Writes OUT/stdout.txt, OUT/stderr.txt (the script's output, copied back after the window),
# OUT/invocations/, OUT/capprof/window-0.tsv, OUT/profile/summary.txt, OUT/profile.json and
# OUT/result.json (PASS: exit 0, one complete window, every test name of the script and its Total line).
set -euo pipefail
ARM=$1 IMAGE=$(readlink -f "$2") SCRIPT=$(readlink -f "$3") OUT=$4
[ -e "$OUT" ] && { echo "profile-php: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd -P)
SHARE=$(cd "${SHARE:-$OUT/../..}" && pwd -P)
KIT=${KIT:-$HOME/prof-dev/kit3} PL=${PL:-$HOME/prof-dev/platform-dev3} VMHOST=${VMHOST:-$HOME/prof-dev/host}
NM=${NM:-$HOME/prof-dev/cap3/build/bin/llvm-nm} LIMIT=${LIMIT:-3600}
for p in "$OUT" "$IMAGE" "$SCRIPT"; do case "$p" in "$SHARE"/*) ;; *) echo "profile-php: $p is not below $SHARE" >&2; exit 2 ;; esac; done
case $ARM in
  capstone) UP=(--share-model mapped-xattr --exact-bounds); PLAINENV=() ;;
  plain) UP=(--share-model none); PLAINENV=(CAPSTONE_SPEC_PLAIN=1) ;;
  *) echo "profile-php: arm $ARM?" >&2; exit 2 ;;
esac
R=$OUT/record.txt
{ echo "arm $ARM image $IMAGE script $SCRIPT limit $LIMIT"; echo "platform $PL kit $KIT share $SHARE"
  sha256sum "$IMAGE" "$SCRIPT"; } > "$R"
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
( cd "$(dirname "$SCRIPT")" && env SPEC="$OUT/nospec" CAPSTONE_SPEC_RECORD="$OUT" CAPSTONE_SPEC_KIT="$KIT" \
    CAPSTONE_VM_STATE="$OUT/vm" CAPSTONE_VM_SHARE="$SHARE" CAPSTONE_VM_HOST="$VMHOST" \
    CAPSTONE_SPEC_PROFRUN=/tmp/profrun CAPSTONE_SPEC_STAGE=tmpfs "${PLAINENV[@]}" \
    timeout "$LIMIT" bash "$KIT/tools/capstone-run.sh" "$IMAGE" -n "$(basename "$SCRIPT")" \
    > "$OUT/stdout.txt" 2> "$OUT/stderr.txt" ) || rc=$?
end=$(date -u +%FT%TZ); wall=$(( $(date +%s) - t0 ))
vm exec -- sh -c "dmesg | tail -n 200; echo ---; dmesg | wc -l" > "$OUT/dmesg.txt" 2>&1 || true
vm down > /dev/null 2>&1 || true
trap - EXIT
python3 - "$OUT" "$SCRIPT" "$rc" "$start" "$end" "$wall" <<'PY'
import glob, json, os, re, sys
out, script, rc, start, end, wall = sys.argv[1:7]
bad = []
if int(rc) != 0: bad.append(f"exit {rc}" + (" (timeout)" if int(rc) == 124 else ""))
inv = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, "invocations", "[0-9]" * 4 + ".json")))]
guest = [d for d in inv if d.get("route") in ("guest", "guest-plain")]
if len(guest) != 1: bad.append(f"{len(guest)} guest invocations, expected 1")
tsv = glob.glob(os.path.join(out, "capprof", "window-*.tsv"))
if len(tsv) != 1: bad.append(f"{len(tsv)} windows, expected 1")
for p in tsv:
    lines = open(p).read().splitlines()
    if not lines or not lines[-1].startswith("# total "): bad.append(f"{p}: no footer"); continue
    if sum(int(l.split("\t")[3]) for l in lines if not l.startswith("#")) != int(lines[-1].split()[2]):
        bad.append(f"{p}: records do not sum to the footer")
if glob.glob(os.path.join(out, "capprof", "*.invalid")): bad.append("an invalid window")
# Every test the script times must have printed its line, and the Total line last.
names = re.findall(r"end_test\(\$t, (['\"])(.*?)\1", open(script).read())
text = open(os.path.join(out, "stdout.txt"), errors="replace").read()
printed = [l for l in text.splitlines() if l.strip()]
missing = [n for _, n in names if not any(l.startswith(n.replace("\\'", "'")) for l in printed)]
if not names: bad.append("no end_test names found in the script")
if missing: bad.append(f"missing test lines: {missing}")
if not printed or not printed[-1].startswith("Total"): bad.append("no Total line at the end")
r = {"verdict": "FAIL" if bad else "PASS", "reasons": bad, "tests": len(names), "start": start, "end": end,
     "wall_s": int(wall), "guest_exit": guest[0].get("detail", {}).get("guest") if guest else None}
json.dump(r, open(os.path.join(out, "result.json"), "w"), indent=1)
print(r["verdict"], bad, f"wall {wall}s")
PY
mkdir -p "$OUT/profile"
python3 "$KIT/tools/prof-report.py" --out "$OUT" --platform "$PL" --kind "$ARM" --nm "$NM" \
  > "$OUT/profile/summary.txt" 2> "$OUT/profile/report.err" || { echo "prof-report failed: $(tail -1 "$OUT/profile/report.err")" >&2; exit 1; }
head -14 "$OUT/profile/summary.txt"
grep -q '"verdict": "PASS"' "$OUT/result.json"
