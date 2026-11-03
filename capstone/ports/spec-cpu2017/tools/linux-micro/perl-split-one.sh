#!/bin/bash
# perl-split-one.sh GROUPDIR OUT KIND PLATFORM: one group of perl-split-prep.py's split in its own guest,
# booted as tools/profile.sh boots one (capprof plugin, -icount 0, no-pcrel, test CSRs, ASLR off,
# syslogd/klogd stopped; share model mapped-xattr; capstone: exact bounds), and invoked as
# tools/capstone-run.sh invokes the test input with CAPSTONE_SPEC_STAGE=tmpfs and the relative-stage
# fix: the executable staged to the guest's tmpfs as /tmp/stage-0001-<name>, run by a relative path
# through 24 "../" (t/TEST re-runs it from t/ as "../$^X"), under profrun (one window), with stdout
# and stderr to the guest's tmpfs and copied back after the window. Then prof-report.py, and the
# group's test.out checked against SPEC's lines for its scripts (name and verdict per script).
#   SLOTS=N  saved-context slots, QEMU property and module parameter (default: the platform's, 1)
#   MODPARAMS, ENV (space-separated NAME=value): an arm's module parameters and guest environment
# Environment as run-micro.sh (KIT, VMHOST, NM, SHARE).
set -uo pipefail
G=$(readlink -f "$1") OUT=$2 KIND=$3 PL=$(readlink -f "$4")
KIT=${KIT:-$HOME/prof-dev/kit} VMHOST=${VMHOST:-$HOME/prof-dev/host} NM=${NM:-$HOME/prof-dev/cap/build/bin/llvm-nm} SHARE=${SHARE:-$HOME}
MODPARAMS=${MODPARAMS:-}; read -r -a ARMENV <<< "${ENV:-}"
name=$(basename "$G"); exe=$(ls "$G"/perlbench_r_base.* | head -1); ename=$(basename "$exe")
[ -e "$OUT" ] && { echo "exists: $OUT" >&2; exit 2; }
mkdir -p "$OUT/capprof" "$OUT/invocations"; OUT=$(readlink -f "$OUT")
case "$OUT" in "$SHARE"/*) ;; *) echo "OUT must be below SHARE" >&2; exit 2 ;; esac
case "$G" in "$SHARE"/*) ;; *) echo "GROUPDIR must be below SHARE" >&2; exit 2 ;; esac
REL=${OUT#"$SHARE"/} GREL=${G#"$SHARE"/}
STATES=${CAPSTONE_SPEC_VM_STATES:-$HOME/.capstone-spec-vm}; mkdir -p "$STATES"
VMSTATE=$STATES/ps-$(printf '%s' "$OUT" | sha256sum | cut -c1-12); rm -rf "$VMSTATE"; mkdir -m 700 "$VMSTATE"; ln -s "$VMSTATE" "$OUT/vm"
export CAPSTONE_QEMU_LOCK=$OUT/qemu.lock
vm() { env PYTHONPATH="$VMHOST" python3 -m capstone_vm --state "$OUT/vm" "$@" < /dev/null; }
UP=(--cpu-prop x-capstone-test-csrs=true); [ "$KIND" = capstone ] && UP+=(--exact-bounds)
if [ -n "${SLOTS:-}" ]; then UP+=(--cpu-prop x-capstone-supervisor-slots=$SLOTS); MODPARAMS="${MODPARAMS:+$MODPARAMS }supervisor_slots=$SLOTS"; fi
sha256sum "$PL/qemu-system-riscv64" "$PL/images/Image" "$PL/adapter/capstone-vexec" "$PL/adapter/module/capstone_vm.ko" \
  "$PL/plugins/libcapprof.so" "$PL/profrun" "$PL/kernel/System.map" "$exe" "$G/t/TEST-$name" > "$OUT/inputs.sha256"
echo "group $name kind $KIND platform $PL slots ${SLOTS:-1} modparams ${MODPARAMS:-none} env ${ENV:-none}" > "$OUT/record.txt"
vm up --profile virtual --memory 8G --share-model mapped-xattr --share "$SHARE" \
  --plugin "$PL/plugins/libcapprof.so" --plugin-arg out="$OUT/capprof" --no-pcrel --icount 0 "${UP[@]}" \
  --qemu "$PL/qemu-system-riscv64" --kernel "$PL/images/Image" --firmware "$PL/images/fw_jump.elf" \
  --rootfs "$PL/images/rootfs.ext2" --launcher "$PL/adapter/capstone-vexec" --job-helper "$PL/adapter/capstone-job" \
  --module "$PL/adapter/module/capstone_vm.ko" --ssh-server "$PL/ssh-server/dropbearmulti" > "$OUT/vm-up.log" 2>&1 \
  || { echo "$name vm up failed"; exit 3; }
trap 'vm down > /dev/null 2>&1 || true' EXIT
cp "$PL/profrun" "$OUT/profrun"
SECS="cd /sys/module/capstone_vm/sections && for f in .* *; do [ -f \"\$f\" ] && echo \"\$f \$(cat \"\$f\")\"; done; cat /proc/cmdline; for p in /sys/module/capstone_vm/parameters/*; do echo \"param \$(basename \$p)=\$(cat \$p)\"; done"
vm exec -- sh -c "echo 0 > /proc/sys/kernel/randomize_va_space && cp /mnt/host/$REL/profrun /tmp/profrun && chmod 0755 /tmp/profrun && killall syslogd klogd 2>/dev/null; $SECS" > "$OUT/module-sections.txt" 2>&1
if [ -n "$MODPARAMS" ]; then
  vm exec -- sh -c "rmmod capstone_vm && insmod /mnt/control/module $MODPARAMS && $SECS" > "$OUT/module-sections.txt" 2>&1 || { echo "$name module reload failed"; exit 4; }
fi
staged=/tmp/stage-0001-$ename
vm exec -- sh -c "cp /mnt/host/$GREL/$ename $staged && chmod 0755 $staged" || { echo "$name staging failed"; exit 4; }
rel="$(printf '../%.0s' $(seq 1 24))${staged#/}"
start=$(date -u +%FT%TZ); rc=0
if [ "$KIND" = capstone ]; then
  EOPT=(); for e in "${ARMENV[@]}"; do EOPT+=(-e "$e"); done
  vm run --cwd "/mnt/host/$GREL" --result "$OUT/invocations/0000.guest.json" -e CAPSTONE_VM_STATS=1 \
    -e CAPSTONE_PROFILE_RECORD=/mnt/host/$REL/records.txt "${EOPT[@]}" \
    --prefix sh --prefix=-c --prefix "exec \"\$0\" \"\$@\" >/tmp/out-0001 2>/tmp/err-0001" --prefix /tmp/profrun \
    -- "$rel" -I. -I./lib "test-$name.pl" || rc=$?
else
  vm exec -- sh -c "cd /mnt/host/$GREL && exec ${ENV:+env $ENV }/tmp/profrun $rel -I. -I./lib test-$name.pl >/tmp/out-0001 2>/tmp/err-0001" || rc=$?
fi
end=$(date -u +%FT%TZ)
vm exec -- sh -c "cp /tmp/out-0001 /mnt/host/$REL/test.out && cp /tmp/err-0001 /mnt/host/$REL/test.err" || rc=97
vm exec -- sh -c "dmesg | grep -i -E 'capstone|BUG|Oops|warn' | tail -n 20" > "$OUT/dmesg.txt" 2>&1
vm down > /dev/null 2>&1; trap - EXIT
python3 - "$OUT" "$G" "$KIND" "$exe" "$rc" "$start" "$end" "$name" <<'PY'
import glob, hashlib, json, os, re, sys
out, g, kind, exe, rc, start, end, name = sys.argv[1:9]
wins = glob.glob(os.path.join(out, "capprof", "window-*.tsv"))
recs = [l for l in open(os.path.join(out, "records.txt")).read().splitlines() if "entry=" in l] if os.path.exists(os.path.join(out, "records.txt")) else []
ranges = sorted(set(l.split(" image=")[0].split("entry=")[1] for l in recs))
d = {"route": "guest" if kind == "capstone" else "guest-plain", "exe": exe,
     "exe_sha256": hashlib.sha256(open(exe, "rb").read()).hexdigest(), "argv": ["test-" + name + ".pl"],
     "rc": int(rc), "start": start, "end": end, "detail": {"window": 0 if len(wins) == 1 else None},
     "image_records": len(recs), "image_ranges": ranges}
json.dump(d, open(os.path.join(out, "invocations", "0000.json"), "w"), indent=1)
if kind == "capstone":
    open(os.path.join(out, "image-records.txt"), "w").write((recs[-1] if recs else "MISSING") + "\n")
def verdicts(path):
    r = {}
    for l in open(path).read().splitlines():
        m = re.match(r"t/(\S+) \.+ (.*)$", l)
        if m: r[m.group(1)] = m.group(2)
    return r
got = verdicts(os.path.join(out, "test.out")) if os.path.exists(os.path.join(out, "test.out")) else {}
want = verdicts(os.path.join(g, "expected.out"))
last = open(os.path.join(out, "test.out")).read().splitlines()[-1:] if got else []
ok = got == want and last == ["All tests successful."] and int(rc) == 0 and len(wins) == 1 and (kind == "plain" or len(ranges) == 1)
bad = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
json.dump({"verdict": "PASS" if ok else "FAIL", "rc": int(rc), "windows": len(wins), "image_records": len(recs),
           "image_ranges": ranges, "scripts": len(want), "mismatch": bad, "last": last},
          open(os.path.join(out, "check.json"), "w"), indent=1)
print(f"{name} {'PASS' if ok else 'FAIL'} rc={rc} windows={len(wins)} records={len(recs)} ranges={len(ranges)} mismatch={bad[:5]}")
PY
python3 "$KIT/tools/prof-report.py" --out "$OUT" --platform "$PL" --kind "$KIND" --nm "$NM" > "$OUT/summary.txt" 2> "$OUT/report.err" \
  || echo "$name prof-report failed: $(tail -1 "$OUT/report.err")"
