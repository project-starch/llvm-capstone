#!/bin/bash
# run-micro.sh OUT PLATFORM KIND MEMORY LIST [MODPARAMS] [ENV...]
#   One guest booted the way tools/profile.sh boots it (capprof plugin, -icount 0, no-pcrel,
#   x-capstone-test-csrs=true, ASLR off, syslogd/klogd stopped), then every line of LIST run in it
#   in order, each a process that opens exactly one capprof window around its measured phase
#   (lqmicro.c). Afterwards tools/prof-report.py attributes every window to program / launcher /
#   module / kernel, as for a SPEC run.
#
#   OUT        new directory for this guest's results
#   PLATFORM   a platform directory as profile.sh reads it (qemu-system-riscv64, images/,
#              adapter/, plugins/libcapprof.so, profrun, kernel/System.map)
#   KIND       capstone | plain (the binaries are $MICRO/lqmicro-cap or $MICRO/lqmicro-plain)
#   MEMORY     guest memory, e.g. 8G
#   LIST       lines "[NAME=value ...] -- lqmicro arguments"; NAME=value words are the
#              invocation's guest environment (capstone: the launcher's), except HOG=MB, which
#              starts lq-hog MB in the background first and waits for its READY
#   MODPARAMS  if not empty, the module is reloaded with these insmod parameters first
#   ENV        NAME=value words added to every invocation (e.g. CAPSTONE_VM_HUGE=1)
#
# Environment: MICRO (directory holding lqmicro-cap, lqmicro-plain, lq-hog; default: this
# script's), KIT (the SPEC kit, for prof-report.py; default ~/prof-dev/kit), VMHOST
# (capstone_vm package; default ~/prof-dev/host), NM (llvm-nm; default ~/prof-dev/cap/build/bin/llvm-nm),
# SHARE (the 9p share, must contain OUT; default ~).
#
# What the modes need: fork in a Capstone arm needs a launcher with the CAPSTONE_VM_TORTURE_FORK
# hook (experiment/unpinned-domains; inert unset) and that variable on the line; exec in a Capstone
# arm needs SLOTS=2; compact needs a fragmented guest: a 2G guest and HOG=1200 on its lines.
# Lists used for the 2026-11-03 measurement (one guest per arm and list):
#   capstone: -- touch 4 | -- touch 64 | -- touch 256 | CAPSTONE_VM_TORTURE_FORK=3 -- fork 0 4 |
#             CAPSTONE_VM_TORTURE_FORK=3 -- fork 1024 4 | -- exec 10 /tmp/lqmicro | -- exec 100 /tmp/lqmicro
#   plain:    the same with "-- fork 0 4", "-- fork 1024 4" and LQ_FORK_CHILD=1 copies of both
#   heap:     -- heap 1 1000 | -- heap 3 1000 | -- heap 1 1000 (repeat, for the noise)
#   compact:  HOG=1200 -- compact 64 0 | HOG=1200 -- compact 64 1   (2G guest)
# and micro-table.py over the outputs.
set -uo pipefail
OUT=$1 PL=$(readlink -f "$2") KIND=$3 MEM=$4 LIST=$(readlink -f "$5") MODPARAMS=${6:-}
shift 5; [ $# -gt 0 ] && shift
EXTRA_ENV=("$@")
MICRO=${MICRO:-$(cd "$(dirname "$0")" && pwd)} KIT=${KIT:-$HOME/prof-dev/kit} VMHOST=${VMHOST:-$HOME/prof-dev/host}
NM=${NM:-$HOME/prof-dev/cap/build/bin/llvm-nm} SHARE=${SHARE:-$HOME}
case $KIND in capstone) BIN=$MICRO/lqmicro-cap ;; plain) BIN=$MICRO/lqmicro-plain ;; *) echo "KIND capstone|plain" >&2; exit 2 ;; esac
# REPORT_ONLY=1: only the attribution step below, on an existing OUT
if [ -z "${REPORT_ONLY:-}" ]; then
[ -e "$OUT" ] && { echo "exists: $OUT" >&2; exit 2; }
mkdir -p "$OUT/capprof" "$OUT/invocations" "$OUT/stage"; OUT=$(readlink -f "$OUT")
case "$OUT" in "$SHARE"/*) ;; *) echo "OUT must be below SHARE=$SHARE" >&2; exit 2 ;; esac
fi
REL=${OUT#"$SHARE"/}
if [ -n "${REPORT_ONLY:-}" ]; then OUT=$(readlink -f "$OUT"); else
cp "$BIN" "$OUT/stage/lqmicro"; cp "$MICRO/lqmicro-plain" "$OUT/stage/lqclose"; cp "$MICRO/lq-hog" "$OUT/stage/lq-hog"; cp "$LIST" "$OUT/list.txt"
sha256sum "$PL/qemu-system-riscv64" "$PL/images/Image" "$PL/images/fw_jump.elf" "$PL/images/rootfs.ext2" \
  "$PL/adapter/capstone-vexec" "$PL/adapter/capstone-job" "$PL/adapter/module/capstone_vm.ko" \
  "$PL/plugins/libcapprof.so" "$PL/kernel/System.map" "$BIN" "$MICRO/lq-hog" > "$OUT/inputs.sha256"
fi
if [ -z "${REPORT_ONLY:-}" ]; then
exec 3< "$LIST"   # the list is read on fd 3: every vm call below would otherwise read it from stdin
echo "kind $KIND memory $MEM platform $PL slots ${SLOTS:-1} modparams ${MODPARAMS:-none} env ${EXTRA_ENV[*]:-none}" > "$OUT/record.txt"
STATES=${CAPSTONE_SPEC_VM_STATES:-$HOME/.capstone-spec-vm}; mkdir -p "$STATES"
VMSTATE=$STATES/lq-$(printf '%s' "$OUT" | sha256sum | cut -c1-12); rm -rf "$VMSTATE"; mkdir -m 700 "$VMSTATE"
ln -s "$VMSTATE" "$OUT/vm"
export CAPSTONE_QEMU_LOCK=$OUT/qemu.lock
vm() { env PYTHONPATH="$VMHOST" python3 -m capstone_vm --state "$OUT/vm" "$@"; }
UP=(--cpu-prop x-capstone-test-csrs=false --cpu-prop x-capstone-raw-moves=true)
# Load the requested mode before provision opens its first context.
for param in $MODPARAMS; do UP+=(--module-param "$param"); done
# SLOTS=N: N saved-context slots (QEMU property and module parameter). With the default 1 a domain
# that spawns a domain and waits for it deadlocks: the child waits in vm_ioctl for the slot its
# waiting parent holds (the exec mode, and perlbench's test.pl)
if [ -n "${SLOTS:-}" ]; then UP+=(--cpu-prop x-capstone-supervisor-slots=$SLOTS); MODPARAMS="${MODPARAMS:+$MODPARAMS }supervisor_slots=$SLOTS"; fi
vm up --profile virtual --memory "$MEM" --share-model mapped-xattr --share "$SHARE" \
  --plugin "$PL/plugins/libcapprof.so" --plugin-arg out="$OUT/capprof" --no-pcrel --icount 0 "${UP[@]}" \
  --qemu "$PL/qemu-system-riscv64" --kernel "$PL/images/Image" --firmware "$PL/images/fw_jump.elf" \
  --rootfs "$PL/images/rootfs.ext2" --launcher "$PL/adapter/capstone-vexec" --job-helper "$PL/adapter/capstone-job" \
  --module "$PL/adapter/module/capstone_vm.ko" --ssh-server "$PL/ssh-server/dropbearmulti" > "$OUT/vm-up.log" 2>&1 \
  || { echo "vm up failed"; tail -5 "$OUT/vm-up.log"; exit 3; }
trap 'vm down > /dev/null 2>&1 || true' EXIT
SECS="cd /sys/module/capstone_vm/sections && for f in .* *; do [ -f \"\$f\" ] && echo \"\$f \$(cat \"\$f\")\"; done; cat /proc/cmdline; for p in /sys/module/capstone_vm/parameters/*; do echo \"param \$(basename \$p)=\$(cat \$p)\"; done"
vm exec -- sh -c "echo 0 > /proc/sys/kernel/randomize_va_space; killall syslogd klogd 2>/dev/null; cp /mnt/host/$REL/stage/lqmicro /mnt/host/$REL/stage/lq-hog /mnt/host/$REL/stage/lqclose /tmp/ && chmod 0755 /tmp/lqmicro /tmp/lq-hog /tmp/lqclose; uname -a; grep -E 'MemTotal' /proc/meminfo; cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null; $SECS" > "$OUT/module-sections.txt" 2>&1
if [ -n "$MODPARAMS" ]; then
  vm exec -- sh -c "rmmod capstone_vm && insmod /mnt/control/module $MODPARAMS && $SECS" > "$OUT/module-sections.txt" 2>&1 \
    || { echo "module reload failed"; cat "$OUT/module-sections.txt"; exit 4; }
fi
# the window files' indices (complete .tsv and .invalid alike): an invocation's window is the one
# new .tsv index; a failure leaves none, and a crash inside a window leaves it open (closed below)
nwin() { find "$OUT/capprof" -maxdepth 1 -name 'window-*' ! -name '*.part*' -printf '%f\n' | sed 's/window-\([0-9]*\)\.\(tsv\|invalid\)/\1:\2/' | sort -n | tr '\n' ' '; }
id=0
while IFS= read -r line <&3 || [ -n "$line" ]; do
  case "$line" in ""|"#"*) continue ;; esac
  envpart=${line%%--*}; args=${line#*--}
  read -r -a ENVS <<< "$envpart"; read -r -a ARGS <<< "$args"
  hog=; RUNENV=("${EXTRA_ENV[@]}")
  for e in "${ENVS[@]}"; do case $e in HOG=*) hog=${e#HOG=} ;; *) RUNENV+=("$e") ;; esac; done
  n=$(printf '%04d' $id); before=$(nwin)
  if [ -n "$hog" ]; then
    vm exec -- sh -c "rm -f /tmp/lq-hog-exit; nohup /tmp/lq-hog $hog > /tmp/hog.out 2>&1 < /dev/null & for i in \$(seq 1 600); do grep -q READY /tmp/hog.out && break; sleep 1; done; cat /tmp/hog.out; grep -E 'MemFree' /proc/meminfo" > "$OUT/invocations/$n.hog" 2>&1
  fi
  start=$(date -u +%FT%TZ)
  if [ "$KIND" = capstone ]; then
    EOPT=(); for e in "${RUNENV[@]}"; do EOPT+=(-e "$e"); done
    # The launcher's output (a CAPSTONE_VM_ADD_OK line per grant) goes to the guest's tmpfs and is
    # copied back after the window, as tools/capstone-run.sh does: streamed through the SSH session
    # each line costs ~20k attempts of the SSH server and the network stack inside the window.
    vm run --cwd /tmp --result "$OUT/invocations/$n.guest.json" -e CAPSTONE_VM_STATS=1 \
      -e CAPSTONE_PROFILE_RECORD=/mnt/host/$REL/invocations/$n.rec "${EOPT[@]}" \
      --prefix sh --prefix=-c --prefix 'exec "$0" "$@" >/tmp/lq.out 2>/tmp/lq.err' -- /tmp/lqmicro "${ARGS[@]}" \
      > "$OUT/invocations/$n.vm.out" 2> "$OUT/invocations/$n.vm.err"; rc=$?
    vm exec -- sh -c "cat /tmp/lq.out; rm -f /tmp/lq.out" > "$OUT/invocations/$n.out" 2>/dev/null
    vm exec -- sh -c "cat /tmp/lq.err; rm -f /tmp/lq.err" > "$OUT/invocations/$n.err" 2>/dev/null
  else
    vm exec -- sh -c "cd /tmp && exec env ${RUNENV[*]} /tmp/lqmicro ${ARGS[*]}" > "$OUT/invocations/$n.out" 2> "$OUT/invocations/$n.err"; rc=$?
  fi
  end=$(date -u +%FT%TZ)
  if [ $rc != 0 ]; then vm exec -- /tmp/lqclose close > /dev/null 2>&1; fi
  after=$(nwin)
  [ -n "$hog" ] && vm exec -- sh -c "killall lq-hog 2>/dev/null; rm -f /tmp/lq-hog-exit" > /dev/null 2>&1
  # a child the experiment launcher forked sleeps for CAPSTONE_VM_TORTURE_FORK seconds (plain: 3 s);
  # let it exit before the next window opens
  case " ${ARGS[*]} " in *" fork "*) vm exec -- sleep 5 > /dev/null 2>&1 ;; esac
  python3 - "$OUT" "$n" "$KIND" "$OUT/stage/lqmicro" "$rc" "$before" "$after" "$start" "$end" "${RUNENV[*]}" "${ARGS[@]}" <<'PY'
import hashlib, json, os, sys
out, n, kind, exe, rc, before, after, start, end, env = sys.argv[1:11]; args = sys.argv[11:]
rec = os.path.join(out, "invocations", n + ".rec")
lines = [l for l in open(rec).read().splitlines() if "entry=" in l] if os.path.exists(rec) else []
d = {"route": "guest" if kind == "capstone" else "guest-plain", "exe": exe,
     "exe_sha256": hashlib.sha256(open(exe, "rb").read()).hexdigest(), "argv": args, "env": env,
     "rc": int(rc), "start": start, "end": end, "windows_before": before, "windows_after": after,
     "detail": {"window": None},
     "image_records": len(lines), "image_record": lines[-1] if lines else None,
     "image_ranges": sorted(set(l.split(" image=")[0].split("entry=")[1] for l in lines))}
new = sorted(set(after.split()) - set(before.split()), key=lambda x: int(x.split(":")[0]))
if int(rc) == 0 and len(new) == 1 and new[0].endswith(":tsv"):
    d["detail"]["window"] = int(new[0].split(":")[0])
json.dump(d, open(os.path.join(out, "invocations", n + ".json"), "w"), indent=1)
print(f"{n} rc={rc} window {d['detail']['window']} new {' '.join(new) or '-'} records={len(lines)} {' '.join(args)}")
PY
  id=$((id+1))
done
vm exec -- sh -c "for p in /sys/module/captag/parameters/* /sys/module/capstone_vm/parameters/unpinned_faults; do [ -f \$p ] && echo \"\$p=\$(cat \$p)\"; done; grep -E 'pgmigrate|thp_fault_alloc|compact_' /proc/vmstat; dmesg | grep -i -E 'capstone|captag|BUG|Oops|warn' | tail -n 20" > "$OUT/guest-after.txt" 2>&1
vm down > /dev/null 2>&1; trap - EXIT
fi
# prof-report: one image record per window (the domain's own, written last at its exit; an exec
# run's children write theirs before it), and only the invocations that made exactly one window
python3 - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]; recs = []
for f in sorted(glob.glob(os.path.join(out, "invocations", "[0-9]" * 4 + ".json"))):
    d = json.load(open(f))
    if d["detail"]["window"] is None:
        d["route"] = "no-window"; json.dump(d, open(f, "w"), indent=1); continue
    if d["route"] == "guest": recs.append(d["image_record"] or "MISSING")
open(os.path.join(out, "image-records.txt"), "w").write("".join(r + "\n" for r in recs))
PY
python3 "$KIT/tools/prof-report.py" --out "$OUT" --platform "$PL" --kind "$KIND" --nm "$NM" > "$OUT/summary.txt" 2> "$OUT/report.err" \
  || { echo "prof-report failed: $(tail -2 "$OUT/report.err")"; exit 5; }
grep -E '^window' "$OUT/summary.txt" | head -40
