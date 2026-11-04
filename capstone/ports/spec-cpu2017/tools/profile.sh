#!/bin/bash
# Profile one benchmark at one size in one arm: runcpu runs and validates it in a freshly booted
# guest whose QEMU carries the capprof plugin, and every guest invocation runs under the window
# helper, so the plugin counts each invocation's instruction attempts in its own window.
#
# usage: profile.sh ARMS ARM SIZE BENCH OUT
#   ARMS   a JSON file describing the host's arms (config/profile-arms.example.json): the share,
#          the SPEC tree, the capstone_vm source, the host ldecod, the platform directory, and for
#          each arm its kind (capstone | plain), config, label, SDK or plain compiler, CPU properties
#          and share model. Nothing here is read from manifest.lock.json: a profiling platform is
#          whatever the arms file says, and this script records what it used.
#   OUT    new directory, below the share: runcpu.log, SPEC's report and log, invocations/,
#          capprof/window-N.tsv (one per guest invocation; invocations/NNNN.json names the window),
#          image-records.txt, module-sections.txt, inputs.sha256, result.json (tools/verdict.py),
#          profile/window-N.json (tools/prof-report.py) and profile.json (the sum).
#
# The platform directory holds qemu-system-riscv64, images/{Image,fw_jump.elf,rootfs.ext2},
# adapter/{capstone-vexec,capstone-job,module/capstone_vm.ko}, ssh-server/dropbearmulti,
# plugins/libcapprof.so, profrun (a static guest binary, tools/profrun.c built by the plain
# compiler) and kernel/System.map of the Image. Counts are reproducible because the guest runs with
# -icount: guest time follows the instruction count, so timers and the supervisor quantum fall at
# the same instructions every run (tools/profile.md).
set -euo pipefail
ARMS=$(readlink -f "$1") ARM=$2 SIZE=$3 BENCH=$4 OUT=$5
[ -e "$OUT" ] && { echo "profile: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd -P)
KIT=$(cd "$(dirname "$0")/.." && pwd -P)
# Values are printed with ~ expanded, so the arms file may be written as the example is.
j() { python3 -c "import json,os,sys;d=json.load(open(sys.argv[1]));[d:=d[k] for k in sys.argv[2].split('.')];print(os.path.expanduser(d) if isinstance(d,str) else d if not isinstance(d,(list,dict)) else json.dumps(d))" "$ARMS" "$1"; }
jopt() { python3 -c "import json,os,sys;d=json.load(open(sys.argv[1]));[d:=d.get(k,{}) for k in sys.argv[2].split('.')];print(os.path.expanduser(d) if isinstance(d,str) else d if d!={} else sys.argv[3])" "$ARMS" "$1" "$2"; }
SHARE=$(cd "$(j share)" && pwd -P); SPEC_ROOT=$(cd "$(j spec_root)" && pwd -P); VMHOST=$(j vm_host); PL=$(cd "$(j platform.dir)" && pwd -P)
KIND=$(j "arms.$ARM.kind"); CFG=$(j "arms.$ARM.config"); LABEL=$(j "arms.$ARM.label"); SHARE_MODEL=$(jopt "arms.$ARM.share_model" mapped-xattr)
case "$OUT" in "$SHARE"/*) ;; *) echo "profile: $OUT is not below the share $SHARE" >&2; exit 2 ;; esac
case "$SPEC_ROOT" in "$SHARE"/*) ;; *) echo "profile: the SPEC tree is not below the share $SHARE" >&2; exit 2 ;; esac
case $KIND in
  capstone) DEFINE=(--define sdk="$(j "arms.$ARM.sdk")") ;;
  plain) DEFINE=(--define rvcc="$(j "arms.$ARM.rvcc")") ;;
  *) echo "profile: arm $ARM has kind $KIND, not capstone or plain" >&2; exit 2 ;;
esac
# CPU properties: the platform's (what its module build needs, e.g. x-capstone-test-csrs=true for the
# paged table, in every arm, the plain one included: the guest boots the module) plus the arm's own.
mapfile -t CPU_PROPS < <(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print('\n'.join(d['platform'].get('cpu_props',[])+d['arms'][sys.argv[2]].get('cpu_props',[])))" "$ARMS" "$ARM")
UP_EXTRA=()
for p in "${CPU_PROPS[@]}"; do [ -n "$p" ] && UP_EXTRA+=(--cpu-prop "$p"); done
[ "$(jopt "arms.$ARM.exact_bounds" false)" = True ] && UP_EXTRA+=(--exact-bounds)
ICOUNT=$(jopt platform.icount 0); MEM=$(jopt platform.memory 8G)
# A work-bounded slice (tools/profile.md, "Work-bounded slices"): the window closes at the K-th
# entry of a function and the run ends there. PROFILE_STOP_AT=SYMBOL:K[,START_SYMBOL:K] (env) or
# the arms file's "stop_at": {"symbol": S, "count": K[, "start_symbol": S2, "start_count": K2]},
# the arm's own over the file's top-level one. The symbol is resolved in the arm's binary.
STOP_JSON=$(python3 - "$ARMS" "$ARM" "${PROFILE_STOP_AT:-}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); env = sys.argv[3]
s = d["arms"][sys.argv[2]].get("stop_at", d.get("stop_at"))
if env:
    parts = [p.rpartition(":") for p in env.split(",")]
    if any(not n or not k.isdigit() for n, _, k in parts) or len(parts) > 2:
        sys.exit("profile: PROFILE_STOP_AT is SYMBOL:K[,START_SYMBOL:K]")
    s = {"symbol": parts[0][0], "count": int(parts[0][2])}
    if len(parts) == 2:
        s.update(start_symbol=parts[1][0], start_count=int(parts[1][2]))
print(json.dumps(s) if s else "")
PY
)
STOP_ARGS=()
if [ -n "$STOP_JSON" ]; then
  stop_sym=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['symbol'])" "$STOP_JSON")
  # the arm's build of this benchmark that defines the symbol; with several, the benchmark's own
  cands=()
  for e in "$SPEC_ROOT/benchspec/CPU/$BENCH/exe/"*"_base.$LABEL"; do
    [ -f "$e" ] && "$(jopt nm llvm-nm)" --defined-only "$e" 2>/dev/null | awk -v s="$stop_sym" '$3 == s && $2 ~ /^[TtWw]$/ {f=1} END {exit !f}' && cands+=("$e")
  done
  [ ${#cands[@]} = 1 ] && STOP_EXE=${cands[0]} || STOP_EXE=$SPEC_ROOT/benchspec/CPU/$BENCH/exe/${BENCH#*.}_base.$LABEL
  [ ${#cands[@]} -ge 1 ] || { echo "profile: no $BENCH binary of label $LABEL defines $stop_sym" >&2; exit 2; }
  printf '%s\n' "${cands[@]}" | grep -qxF "$STOP_EXE" || { echo "profile: $stop_sym is in ${cands[*]}, not one binary" >&2; exit 2; }
  python3 "$KIT/tools/stop-at.py" resolve --nm "$(jopt nm llvm-nm)" --kind "$KIND" --exe "$STOP_EXE" \
    $(python3 -c "import json,sys;d=json.loads(sys.argv[1]);print('--symbol',d['symbol'],'--count',d['count'],*(['--start-symbol',d['start_symbol'],'--start-count',d['start_count']] if d.get('start_symbol') else []))" "$STOP_JSON") \
    > "$OUT/stop-at.json"
  mapfile -t stop_plugin < <(python3 -c "import json,sys;print('\n'.join(json.load(open(sys.argv[1]))['plugin_args']))" "$OUT/stop-at.json")
  for a in "${stop_plugin[@]}"; do STOP_ARGS+=(--plugin-arg "$a"); done
fi
R=$OUT/record.txt; : > "$R"
note() { echo "$*" | tee -a "$R"; }
note "arm $ARM kind $KIND config $CFG label $LABEL size $SIZE bench $BENCH"
note "arm_json $(j "arms.$ARM")"
note "platform $PL icount $ICOUNT memory $MEM share_model $SHARE_MODEL"
note "stop_at ${STOP_JSON:-none}${STOP_ARGS:+ plugin ${STOP_ARGS[*]}}"
[ -f "$PL/platform.json" ] && python3 - "$PL" >> "$R" <<'PY'
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
STATES=$(mkdir -p "${CAPSTONE_SPEC_VM_STATES:-$HOME/.capstone-spec-vm}" && cd "${CAPSTONE_SPEC_VM_STATES:-$HOME/.capstone-spec-vm}" && pwd -P)
VMSTATE=$STATES/$(printf '%s' "$OUT" | sha256sum | cut -c1-12)
if [ -d "$VMSTATE" ]; then   # left by an earlier attempt at the same OUT: reuse unless its guest is still up
  env PYTHONPATH="$VMHOST" python3 -m capstone_vm --state "$VMSTATE" status > /dev/null 2>&1 && { echo "profile: a VM is running in $VMSTATE" >&2; exit 2; }
  rm -rf "$VMSTATE"
fi
mkdir -m 700 "$VMSTATE"; ln -s "$VMSTATE" "$OUT/vm"
vm() { env PYTHONPATH="$VMHOST" python3 -m capstone_vm --state "$OUT/vm" "$@"; }
mkdir -p "$OUT/capprof"
# Apply the contract on the FIRST module load: vm up opens the device for
# --stats before returning. ON must bypass pinned discovery at that point.
MODPARAMS=$(jopt "arms.$ARM.module_params" ""); note "module_params ${MODPARAMS:-none}"
read -r -a MODULE_ARGS <<< "$MODPARAMS"
for param in "${MODULE_ARGS[@]}"; do UP_EXTRA+=(--module-param "$param"); done
vm up --profile virtual --memory "$MEM" --share-model "$SHARE_MODEL" --share "$SHARE" \
  --plugin "$PL/plugins/libcapprof.so" --plugin-arg out="$OUT/capprof" "${STOP_ARGS[@]}" --no-pcrel --icount "$ICOUNT" "${UP_EXTRA[@]}" \
  --qemu "$PL/qemu-system-riscv64" --kernel "$PL/images/Image" --firmware "$PL/images/fw_jump.elf" \
  --rootfs "$PL/images/rootfs.ext2" \
  --launcher "$PL/adapter/capstone-vexec" --job-helper "$PL/adapter/capstone-job" \
  --module "$PL/adapter/module/capstone_vm.ko" --ssh-server "$PL/ssh-server/dropbearmulti" > "$OUT/vm-up.log" 2>&1
trap 'vm down > /dev/null 2>&1 || true' EXIT
python3 -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1]))['command']))" "$VMSTATE/config.json" > "$OUT/qemu-command.json"
# The window helper goes to the guest's tmpfs (a file the host wrote onto a mapped-xattr share is not
# executable there); the module's section addresses symbolize its functions; syslogd and klogd only
# copy kernel messages and would add their instructions to every window; address-space randomization
# is off, so page-fault and pinning work does not vary with the layout (vt4 of the fault test counted
# 365k attempts in one run and 999k in the next with it on).
cp "$PL/profrun" "$OUT/profrun"
vm exec -- sh -c "echo 0 > /proc/sys/kernel/randomize_va_space && cp /mnt/host/${OUT#"$SHARE"/}/profrun /tmp/profrun && chmod 0755 /tmp/profrun && cd /sys/module/capstone_vm/sections && for f in .* *; do [ -f \"\$f\" ] && echo \"\$f \$(cat \"\$f\")\"; done; killall syslogd klogd 2>/dev/null; echo randomize_va_space=\$(cat /proc/sys/kernel/randomize_va_space); uname -a; cat /proc/cmdline; for p in /sys/module/capstone_vm/parameters/*; do echo \"param \$(basename \$p)=\$(cat \$p)\"; done" > "$OUT/module-sections.txt"
# Read back the selected parameters before profiling. The plain arm boots the
# same module contract even though its benchmark executes as ordinary Linux.
if [ -n "$MODPARAMS" ]; then
  vm exec -- sh -c 'for p in /sys/module/capstone_vm/parameters/*; do echo "param $(basename "$p")=$(cat "$p")"; done' > "$OUT/module-params.txt" 2>&1
  python3 - "$OUT/module-params.txt" "${MODULE_ARGS[@]}" <<'PY_PARAMS'
from pathlib import Path
import sys
actual = dict(line.removeprefix('param ').split('=', 1)
              for line in Path(sys.argv[1]).read_text().splitlines() if line.startswith('param '))
def norm(value):
    return {'Y': '1', 'N': '0', 'true': '1', 'false': '0'}.get(value, value)
for item in sys.argv[2:]:
    key, value = item.split('=', 1)
    if norm(actual.get(key, 'missing')) != norm(value):
        sys.exit(f'profile: module parameter mismatch: {key}')
PY_PARAMS
fi
export CAPSTONE_SPEC_PROFRUN=/tmp/profrun
STAGE=$(jopt "arms.$ARM.stage_image" share); note "stage_image $STAGE"
GUEST_ENV=$(python3 -c "import json,sys;print(' '.join(json.load(open(sys.argv[1]))['arms'][sys.argv[2]].get('env',[])))" "$ARMS" "$ARM"); note "guest_env ${GUEST_ENV:-none}"
[ -z "$GUEST_ENV" ] || export CAPSTONE_SPEC_GUEST_ENV=$GUEST_ENV
[ "$STAGE" = share ] || export CAPSTONE_SPEC_STAGE=$STAGE
cd "$SPEC_ROOT"
set +u; source ./shrc > /dev/null; set -u
start=$(date -u +%FT%TZ); rc=0
runcpu --config="$CFG" --action=validate --nobuild --size="$SIZE" --tune=base "${DEFINE[@]}" \
  --define kit="$KIT" --define arm_label="$LABEL" --define vm_state="$OUT/vm" --define vm_share="$SHARE" \
  --define vm_host="$VMHOST" --define ldecod_host="$(j ldecod_host)" --define record="$OUT" \
  "$BENCH" > "$OUT/runcpu.log" 2>&1 || rc=$?
end=$(date -u +%FT%TZ)
# The guest's kernel log of the run: what the module and the kernel said, and how often (the
# kernel formats every line even when the console's level drops it).
vm exec -- sh -c "dmesg | tail -n 400; echo ---; dmesg | wc -l; cat /proc/sys/kernel/printk" > "$OUT/dmesg.txt" 2>&1 || true
vm down > /dev/null 2>&1 || true
trap - EXIT
mkdir -p "$OUT/spec"
for f in $(grep -o "result/CPU2017\.[0-9]*\.[A-Za-z0-9._-]*" "$OUT/runcpu.log" | sort -u); do [ -f "$f" ] && cp "$f" "$OUT/spec/"; done
rundir=$(python3 - "$OUT" <<'PY'
import glob, json, os, sys
for f in sorted(glob.glob(os.path.join(sys.argv[1], "invocations", "[0-9][0-9][0-9][0-9].json"))):
    d = json.load(open(f))
    if d.get("route") in ("guest", "guest-plain"): print(d["cwd"]); break
PY
)
if [ -n "$rundir" ] && [ -d "$rundir" ]; then
  echo "$rundir" > "$OUT/rundir.txt"
  for f in "$rundir"/*.cmd "$rundir"/*.mis "$rundir"/*.out "$rundir"/*.err; do
    [ -f "$f" ] && [ "$(stat -c %s "$f")" -lt 16777216 ] && cp "$f" "$OUT/spec/"
  done
  ( cd "$rundir" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum ) > "$OUT/rundir.sha256"
fi
python3 "$KIT/tools/verdict.py" "$OUT" "$BENCH" "$SIZE" "$rc" "$start" "$end" "$LABEL" > "$OUT/result.json"
# verdict.py's route check: a plain arm's invocations are "guest-plain"
python3 - "$OUT" "$KIND" <<'PY'
import json, sys
out, kind = sys.argv[1:3]
r = json.load(open(f"{out}/result.json"))
if kind == "plain":
    routes = r.get("routes", {})
    r["routes_note"] = "plain arm: guest-plain invocations stand for guest ones"
    if routes.get("guest-plain") and not routes.get("guest"):
        r["reasons"] = [x for x in r.get("reasons", []) if "guest" not in x]
        r["verdict"] = "PASS" if not r["reasons"] else "FAIL"
    json.dump(r, open(f"{out}/result.json", "w"), indent=1)
print(r["verdict"], r.get("reasons"))
PY
# A slice: runcpu cannot validate a run that ended inside the program, so the verdict is SLICE when
# the plugin stopped where asked and everything before the stop passed (tools/stop-at.py finish).
[ -z "$STOP_JSON" ] || python3 "$KIT/tools/stop-at.py" finish "$OUT" || echo "profile: stop-at.py finish failed; the verdict stays" >&2
# Every guest invocation must have its window, complete (capprof's "# total" footer) and valid, and
# no window may exist without an invocation. A run whose verdict is not PASS (or SLICE) or whose windows are
# not all there exits 1, so a batch cannot read it as a result.
windows_rc=0
python3 - "$OUT" <<'PY' || windows_rc=$?
import glob, json, os, re, sys
out = sys.argv[1]; bad = []
inv = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, "invocations", "[0-9]" * 4 + ".json")))]
guest = [d for d in inv if d.get("route") in ("guest", "guest-plain")]
seen = [d.get("detail", {}).get("window") for d in guest]
if sorted(x for x in seen if x is not None) != list(range(len(guest))):
    bad.append(f"window indices {seen} are not 0..{len(guest) - 1} once each")
stray = [f for f in os.listdir(os.path.join(out, "capprof")) if not re.fullmatch(r"window-\d+\.tsv|stopped", f)]
if stray:
    bad.append(f"stray files in capprof/: {stray}")
for d in guest:
    n = d.get("detail", {}).get("window")
    p = os.path.join(out, "capprof", f"window-{n}.tsv")
    if n is None or not os.path.exists(p):
        bad.append(f"window {n} missing"); continue
    if glob.glob(os.path.join(out, "capprof", f"window-{n}.invalid")):
        bad.append(f"window {n} invalid: " + open(glob.glob(os.path.join(out, "capprof", f"window-{n}.invalid"))[0]).read().strip())
    lines = open(p).read().splitlines()
    if not lines or not lines[-1].startswith("# total "):
        bad.append(f"window {n} has no footer (incomplete file)"); continue
    total = int(lines[-1].split()[2])
    counted = sum(int(l.split("\t")[3]) for l in lines if not l.startswith("#"))
    if counted != total:
        bad.append(f"window {n}: records sum to {counted}, footer says {total}")
extra = len(glob.glob(os.path.join(out, "capprof", "window-*.tsv"))) - len(guest)
if extra:
    bad.append(f"{extra} more windows than guest invocations")
r = json.load(open(os.path.join(out, "result.json")))
if r.get("verdict") not in ("PASS", "SLICE"):
    bad.append("verdict " + str(r.get("verdict")))
json.dump({"windows_ok": not bad, "problems": bad, "guest_invocations": len(guest)}, open(os.path.join(out, "windows.json"), "w"), indent=1)
if bad and r.get("verdict") in ("PASS", "SLICE"):   # the verdict file must not say PASS for a run without complete windows
    r["verdict"] = "FAIL"; r["reasons"] = r.get("reasons", []) + ["windows: " + "; ".join(bad)]
    json.dump(r, open(os.path.join(out, "result.json"), "w"), indent=1)
print("windows:", "ok" if not bad else bad)
sys.exit(1 if bad else 0)
PY
# Attribution of every window to layers and functions.
mkdir -p "$OUT/profile"
report_rc=0
python3 "$KIT/tools/prof-report.py" --out "$OUT" --platform "$PL" --kind "$KIND" --nm "$(jopt nm llvm-nm)" > "$OUT/profile/summary.txt" 2> "$OUT/profile/report.err" || report_rc=$?
[ $report_rc = 0 ] || { echo "prof-report failed (rc=$report_rc): $(tail -1 "$OUT/profile/report.err")" | tee -a "$OUT/profile/summary.txt" >&2; }
# A slice the attribution refuses (its stop is not in the image it names) is no result.
[ $report_rc = 0 ] || python3 - "$OUT" <<'PY'
import json, sys
p = sys.argv[1] + "/result.json"; r = json.load(open(p))
if r.get("verdict") == "SLICE":
    r["verdict"] = "FAIL"; r["reasons"] = r.get("reasons", []) + ["prof-report refused the slice: " + open(sys.argv[1] + "/profile/report.err").read().strip().splitlines()[-1]]
    json.dump(r, open(p, "w"), indent=1)
PY
head -30 "$OUT/profile/summary.txt"
[ $windows_rc = 0 ] && [ $report_rc = 0 ] || exit 1
