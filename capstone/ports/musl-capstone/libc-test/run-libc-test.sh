#!/usr/bin/env bash
# Run every built libc-test domain, one QEMU boot per CHUNK of tests, and summarise.
#
# Why the boot is the unit. A domain that never comes back takes the guest
# with it: a fault on re-entry does (M-1), and so does a domain that spins,
# because it holds the only hart and the guest kernel never runs again. Nothing
# inside the guest can recover, not alarm() in the host process, not kill -9
# from the shell. Measured 2026-09-16: with fifty tests on one boot, a hang in
# the fifteenth cost the other thirty-five, twice.
#
# Boot-to-login is about forty seconds under TCG on this host (fifteen boots
# in nine minutes, 2026-09-16), so the default is one test per boot: a hang
# costs its own boot and nothing else, and the table is exact. CHUNK=10 gets
# batching back for a machine where the boot is the expensive part; there,
# tests known to hang or fault run last, from quarantine.txt, so they only ever
# cost the tail of their chunk. A boot that never reaches a shell is booted
# once more, then it is a verdict of its own, NOBOOT, so a stalled emulator is
# not mistaken for a hung test. Measured 2026-09-16: two of the first twenty
# boots stalled, one in the OpenSBI banner and one after init's last OK, both
# with a compiler build loading the host; the smoke runner's default login
# wait is sixteen minutes, three is plenty for a boot that takes eight seconds.
#
# Every run gets its own log directory, logs/<RUN_ID>/, with the toolchain that
# produced the domains recorded beside the chunk logs; logs/latest points at
# the newest. Two runs with two compilers are two tables, not one overwritten.
# TESTS="inet_pton mntent" runs only those, in the usual order.
#
# The success markers are the boot control and the chunk-done line only. Per-
# test verdicts are results to report, not reasons to call the run broken.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-libc-test}
SHARE_DIR=${SHARE_DIR:-$OUT_DIR/share}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
LOG_DIR=${LOG_DIR:-$OUT_DIR/logs/$RUN_ID}
PER_TEST_TIMEOUT=${PER_TEST_TIMEOUT:-60}
CHUNK=${CHUNK:-1}
# The workload budget the smoke runner keeps separate from boot and setup: a
# generous minute per test in the chunk, plus margin. A hang at the tail of a
# chunk costs this much waiting and no more.
export CAPSTONE_GUEST_COMMAND_TIMEOUT=${CAPSTONE_GUEST_COMMAND_TIMEOUT:-$((CHUNK * 75 + 120))}
export CAPSTONE_QEMU_LOGIN_TIMEOUT=${CAPSTONE_QEMU_LOGIN_TIMEOUT:-180}

OUT_DIR="$OUT_DIR" SHARE_DIR="$SHARE_DIR" bash "$SCRIPT_DIR/build-libc-test.sh"
mkdir -p "$LOG_DIR"; ln -sfn "$LOG_DIR" "$OUT_DIR/logs/latest"
{
  echo "CAPSTONE_LLVM_BUILD_DIR=$CAPSTONE_LLVM_BUILD_DIR"
  "$CAPSTONE_LLVM_BUILD_DIR/bin/clang" --version | head -1
  echo "compiler rev $(git -C "$CAPSTONE_LLVM_BUILD_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "CAPSTONE_QEMU_BINARY=${CAPSTONE_QEMU_BINARY:-}"
} > "$LOG_DIR/toolchain.txt"

# Order: everything not quarantined, then the quarantined tail.
QUAR="$SCRIPT_DIR/quarantine.txt"
mapfile -t ORDER < <( (for d in "$SHARE_DIR"/lt-*.dom; do n=$(basename "$d" .dom)
  if grep -vE '^\s*(#|$)' "$QUAR" | awk '{print $1}' | grep -qx "${n#lt-}"; then echo "2 $n"; else echo "1 $n"; fi
done) | sort | awk '{print $2}')
if [[ -n "${TESTS:-}" ]]; then
  keep=()
  for n in "${ORDER[@]}"; do for t in $TESTS; do [[ "${n#lt-}" == "$t" ]] && keep+=("$n"); done; done
  ORDER=("${keep[@]}")
fi
printf 'order (%d): %s\n' "${#ORDER[@]}" "${ORDER[*]}"

chunk_no=0
for ((i = 0; i < ${#ORDER[@]}; i += CHUNK)); do
  chunk_no=$((chunk_no + 1))
  names=("${ORDER[@]:i:CHUNK}")
  LOG="$LOG_DIR/chunk-$chunk_no.log"
  printf '== chunk %d: %s\n' "$chunk_no" "${names[*]}"
  # Each test in the background with a shell watchdog. This does not rescue a
  # guest whose hart is gone, it only keeps a killable hang from stalling the
  # chunk; the runner's budget is the net under that.
  GUEST_CMD="echo __CAPSTONE_QEMU_BOOT_CONTROL_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; for n in ${names[*]}; do echo LT-BEGIN \$n; /tmp/lt.user /mnt/host/\$n.dom $PER_TEST_TIMEOUT & p=\$!; i=0; while kill -0 \$p 2>/dev/null && [ \$i -lt $((PER_TEST_TIMEOUT + 10)) ]; do sleep 1; i=\$((i+1)); done; if kill -0 \$p 2>/dev/null; then kill -9 \$p 2>/dev/null; echo LT-END \$n rc=137; else wait \$p; echo LT-END \$n rc=\$?; fi; done; echo __CAPSTONE_LIBC_TEST_CHUNK_DONE__"
  for attempt in 1 2; do
    python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
      --share-dir "$SHARE_DIR" --log-file "$LOG" \
      --timeout-multiplier "${TIMEOUT_MULTIPLIER:-8}" \
      --guest-command "$GUEST_CMD" \
      --success-marker '__CAPSTONE_QEMU_BOOT_CONTROL_OK__' \
      --success-marker '__CAPSTONE_LIBC_TEST_CHUNK_DONE__' || printf '== chunk %d did not complete\n' "$chunk_no"
    # A boot that never reached a shell ran nothing: keep its log and boot once more.
    grep -q '__CAPSTONE_QEMU_BOOT_CONTROL_OK__' "$LOG" && break
    if [[ $attempt -eq 1 ]]; then
      mv "$LOG" "$LOG_DIR/chunk-$chunk_no.noboot.log"
      printf '== chunk %d: no shell, booting again\n' "$chunk_no"
    fi
  done
  # The plan for this boot, in the log the summary reads, so a boot that never
  # reached a shell still names what it was to run.
  printf '\n== planned: %s\n' "${names[*]}" >> "$LOG"
done

python3 - "$LOG_DIR" "$OUT_DIR/manifest.txt" "$LOG_DIR/results.txt" <<'PY'
import re, sys, collections, glob
logdir, manifest, out = sys.argv[1:4]
logs = [open(f, errors="replace").read() for f in sorted(glob.glob(logdir + "/chunk-*.log"))]
text = "\n".join(logs)
# riscv64 numbers of the syscalls a single-process libc may still ask for; an
# unknown number stays a number.
SYSNAME = {25: "fcntl", 29: "ioctl", 34: "mkdirat", 35: "unlinkat", 48: "faccessat", 49: "chdir",
    56: "openat", 57: "close", 59: "pipe2", 61: "getdents64", 62: "lseek", 63: "read", 64: "write",
    65: "readv", 66: "writev", 78: "readlinkat", 79: "fstatat", 80: "fstat", 88: "utimensat",
    93: "exit", 94: "exit_group", 98: "futex", 101: "nanosleep", 113: "clock_gettime",
    115: "clock_nanosleep", 129: "kill", 130: "tkill", 132: "sigaltstack", 134: "rt_sigaction",
    135: "rt_sigprocmask", 160: "uname", 165: "getrusage", 169: "gettimeofday", 172: "getpid",
    173: "getppid", 174: "getuid", 178: "gettid", 198: "socket", 214: "brk", 215: "munmap",
    220: "clone", 221: "execve", 222: "mmap", 226: "mprotect", 233: "madvise", 260: "wait4",
    261: "prlimit64", 278: "getrandom", 291: "statx"}
def sysnames(numbers):
    # The domain prints distinct numbers with how often each was asked, "88x9".
    seen = []
    for tok in numbers.split():
        m = re.fullmatch(r"(-?\d+)(x\d+)?", tok)
        name = SYSNAME.get(int(m.group(1)), m.group(1)) + (m.group(2) or "") if m else tok
        if name not in seen: seen.append(name)
    return " ".join(seen)
res = {}
for m in re.finditer(r"LT-BEGIN (\S+)(.*?)LT-RESULT (\S+) status=(-?\d+) rounds=(\d+) (PASS|FAIL)([^\n]*)", text, re.S):
    n = m.group(3)
    if n.startswith("lt-"): n = n[3:]
    if n.endswith(".dom"): n = n[:-4]
    extra = m.group(7).strip()
    u = re.search(r"UNSERVED syscalls: (.*)", m.group(2))
    if u: extra = extra.replace("UNSERVED", "UNSERVED " + sysnames(u.group(1)))
    res[n] = ("PASS" if m.group(6) == "PASS" else "FAIL", int(m.group(4)), int(m.group(5)), extra)
ends = dict(re.findall(r"LT-END (\S+) rc=(\d+)", text))
begun = set(re.findall(r"LT-BEGIN (\S+)", text))
faults = set()
for m in re.finditer(r"LT-BEGIN (\S+)(.*?)(?=LT-BEGIN|__CAPSTONE_LIBC_TEST_CHUNK_DONE__|$)", text, re.S):
    if "domain halted by capability fault" in m.group(2): faults.add(m.group(1))
# A boot whose shell never echoed the boot-control marker ran nothing it planned.
noboot = set()
for t in logs:
    if "__CAPSTONE_QEMU_BOOT_CONTROL_OK__" not in t:
        for m in re.finditer(r"== planned: (.*)", t): noboot.update(m.group(1).split())
rows = []
for line in open(manifest):
    kind, name, *rest = line.rstrip("\n").split(" ", 2)
    reason = rest[0] if rest else ""; key = "lt-" + name
    if kind == "excluded": rows.append((name, "EXCLUDED", reason))
    elif kind == "failed":  rows.append((name, "NOBUILD", reason))
    elif name in res:
        v, st, r, extra = res[name]; rows.append((name, v, f"status={st} rounds={r} {extra}".strip()))
    elif key in faults: rows.append((name, "FAULT", "capability fault, see log"))
    elif ends.get(key) in ("124", "142"): rows.append((name, "TIMEOUT", "alarm after per-test budget"))
    elif ends.get(key) == "137": rows.append((name, "WEDGED", "killed by the loop's watchdog"))
    elif key in begun: rows.append((name, "HUNG", "started, never ended: took its chunk's guest with it"))
    elif key in noboot: rows.append((name, "NOBOOT", "the guest never reached a shell, see log"))
    else: rows.append((name, "NOTRUN", "never started (a chunk-mate hung first, or not in TESTS)"))
counts = collections.Counter(v for _, v, _ in rows)
with open(out, "w") as f:
    for name, v, d in sorted(rows, key=lambda r: (r[1], r[0])): f.write(f"{v:9s} {name:24s} {d}\n")
    f.write("\n" + " ".join(f"{k}={counts[k]}" for k in sorted(counts)) + f" total={len(rows)}\n")
print(open(out).read())
PY
cp "$LOG_DIR/results.txt" "$OUT_DIR/results.txt"
echo "results: $LOG_DIR/results.txt (also $OUT_DIR/results.txt)   chunk logs: $LOG_DIR"
