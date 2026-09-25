#!/usr/bin/env bash
# M0 of the intcap plan (capstone/docs/plans/2026-09-25-intcap-implementation.md): compile the
# PostgreSQL 17.5 backend with CHERI's own clang for CheriBSD riscv64-purecap, where a Datum is a
# capability-carrying integer, and collect what CHERI's provenance diagnostics say about it. No code of
# ours is involved; this measures PostgreSQL, before any compiler work.
#
# Caveat, and why the classifier looks at type names: on CheriBSD purecap EVERY uintptr_t is an intcap,
# while the plan's stage 1 changes only Datum. A diagnostic whose operand types mention Datum is ours; one
# that mentions only uintptr_t/intptr_t is a site that stays an address in stage 1.
#
# Output: $ROOT/warnings.log (raw, local only) and a result-lines file (committed).
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${M0_ROOT:-/tmp/capstone/pg-intcap-m0}
TARBALL=${M0_TARBALL:-/tmp/capstone/pg-mmgr-host/pg.tar.bz2}
SDK=${CHERI_SDK:-/home/biecho/cheri/output/sdk}
SYSROOT=${CHERI_SYSROOT:-$SDK/sysroot-riscv64-purecap}
JOBS=${M0_JOBS:-6}
OUT=${M0_RESULT:-$SCRIPT_DIR/results/$(date +%F)/intcap-m0.txt}
WARN="-Wcheri-provenance -Wcheri-capability-misuse -Wcheri-bitwise-operations -Wshorten-cap-to-int"

for f in "$TARBALL" "$SDK/bin/clang" "$SYSROOT/usr/include/stdint.h"; do
  [[ -e $f ]] || { echo "missing: $f" >&2; exit 2; }
done
rm -rf "$ROOT"; mkdir -p "$ROOT"
tar -xjf "$TARBALL" -C "$ROOT"
SRC=$(ls -d "$ROOT"/postgresql-17.* | head -1)
[[ -d $SRC ]] || { echo "no source tree under $ROOT" >&2; exit 2; }

CC_LINE="$SDK/bin/clang --target=riscv64-unknown-freebsd13 -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax --sysroot=$SYSROOT -B$SDK/bin"
cd "$SRC"
CC="$CC_LINE" CFLAGS="-O2 $WARN" AR="$SDK/bin/llvm-ar" RANLIB="$SDK/bin/llvm-ranlib" \
  ./configure --host=riscv64-unknown-freebsd13 --without-readline --without-zlib --without-icu \
  --without-libxml --without-gssapi --without-openssl --without-ldap \
  --with-system-tzdata=/usr/share/zoneinfo > "$ROOT/configure.log" 2>&1 \
  || { echo "configure failed; see $ROOT/configure.log" >&2; exit 2; }
grep -E "define (SIZEOF_VOID_P|SIZEOF_SIZE_T|MAXIMUM_ALIGNOF|ALIGNOF_DOUBLE)" src/include/pg_config.h > "$ROOT/config-values.txt"

# -k: keep going past files that do not compile, and count them rather than stop at the first.
make -k -j"$JOBS" -C src/common > "$ROOT/make-common.log" 2>&1
make -k -j"$JOBS" -C src/port > "$ROOT/make-port.log" 2>&1
make -k -j"$JOBS" -C src/backend > "$ROOT/make-backend.log" 2>&1
cat "$ROOT"/make-*.log > "$ROOT/warnings.log"

mkdir -p "$(dirname "$OUT")"
python3 - "$ROOT" "$SRC" "$OUT" <<'PY'
import sys, re, collections, pathlib
root, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(f"{root}/warnings.log", errors="replace").read()
diag = re.compile(r"^(?P<file>[^\s:]+\.[ch]):(?P<line>\d+):\d+: (?P<kind>warning|error): (?P<msg>.*?) \[(?P<flag>-W[\w-]+)\]$", re.M)
err  = re.compile(r"^(?P<file>[^\s:]+\.[ch]):(?P<line>\d+):\d+: error: (?P<msg>.*)$", re.M)
sites = {}
for m in diag.finditer(text):
    f = m["file"].replace(src + "/", "")
    key = (f, int(m["line"]), m["flag"])
    scope = "Datum" if "Datum" in m["msg"] else ("uintptr/intptr" if re.search(r"u?intptr_t", m["msg"]) else "other")
    sites.setdefault(key, (scope, m["msg"][:140]))
errors = sorted({(e["file"].replace(src + "/", ""), int(e["line"]), e["msg"][:120]) for e in err.finditer(text)})
objs = len(list(pathlib.Path(src, "src").rglob("*.o")))
by = collections.Counter((k[2], v[0]) for k, v in sites.items())
lines = [f"# intcap M0: PostgreSQL 17.5 compiled by CHERI clang (riscv64-purecap), unique diagnostic sites",
         f"objects built: {objs}", f"compile errors (unique): {len(errors)}",
         "config: " + " ".join(l.split()[1] + "=" + l.split()[2] for l in open(f"{root}/config-values.txt"))]
lines += [f"sites {flag} [{scope}]: {n}" for (flag, scope), n in sorted(by.items())]
lines += ["", "## sites (file:line flag [scope] message)"]
lines += [f"{f}:{l} {fl} [{s}] {msg}" for (f, l, fl), (s, msg) in sorted(sites.items())]
lines += ["", "## compile errors"] + [f"{f}:{l} {m}" for f, l, m in errors]
if not sites and not errors and objs == 0:
    print("no objects and no diagnostics: the build did not run", file=sys.stderr); sys.exit(2)
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:4 + len(by)]))
PY
