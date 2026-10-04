#!/usr/bin/env bash
# Build libc-capstone.a from whatever of musl the compiler currently accepts.
#
# Deliberately builds the PARTIAL set rather than waiting for 100 %: the useful
# question is not "does all of musl compile" but "what does a given program
# actually pull in", and only a linkable archive can answer that. Undefined
# symbols at link time are the work list; see README.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-capstone-build}
OBJ_DIR="$OUT_DIR/obj"
ARCHIVE=${ARCHIVE:-$OUT_DIR/libc-capstone.a}
AR=${CAPSTONE_LLVM_AR:-$CAPSTONE_LLVM_BIN/llvm-ar}

MUSL_SRC_DIR=$(bash "$SCRIPT_DIR/prepare-musl-capstone.sh" | tail -1)

rm -rf "$OBJ_DIR"
mkdir -p "$OBJ_DIR"

# The survey owns the flags and the file set; --objects makes it keep the
# output. It exits 1 on a regression, which must not stop the build here: a
# partial archive is the point. Only a harness error (2) is fatal.
set +e
# MUSL_SURVEY_JOBS: the survey defaults to min(16, cpus); a shared host may need fewer.
python3 "$SCRIPT_DIR/survey-musl-capstone.py" "$MUSL_SRC_DIR" \
        ${MUSL_SURVEY_JOBS:+--jobs "$MUSL_SURVEY_JOBS"} \
        --objects "$OBJ_DIR" > "$OUT_DIR/survey.txt"
survey_status=$?
set -e
if [[ $survey_status -ge 2 ]]; then
  cat "$OUT_DIR/survey.txt" >&2
  echo "survey could not measure; build aborted" >&2
  exit 2
fi

objects=("$OBJ_DIR"/*.o)
if [[ ${#objects[@]} -eq 0 || ! -e "${objects[0]}" ]]; then
  echo "no objects produced in $OBJ_DIR" >&2
  exit 2
fi

# UNDER LTO, VERIFY EVERY MEMBER SURVIVES CODEGEN, and drop the ones that do not, loudly and by name.
# Ported from the c128 line (origin/rebased/c128-3-musl). Under LTO, instruction selection moves to the LINK,
# so a member the backend cannot select for stops being one bad object and becomes a failed link -- and lld
# extracts every bitcode member that defines a runtime LIBCALL name (acosl, the fp128 math family) whether the
# program calls it or not (B0, 2026-10-04: "<internal>: reference to acosl", then C-43). Codegen is run with
# the SAME -mllvm options the link will pass as --plugin-opt (every one in MUSL_CAPSTONE_EXTRA_CFLAGS, not
# only the ABI switch). A no-op without -flto.
if [[ " ${MUSL_CAPSTONE_EXTRA_CFLAGS:-} " == *" -flto "* ]]; then
  echo "LTO build: verifying every member survives codegen"
  readarray -t _drop < <(CAPSTONE_LLC="$CAPSTONE_LLVM_BIN/llc" \
                         CAPSTONE_ABI_FLAGS="${MUSL_CAPSTONE_EXTRA_CFLAGS:-}" \
                         VERIFY_JOBS="${MUSL_SURVEY_JOBS:-8}" \
                         python3 - "${objects[@]}" <<'VERIFY'
import concurrent.futures as cf, os, shlex, subprocess, sys
llc = os.environ["CAPSTONE_LLC"]
words = shlex.split(os.environ.get("CAPSTONE_ABI_FLAGS", ""))
llc_flags = [words[i + 1] for i, w in enumerate(words) if w == "-mllvm" and i + 1 < len(words)]
objs = sys.argv[1:]
if not objs:
    sys.exit("no objects to verify")
BITCODE_MAGIC = bytes.fromhex("4243c0de")   # hex on purpose: a \x escape was once mangled, and the
                                           # verification then passed 1377 objects having read none
seen = [0]
def check(o):
    with open(o, "rb") as fh:
        if fh.read(4) != BITCODE_MAGIC:
            return None
    seen[0] += 1
    r = subprocess.run([llc, "-mtriple=capstone64-unknown-elf", "-mattr=+m,+a", *llc_flags,
                        "-filetype=obj", o, "-o", os.devnull], capture_output=True)
    if r.returncode == 0:
        return None
    err = (r.stderr or b"").decode("utf-8", "replace")
    first = next((l for l in err.splitlines()
                  if "error" in l or "LLVM ERROR" in l or "Cannot select" in l or "Assertion" in l), "")
    return "%s\t%s" % (o, first.strip()[:120])
with cf.ThreadPoolExecutor(max_workers=int(os.environ.get("VERIFY_JOBS", "8"))) as ex:
    found = [r for r in ex.map(check, objs) if r]
# THE SELF-TEST: under LTO essentially every member is bitcode, so recognising none means the magic test is
# broken, not that the archive is clean.
if seen[0] == 0:
    sys.exit("verification recognised no bitcode among %d objects; refusing to report a clean archive" % len(objs))
print("VERIFIED\t%d bitcode members, llc flags %s" % (seen[0], " ".join(llc_flags)), file=sys.stderr)
for line in found:
    print(line)
VERIFY
)
  if (( ${#_drop[@]} )); then
    echo "  dropping ${#_drop[@]} member(s) the backend cannot codegen:"
    _keep=()
    for o in "${objects[@]}"; do
      _hit=0
      for d in "${_drop[@]}"; do
        if [[ ${d%%$'\t'*} == "$o" ]]; then _hit=1; break; fi
      done
      (( _hit )) || _keep+=("$o")
    done
    for d in "${_drop[@]}"; do
      printf '    %-34s %s\n' "$(basename "${d%%$'\t'*}")" "${d#*$'\t'}"
    done
    printf '%s\n' "${_drop[@]}" > "$OUT_DIR/dropped-members.txt"
    objects=("${_keep[@]}")
  else
    echo "  every member codegens"
  fi
fi

rm -f "$ARCHIVE"
"$AR" rcs "$ARCHIVE" "${objects[@]}"

grep -E '^(surveyed|compiled|failed)' "$OUT_DIR/survey.txt"
printf 'archived       %d objects -> %s\n' "${#objects[@]}" "$ARCHIVE"
