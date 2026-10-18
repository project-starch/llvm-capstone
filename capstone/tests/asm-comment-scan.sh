#!/usr/bin/env bash
# asm-comment-scan.sh -- the two ways a COMMENT in a capability .S test breaks the build.
#
# WHY THIS EXISTS. verif/tests/custom/capstone/*.S goes through cpp before the assembler, so a `#` line is not a
# comment to cpp. Two distinct traps, each of which has cost runs more than once:
#
#   1. A MACRO NAME followed by `(`, with or without a space, is an invocation: `the RETURN (0)` fails with
#      "macro RETURN requires 3 arguments". Instances 2026-10-01 (three), 2026-10-04 (two), 2026-10-14.
#   2. A DIRECTIVE WORD right after the `#` makes the line a directive: a header line `#   if it reproduces: ...`
#      is `#if`, and the build dies with "missing binary operator before token" plus "unterminated #if".
#      Instances 2026-10-18 (two, in the same session, the second AFTER the lesson had been written down).
#
# Both failures are invisible in the obvious place: cva6.py reports RC=1 and then SIMULATES THE PREVIOUS BUILD'S
# object, so a per-test log appears with the old program's readings. A run whose readings exactly repeat the previous
# run's is the first sign, which is why this check belongs BEFORE the run and not in the reading of it.
#
# USAGE
#     bash capstone/tests/asm-comment-scan.sh <file.S> [more.S ...]      # exit 1 = BLOCKED
#     bash capstone/tests/asm-comment-scan.sh --self-test                # proves the check can FIRE
#
# A script, not a judgement call, and not a subagent: it is mechanical, it runs on every edit, and its value is that
# nobody has to remember the two shapes. ASM_INSN is where the macro list comes from; override it for a tree whose
# headers live elsewhere.
set -u

ASM_INSN="${ASM_INSN:-}"
DIRECTIVES='if|ifdef|ifndef|else|elif|endif|define|undef|include|line|error|warning|pragma'

find_asm_insn() {
    [ -n "$ASM_INSN" ] && { echo "$ASM_INSN"; return; }
    local d
    d=$(dirname -- "$1")
    while [ "$d" != "/" ] && [ -n "$d" ]; do
        [ -f "$d/asm_insn.h" ] && { echo "$d/asm_insn.h"; return; }
        d=$(dirname -- "$d")
    done
    echo ""
}

scan_one() {
    local f="$1" hdr
    hdr=$(find_asm_insn "$f")
    ASM_INSN_PATH="$hdr" python3 - "$f" <<'PY'
import os, re, sys
path = sys.argv[1]
hdr = os.environ.get("ASM_INSN_PATH", "")
macros = set()
if hdr and os.path.exists(hdr):
    macros = set(re.findall(r'#define\s+([A-Z_][A-Z0-9_]*)\s*\(', open(hdr).read()))
DIRS = r'(if|ifdef|ifndef|else|elif|endif|define|undef|include|line|error|warning|pragma)\b'
REAL = ("#include", "#define", "#ifdef", "#ifndef", "#else", "#endif", "#if ", "#if(", "#undef", "#elif", "#pragma", "#error", "#warning", "#line")
bad = []
for i, line in enumerate(open(path, errors="replace").read().splitlines(), 1):
    s = line.strip()
    # (2) a directive word after the #, on a line that is plainly prose (indented after the #)
    if re.match(r'^#\s+' + DIRS, line) and not any(line.startswith(k) for k in REAL):
        bad.append((i, "directive-like comment", line[:100]))
    # (1) a macro name followed by ( anywhere in the comment part
    if "#" in line and not any(s.startswith(k) for k in REAL):
        comment = line.split("#", 1)[1]
        for m in macros:
            if re.search(r'\b' + m + r'\s*\(', comment):
                bad.append((i, "macro %s invoked in a comment" % m, line[:100]))
                break
for i, why, text in bad:
    print("%s:%d: %s: %s" % (path, i, why, text))
sys.exit(1 if bad else 0)
PY
}

if [ "${1:-}" = "--self-test" ]; then
    t=$(mktemp -d)
    cat > "$t/asm_insn.h" <<'EOF'
#define RETURN(a,b,c)  .insn r 0
EOF
    printf '%s\n' '# a header' '#   if it reproduces: this line is a directive' '# the RETURN (0) path' 'nop' > "$t/bad.S"
    printf '%s\n' '# a header' '#   a reproduction reads: this line is prose' '# the return (0) path' '#ifdef X' '#endif' 'nop' > "$t/good.S"
    out=$(scan_one "$t/bad.S"); rc=$?
    n=$(printf '%s\n' "$out" | grep -c ':')
    echo "$out"
    good_out=$(scan_one "$t/good.S"); grc=$?
    rm -rf "$t"
    if [ "$rc" = "1" ] && [ "$n" -ge 2 ] && [ "$grc" = "0" ]; then
        echo "SELF-TEST PASS (both shapes caught, the clean file accepted)"
        exit 0
    fi
    echo "SELF-TEST FAILED (bad-rc=$rc hits=$n good-rc=$grc)"
    exit 1
fi

[ "$#" -gt 0 ] || { echo "usage: asm-comment-scan.sh <file.S> ... | --self-test" >&2; exit 2; }
status=0
for f in "$@"; do
    [ -f "$f" ] || { echo "asm-comment-scan: no such file: $f" >&2; status=1; continue; }
    scan_one "$f" || status=1
done
[ "$status" = "0" ] && echo "asm-comment-scan: clean ($# file(s))"
exit "$status"
