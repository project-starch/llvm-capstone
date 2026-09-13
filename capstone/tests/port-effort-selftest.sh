#!/usr/bin/env bash
# Negative test for port-effort.py. Needs no board, no build and no QEMU.
#
# The number this tool produces goes into the paper as the cost of a port, and the one way it can
# go wrong quietly is by counting a hunk as nothing because nobody classified it. So every case
# below states both halves: what must be counted and what must be REFUSED.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TOOL=$HERE/port-effort.py
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/p.patch" <<'PATCH'
Two files, three hunks.

--- a/alloc.c
+++ b/alloc.c
@@ -10,3 +10,5 @@
 keep
-gone
+one
+two
 keep
@@ -40,2 +42,5 @@
 keep
+three
+  /* a comment line */
+  ** a continuation line
+
 keep
--- a/app.c
+++ b/app.c
@@ -5,1 +5,2 @@
 keep
+four
PATCH

run() { python3 "$TOOL" "$WORK/p.patch" >"$WORK/out" 2>"$WORK/err"; echo $?; }

fail=0
check() {  # $1 = label, $2 = expected exit, $3 = a string that must appear in out+err
  local got; got=$(run)
  local body; body=$(cat "$WORK/out" "$WORK/err")
  if [[ $got == "$2" && $body == *"$3"* ]]; then
    printf "  ok    %-42s exit %s\n" "$1" "$got"
  else
    printf "  FAIL  %-42s exit %s wanted %s, looking for %s\n" "$1" "$got" "$2" "$3"
    sed 's/^/          /' "$WORK/out" "$WORK/err"; fail=1
  fi
}

check "no .classes file at all is refused" 2 "every hunk needs a class"

printf 'alloc.c:10 hierarchy allocator first hunk\n' > "$WORK/p.patch.classes"
check "a hunk with no class is refused" 2 "app.c:5"

printf 'alloc.c:10 hierarchy allocator x\nalloc.c:42 metadata allocator y\napp.c:5 nonsense application z\n' > "$WORK/p.patch.classes"
check "a class outside the vocabulary is refused" 2 "is not one of"

printf 'alloc.c:10 hierarchy allocator x\nalloc.c:42 metadata allocator y\napp.c:5 hierarchy nowhere z\n' > "$WORK/p.patch.classes"
check "a level outside the vocabulary is refused" 2 "level 'nowhere'"

printf 'alloc.c:10 hierarchy allocator x\nalloc.c:42 metadata allocator y\napp.c:5 hierarchy application z\n' > "$WORK/p.patch.classes"
check "a fully classified patch counts" 0 "TOTAL"
check "added and removed are not netted" 0 "hierarchy   allocator           2        0        0        1"
check "the application count is called out" 0 "which the claim says is zero: +1/-0"
check "comment lines are counted apart from code" 0 "metadata    allocator           1        2        1"
check "a blank line has a column of its own" 0 "TOTAL                           4        2        1"
check "a refuted hypothesis is reported as refuted" 0 "which refutes it"

printf 'alloc.c:10 hierarchy allocator x\nalloc.c:42 metadata allocator y\napp.c:5 hierarchy application z\nalloc.c:999 hierarchy allocator gone\n' > "$WORK/p.patch.classes"
check "a class for a hunk that is gone warns" 0 "no longer has"

if [[ $fail == 0 ]]; then echo "port-effort-selftest: every case behaved"; else echo "port-effort-selftest: FAILED"; fi
exit $fail
