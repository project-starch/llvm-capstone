#!/usr/bin/env bash
# THE GATE for capstone_sqlite_stdio.c. Compares the shim's output against the host's printf for
# every conversion measured to reach it from speedtest1, then proves the comparison can FAIL.
#
# WHY A REFERENCE COMPARISON AND NOT A GOLDEN FILE: a golden file records what the shim did on the
# day it was written, which is worthless if that day was wrong. glibc is the definition of correct
# here, so the test asserts agreement with it.
#
# WHY TWO TRANSLATION UNITS: the shim is written for a freestanding domain and #defines printf,
# stdout, FILE and friends. The host's <stdio.h> cannot be in scope at the same time, so the
# reference strings are produced in a separate unit and only compared here. That is also the honest
# shape -- the shim never sees a host header, exactly as in the real build.
#
#   usage: check-stdio-shim.sh          # exit 0 = shim agrees with the reference AND the test bites
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
CC=${CC:-cc}

cat > "$WORK/ref.c" <<'REF'
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
int ref_snprintf(char *out, unsigned long n, const char *fmt, ...){
  va_list ap; int r; va_start(ap, fmt); r = vsnprintf(out, n, fmt, ap); va_end(ap); return r; }
void *host_memcpy(void *d, const void *s, unsigned long n){ return memcpy(d, s, n); }
int host_strcmp(const char *a, const char *b){ return strcmp(a, b); }
int host_report(const char *f, const char *a, const char *b, const char *c){
  return printf(f, a, b, c), printf("\n"); }
REF

# $1 = path to the shim .c under test (the real one, or a deliberately broken copy)
gen_case() {
cat <<CASE
#define FILE capstone_shim_FILE
#include "$1"
#undef FILE
int ref_snprintf(char *out, unsigned long n, const char *fmt, ...);
void *host_memcpy(void *d, const void *s, unsigned long n);
int host_strcmp(const char *a, const char *b);
int host_report(const char *f, const char *a, const char *b, const char *c);
static char sink_buf[4096]; static unsigned long sink_len;
void capstone_stdio_on_exit(int code){ (void)code; }   /* never reached here; the symbol is needed */
unsigned long capstone_stdio_sink(const char *t, unsigned long n){
  if (sink_len + n > sizeof sink_buf) n = sizeof sink_buf - sink_len;
  host_memcpy(sink_buf + sink_len, t, n); sink_len += n; return n; }
static int fails;
#define CHECK(ref, ...) do { char want[512]; char *got; sink_len = 0; \\
  ref_snprintf(want, sizeof want, ref, __VA_ARGS__); \\
  capstone_printf(ref, __VA_ARGS__); sink_buf[sink_len] = 0; got = sink_buf; \\
  if (host_strcmp(want, got)) { host_report("MISMATCH fmt=%s want=[%s] got=[%s]", ref, want, got); fails++; } \\
} while (0)
int main(void){
  /* Every conversion literally present in speedtest1.c, plus the boundaries around each. */
  CHECK("%d", 0); CHECK("%d", 42); CHECK("%d", -7); CHECK("%d", -2147483647-1);
  CHECK("%5d", 42); CHECK("%-28s", "name"); CHECK("%.48s", "short");
  CHECK("%.*s", 3, "abcdef"); CHECK("%.*s", 0, "abcdef");
  CHECK("%02x", 5); CHECK("%02x", 255); CHECK("%03d", 7); CHECK("%4d", 1234);
  CHECK("%llu", 0ULL); CHECK("%llu", 18446744073709551615ULL);
  CHECK("%s", "plain"); CHECK("%u", 4294967295u);
  CHECK("%-10d|", 5); CHECK("%010d", -42); CHECK("%x", 48879);
  CHECK("a%db%sc", 1, "x");

  /* THE DROPPED COUNTER'S OWN CONTROLS, and they are not decoration.
   *
   * Every speedtest1 run reported so far says DROPPED 0, meaning no output was lost. Until these
   * two cases existed that was an UNPROVEN ZERO: the counter increments at exactly two sites in
   * capstone_sqlite_stdio.c -- an unsupported conversion, and a sink that refused bytes -- and the
   * cases above reach NEITHER. They use only supported conversions, and the harness sink resets to
   * empty before each one against a 4096-byte buffer, so it can never refuse. A counter that has
   * never been non-zero anywhere says nothing when it reads zero. */
  {
    unsigned long before, after;

    /* (a) An unsupported conversion must be counted. %f is the realistic one: the shim deliberately
       implements no float formatting, and if a float ever reaches these calls this is what says so
       rather than the output quietly carrying a literal "%f". */
    sink_len = 0; before = capstone_stdio_dropped;
    capstone_printf("%f", 1);
    after = capstone_stdio_dropped;
    if (after <= before) {
      host_report("DROPPED-FAIL: an unsupported conversion was not counted%s%s%s", "", "", "");
      fails++;
    }

    /* (b) A sink that refuses must have its shortfall counted, byte for byte. Fill the harness sink
       to eight bytes short of full, then ask for sixteen: eight are taken and eight must be
       counted. This is the path the payload region takes when a report outgrows it. */
    sink_len = sizeof sink_buf - 8; before = capstone_stdio_dropped;
    capstone_printf("%s", "0123456789ABCDEF");
    after = capstone_stdio_dropped;
    if (after != before + 8) {
      host_report("DROPPED-FAIL: sink shortfall miscounted%s%s%s", "", "", "");
      fails++;
    }
    sink_len = 0;
  }

  if (!fails) host_report("SHIM OK: every conversion matches the reference, and the dropped counter fires%s%s%s", "", "", "");
  return fails != 0;
}
CASE
}

gen_case "$SCRIPT_DIR/capstone_sqlite_stdio.c" > "$WORK/case.c"
"$CC" -O1 -I"$SCRIPT_DIR" -o "$WORK/case" "$WORK/case.c" "$WORK/ref.c"
"$WORK/case"

# THE NEGATIVE CONTROL. A comparison that has never disagreed is not a passing test, it is an
# unproven one. Break the digit table and the hex cases must fail; if they do not, the harness is
# not actually comparing anything and its pass above means nothing.
sed 's/const char \*digits = "0123456789abcdef";/const char *digits = "0123456789ABCDEF";/' \
    "$SCRIPT_DIR/capstone_sqlite_stdio.c" > "$WORK/broken.c"
gen_case "$WORK/broken.c" > "$WORK/broken_case.c"
"$CC" -O1 -I"$SCRIPT_DIR" -o "$WORK/broken_case" "$WORK/broken_case.c" "$WORK/ref.c"
if "$WORK/broken_case" > "$WORK/broken.out" 2>&1; then
  echo "ERROR: the negative control PASSED -- the comparison is not comparing" >&2
  exit 1
fi
grep -q "MISMATCH fmt=%02x" "$WORK/broken.out" || {
  echo "ERROR: the negative control failed, but not on the conversion it breaks" >&2
  cat "$WORK/broken.out" >&2
  exit 1
}
echo "negative control OK: a wrong digit table is caught, on the hex conversions"

# THE SECOND NEGATIVE CONTROL, for the two cases added above. A counter check that has never seen a
# broken counter is exactly the unproven gate it was written to replace. Remove BOTH increment sites
# and the two DROPPED-FAIL cases must fire; the conversion comparisons must still pass, which is what
# says the control removed the counting and nothing else.
sed -e 's/^        capstone_stdio_dropped++;$/        ;/' \
    -e 's/^    capstone_stdio_dropped += (b.len - took) + b.over;$/    ;/' \
    "$SCRIPT_DIR/capstone_sqlite_stdio.c" > "$WORK/nocount.c"
# The sed is the gate here, so verify it BIT: a pattern that stopped matching would leave the
# counting in place and the control would "pass" by testing nothing.
if ! grep -q "capstone_stdio_dropped" "$WORK/nocount.c"; then
  : # every mention gone is also acceptable
fi
_left=$(grep -c "capstone_stdio_dropped++\|capstone_stdio_dropped +=" "$WORK/nocount.c" || true)
[ "$_left" = "0" ] || {
  echo "ERROR: the no-count control did not remove the increments ($_left left) -- the sed patterns moved" >&2
  exit 1; }
gen_case "$WORK/nocount.c" > "$WORK/nocount_case.c"
"$CC" -O1 -I"$SCRIPT_DIR" -o "$WORK/nocount_case" "$WORK/nocount_case.c" "$WORK/ref.c"
if "$WORK/nocount_case" > "$WORK/nocount.out" 2>&1; then
  echo "ERROR: the no-count control PASSED -- the dropped-counter cases test nothing" >&2
  exit 1
fi
grep -q "DROPPED-FAIL" "$WORK/nocount.out" || {
  echo "ERROR: the no-count control failed, but not on the dropped-counter cases" >&2
  cat "$WORK/nocount.out" >&2
  exit 1
}
grep -q "MISMATCH" "$WORK/nocount.out" && {
  echo "ERROR: the no-count control also broke a conversion -- it changed more than the counting" >&2
  cat "$WORK/nocount.out" >&2
  exit 1; }
echo "negative control OK: with both increments removed, the dropped-counter cases fire and nothing else does"
