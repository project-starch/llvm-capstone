#!/bin/bash
# run-cheribsd.sh OUT: the cheribsd-revocation arm, on a running CheriBSD purecap guest.
#
#   CHERI_PERL=<build.sh's perl> PERL_LIBS="<lib dir> ..." GUEST_PORT=<port> GUEST_KEY=<key> \
#     [ONLY=04,05] [CASE_BUDGET=600] bash run-cheribsd.sh OUT
#
# The interpreter is ports/perl/cheribsd/build.sh's, dynamic, with revocation as the platform ships
# it: the runner refuses a guest whose security.cheri.runtime_revocation_default is not 1, and
# records the whole security.cheri subtree before and after. PERL_LIBS are merged into the guest's
# library in order, the first copy of a file winning: the build's own lib (its Config.pm) first,
# then the native reference's, for the pure-Perl modules perl-cross does not install.
#
# Two preloads go in front of every process (capstone/bug-corpora/tools/cheribsd/build-helpers.sh):
# sicode.so names the fault that ended it (SIGPROT si_code), quarantine-probe.so reports whether
# each free entered the revocation quarantine and whether memory came back while still in it.
# The preload also loads into `timeout`, which allocates nothing and reports last, so only a probe
# line from a process that mapped the shadow is kept.
#
# Every case runs twice: with revocation as the platform ships it (on), and with it switched off
# for that one process (_RUNTIME_REVOCATION_DISABLE=1). The pair says whether revocation made the
# difference: a tag fault the off run shows too, at the same address, is not revocation's. Both
# run with ASLR off for the process (proccontrol), so the addresses are comparable.
#
# Controls, each refusing the run when it fails: the self-test (run under the same preloads) must
# report PROT_CHERI_BOUNDS for a read past an allocation and PROT_CHERI_TAG for a read through a
# freed pointer after a forced revocation, and with revocation switched off that read must not
# fault (the knob is live); the interpreter must evaluate 6*7 with the probe reporting; the
# harness must load.
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CORPUS=$(cd "$HERE/../.." && pwd)
PERL=${CHERI_PERL:?set CHERI_PERL to the perl ports/perl/cheribsd/build.sh produced}
LIBS=${PERL_LIBS:?set PERL_LIBS to the library directories, the build own first}
OUT=${1:?usage: run-cheribsd.sh OUT}
BUDGET=${CASE_BUDGET:-600}
ONLY=${ONLY:-}
SICODE=/root/sicode.so SELFTEST=/root/sicode-selftest QPROBE=/root/quarantine-probe.so
K="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=120 ${GUEST_KEY:+-i $GUEST_KEY}"
G() { ssh $K -p "${GUEST_PORT:?}" root@127.0.0.1 "$@" 2>/dev/null; }
in_subset() { [ -z "$ONLY" ] && return 0; case ",$ONLY," in *",${1%%_*},"*) return 0;; *) return 1;; esac; }
mkdir -p "$OUT/raw"
G true || { echo "guest on port $GUEST_PORT is not reachable" >&2; exit 2; }

G 'sysctl security.cheri' > "$OUT/sysctl-before.txt"
rev=$(awk -F': ' '/runtime_revocation_default/{print $2}' "$OUT/sysctl-before.txt")
[[ $rev == 1 ]] || { echo "REFUSING: runtime_revocation_default is '$rev', not 1" >&2; exit 2; }

# ---- stage the interpreter, its library, the helpers' presence ----------
STAGE=$(mktemp -d)
mkdir -p "$STAGE/lib"
for d in $LIBS; do
  [[ -d $d ]] || { echo "no library dir $d" >&2; exit 2; }
  (cd "$d" && find . -type f \( -name '*.pm' -o -name '*.pl' -o -name '*.pod' -o -name '*.ix' -o -name '*.al' \) -print0) |
    while IFS= read -r -d '' f; do
      [[ -e $STAGE/lib/$f ]] || { mkdir -p "$STAGE/lib/$(dirname "$f")"; cp "$d/$f" "$STAGE/lib/$f"; }
    done
done
for must in strict.pm warnings.pm XSLoader.pm Data/Dumper.pm PerlIO/scalar.pm Config.pm; do
  [[ -f $STAGE/lib/$must ]] || { echo "REFUSING: the staged library has no $must" >&2; exit 2; }
done
cp "$PERL" "$STAGE/perl"
G 'rm -rf /root/perl && mkdir -p /root/perl'
tar -C "$STAGE" -cf - perl lib | G 'tar -C /root/perl -xf -'
rm -rf "$STAGE"
hostsha=$(sha256sum "$PERL" | cut -d' ' -f1)
guestsha=$(G 'sha256sum /root/perl/perl' | cut -d' ' -f1)
[[ $hostsha == "$guestsha" ]] || { echo "REFUSING: guest perl ${guestsha:0:16} is not host ${hostsha:0:16}" >&2; exit 2; }
for f in $SICODE $SELFTEST $QPROBE; do G "test -f $f" || { echo "no $f in the guest" >&2; exit 2; }; done
printf 'arm\tcheribsd-revocation\nperl_sha256\t%s\nrevocation_default\t%s\nstarted_utc\t%s\n' \
  "$hostsha" "$rev" "$(date -u +%FT%TZ)" > "$OUT/run.meta"

# ---- controls ------------------------------------------------------------
PRE="LD_PRELOAD=$SICODE:$QPROBE"
sb=$(G "env $PRE timeout 60 $SELFTEST bounds 2>&1" | grep -a -o 'si_code=[0-9]* ([A-Z_]*)' | head -1)
sr=$(G "env $PRE timeout 60 $SELFTEST revoked 2>&1" | grep -a -o 'si_code=[0-9]* ([A-Z_]*)' | head -1)
[[ $sb == "si_code=1 (PROT_CHERI_BOUNDS)" && $sr == "si_code=2 (PROT_CHERI_TAG)" ]] || {
  echo "REFUSING: self-test gave bounds '${sb:-nothing}', revoked '${sr:-nothing}'" >&2; exit 2; }
OFF="_RUNTIME_REVOCATION_DISABLE=1"
so=$(G "env $OFF $PRE timeout 60 $SELFTEST revoked 2>&1")
printf '%s' "$so" | grep -aq 'SELFTEST revocation is off' && printf '%s' "$so" | grep -aq 'SELFTEST no fault' \
  && ! printf '%s' "$so" | grep -aq SICODE || {
  echo "REFUSING: with $OFF the revoked read still gave: $(printf '%s' "$so" | tail -2)" >&2; exit 2; }
ev=$(G "cd /root/perl && env PERL5LIB=/root/perl/lib $PRE timeout 300 ./perl -e 'print 6*7, qq(\n)' 2>&1")
q=$(printf '%s\n' "$ev" | grep -a '^QUARANTINE shadow=mapped' | tail -1)
printf '%s\n' "$ev" | grep -aqx 42 && [[ -n $q ]] || {
  echo "REFUSING: the eval control gave: $(printf '%s' "$ev" | tail -3)" >&2; exit 2; }
G 'mkdir -p /root/perlcases/_shim'
G 'cat > /root/perlcases/_shim/shim.pl' < "$CORPUS/harness/shim.pl"
sh=$(G "cd /root/perlcases/_shim && env PERL5LIB=/root/perl/lib:. timeout 300 /root/perl/perl -e 'require q(shim.pl); ok(1); print qq(SHIMOK\n)' 2>&1")
printf '%s' "$sh" | grep -aq SHIMOK || { echo "REFUSING: the harness did not load: $(printf '%s' "$sh" | tail -3)" >&2; exit 2; }
printf 'sicode_control\tbounds %s, revoked %s, revoked with revocation off: no fault\neval_control\t42 with %s\nshim_control\tSHIMOK\n' \
  "$sb" "$sr" "$q" >> "$OUT/run.meta"
echo "controls: bounds $sb, revoked $sr, off: no fault; eval 42; shim ok; $q"

# ---- cases, revocation on then off --------------------------------------
printf 'case\trevocation\trc\tsignal\tsi_code\tharness\tquarantine\n' > "$OUT/verdicts.tsv"
for d in "$CORPUS"/[0-9][0-9]_*/; do
  c=$(basename "$d"); in_subset "$c" || continue
  G "rm -rf /root/perlcases/$c && mkdir -p /root/perlcases/$c"
  G "cat > /root/perlcases/$c/trigger.pl" < "$d/trigger.pl"
  G "cat > /root/perlcases/$c/shim.pl" < "$CORPUS/harness/shim.pl"
  for mode in on off; do
    knob=; [[ $mode == off ]] && knob=$OFF
    # ASLR off for the process, so the two runs' fault addresses can be compared.
    out=$(G "cd /root/perlcases/$c && env PERL5LIB=/root/perl/lib:. $knob $PRE timeout $BUDGET proccontrol -m aslr -s disable /root/perl/perl trigger.pl 2>&1; echo RC=\$?")
    out=$(printf '%s' "$out" | tr -d '\000' | LC_ALL=C tr -c '\11\12\15\40-\176' '?')
    printf '%s\n' "$out" > "$OUT/raw/$c-$mode.out"
    rc=$(printf '%s' "$out" | grep -a -oE '^RC=[0-9]+' | tail -1 | cut -d= -f2)
    sig=$(printf '%s' "$out" | grep -a -oE 'SICODE signal=[0-9]+' | head -1 | cut -d= -f2)
    code=$(printf '%s' "$out" | grep -a -oE 'si_code=[0-9]+ \([A-Z_]+\) addr=0x[0-9a-f]+' | head -1)
    harness=$(printf '%s' "$out" | grep -a -m1 -oE '^\[(PASS|FAIL)\].{0,100}|(panic: |Attempt to ).{0,80}' | head -1)
    qline=$(printf '%s' "$out" | grep -a -o 'QUARANTINE shadow=mapped.*' | paste -sd'|' -)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$c" "$mode" "${rc:-?}" "${sig:-}" "${code:-}" "${harness:-}" "${qline:-}" >> "$OUT/verdicts.tsv"
    printf '  %-44s %-3s rc=%-4s %-28s %s\n' "${c:0:44}" "$mode" "${rc:-?}" "${code:-}" "${harness:0:46}"
  done
done

G 'sysctl security.cheri' > "$OUT/sysctl-after.txt"
diff -q "$OUT/sysctl-before.txt" "$OUT/sysctl-after.txt" >/dev/null \
  && echo "security.cheri unchanged across the run" || echo "WARNING: security.cheri changed during the run" >&2
printf 'ended_utc\t%s\n' "$(date -u +%FT%TZ)" >> "$OUT/run.meta"
