#!/bin/bash
# Run the sql-repros cases in a booted CheriBSD purecap guest.
#
#   run-cheribsd.sh [out dir]
#
# The guest is expected to be up on PG_CHERI_PORT (default 10086) with the
# purecap PostgreSQL already deployed under /usr/local/pgsql and a cluster
# initdb'd at PG_CHERI_BASE (default /home/pg/data). This script does not boot
# the guest, deploy the tree or run initdb: each is slow, each is shared, and
# doing them implicitly is how a run ends up measuring a different build from
# the one it names.
#
# ONE CONNECTION AT A TIME AND NO RETRY LOOPS. The guest is shared with other
# lanes.
#
# VERDICTS:
#   detected         exit 162, SIGPROT(34) -- the capability check reporting.
#   port-defect      SIGBUS(10)/SIGSEGV(11). An alignment or representability
#                    fault is the port being wrong, not the defect being
#                    caught; scoring it as a detection would inflate the arm.
#   not-applicable   the case needs an extension this build cannot create.
#                    Outside the denominator, not a verdict about the arm.
#   control-failure  an extension the preflight created failed here, so the
#                    trigger did not reach the defect.
#   not-runnable     the case declares a harness_limit.
#   differential     the case's own EXPECT directive fired.
#   silent           ran to the prompt, no fault, no directive.
#   BADRUN           no stand-alone backend prompt: nothing executed.
set -uo pipefail
C=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)
CORPUS=$(cd "$C/.." && pwd)
PORT=${PG_CHERI_PORT:-10086}
BASE=${PG_CHERI_BASE:-/home/pg/data}
PGBIN=${PG_CHERI_PGBIN:-/usr/local/pgsql/bin}
GATE=${PG_CHERI_GATE:-02}
STAMP=$(date -u +%Y%m%d-%H%M%S)
OUT=${1:-$CORPUS/results/cheribsd-revocation-$STAMP}
PY=${PYTHON:-/home/zephyr/arms/cpython/bin/python3}

K="-i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o StrictHostKeyChecking=no"
K="$K -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10"
G() { ssh -n $K -p "$PORT" root@localhost "$@" 2>/dev/null; }
mkdir -p "$OUT"

CASES=$(cd "$CORPUS" && ls -d [0-9][0-9]_*/ 2>/dev/null | tr -d / | sort)
[ -n "$CASES" ] || { echo "no cases under $CORPUS" >&2; exit 2; }
# ONLY=03,07 runs just those, and the subset is recorded: a rerun after a fix
# must not look like a full pass that happened to score fewer cases.
ONLY=${ONLY:-}
if [ -n "$ONLY" ]; then
  keep=""
  for tag in $CASES; do
    case ",$ONLY," in *",${tag%%_*},"*) keep="$keep $tag" ;; esac
  done
  CASES=$keep
  [ -n "$CASES" ] || { echo "ONLY=$ONLY matched no case" >&2; exit 2; }
fi

KERN=$(G 'uname -r')
[ -n "$KERN" ] || { echo "no answer from the guest on port $PORT" >&2; exit 2; }
REVD=$(G 'sysctl -n security.cheri.runtime_revocation_default')
REVF=$(G 'sysctl -n security.cheri.runtime_revocation_every_free_default')
echo "kernel=$KERN revocation_default=$REVD every_free_default=$REVF"
G "test -x $PGBIN/postgres" >/dev/null || {
  echo "no $PGBIN/postgres in the guest; deploy the purecap tree first" >&2; exit 2; }
G "test -d $BASE" >/dev/null || {
  echo "no cluster at $BASE in the guest; run initdb first" >&2; exit 2; }

# WHICH EXTENSIONS THIS BUILD CAN CREATE, asked of the build itself by creating
# them. Asking what was staged would test the deploy script's intention. A case
# needing one that is absent is not-applicable and leaves the denominator: on
# 2026-10-05 the Capstone arms scored a case `silent` from an image with no
# ltree in it, because CREATE EXTENSION failed, the trigger's real statement
# never ran, and the backend prompt still appeared.
WANT=$(cd "$CORPUS" && cat */trigger.sql 2>/dev/null \
       | grep -oiE 'CREATE[[:space:]]+EXTENSION[[:space:]]+(IF[[:space:]]+NOT[[:space:]]+EXISTS[[:space:]]+)?[a-z_]+' \
       | awk '{print tolower($NF)}' | sort -u)
echo "extensions the corpus needs: $(echo $WANT | tr '\n' ' ')"
PRE=$OUT/preflight.sql
{ echo "SELECT 1 AS backend_runs_sql;"; for e in $WANT; do echo "CREATE EXTENSION $e;"; done; } > "$PRE"
scp $K -P "$PORT" "$PRE" root@localhost:/tmp/preflight.sql >/dev/null 2>&1
preout=$(G "su -m pg -c 'w=\$(mktemp -d /tmp/pre-XXXXXX); cp -R $BASE \$w/data; \
  $PGBIN/postgres --single -D \$w/data postgres < /tmp/preflight.sql 2>&1; rm -rf \$w'")
printf '%s\n' "$preout" > "$OUT/preflight.out"
printf '%s' "$preout" | grep -q 'backend>' || {
  echo "POSITIVE CONTROL FAILED: no stand-alone backend prompt in the guest" >&2; exit 2; }
printf '%s' "$preout" | grep -q 'backend_runs_sql' || {
  echo "POSITIVE CONTROL FAILED: the guest did not answer SELECT 1" >&2; exit 2; }
AVAIL=""
for e in $WANT; do
  if printf '%s' "$preout" | grep -qE "ERROR:.*(extension|library).*\"?$e\"?"; then
    echo "  extension $e: NOT AVAILABLE"
  else
    AVAIL="$AVAIL $e"; echo "  extension $e: available"
  fi
done

TSV=$OUT/matrix.tsv
printf 'case\tarm\tverdict\tevidence\n' > "$TSV"
n=0
for tag in $CASES; do
  n=$((n+1))
  d=$CORPUS/$tag
  limit=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('harness_limit','') or '')" "$d/case.json")
  need=$(grep -oiE 'CREATE[[:space:]]+EXTENSION[[:space:]]+(IF[[:space:]]+NOT[[:space:]]+EXISTS[[:space:]]+)?[a-z_]+' \
         "$d/trigger.sql" 2>/dev/null | awk '{print tolower($NF)}' | sort -u)
  miss=""
  for e in $need; do case " $AVAIL " in *" $e "*) ;; *) miss="$miss $e" ;; esac; done

  if [ -n "$limit" ]; then
    v=not-runnable; why="declared by the case: ${limit:0:110}"
    printf '%-52s %-16s %s\n' "$tag" "$v" "${why:0:60}"
    printf '%s\tcheribsd-revocation\t%s\t%s\n' "$tag" "$v" "$why" >> "$TSV"; continue
  fi
  if [ -n "$miss" ]; then
    v=not-applicable
    why="this build cannot create$miss -- outside this arm's denominator, not a verdict about it"
    printf '%-52s %-16s %s\n' "$tag" "$v" "${why:0:60}"
    printf '%s\tcheribsd-revocation\t%s\t%s\n' "$tag" "$v" "$why" >> "$TSV"; continue
  fi

  scp $K -P "$PORT" "$d/trigger.sql" root@localhost:/tmp/trigger.sql >/dev/null 2>&1
  raw=$(G "su -m pg -c 'w=\$(mktemp -d /tmp/pcrun-XXXXXX); cp -R $BASE \$w/data; \
    timeout 900 $PGBIN/postgres --single -D \$w/data postgres < /tmp/trigger.sql 2>&1; \
    echo __EXIT=\$?; rm -rf \$w'")
  ex=$(printf '%s' "$raw" | grep -o '__EXIT=[0-9]*' | tail -1 | cut -d= -f2)
  body=$(printf '%s' "$raw" | grep -v '__EXIT=')
  LOG=$OUT/$tag.out
  printf '%s\n' "$body" > "$LOG"
  # Everything below reads the FILE. Case 07 prints thousands of rows, and
  # passing that through `printf '%s' "$body" | grep` overflowed the argument
  # list, so every check matched nothing and a run that plainly reached the
  # prompt was recorded BADRUN.
  last=$(grep -ohE '(PANIC|FATAL|ERROR|TRAP):.*' "$LOG" | head -1 | cut -c1-70)

  extfail=""
  for e in $need; do
    grep -qE "ERROR:.*extension \"$e\"" "$LOG" && extfail="$extfail $e"
  done

  # A syntax error means the backend was handed something other than the
  # trigger as written; `postgres --single` has no line continuation, and
  # cases 07 and 09 were scored silent that way on 2026-10-06.
  if grep -qE "ERROR:  (syntax error|unterminated)" "$LOG"; then
    v=control-failure
    why="the trigger did not parse as written: $(grep -ohE 'ERROR:  (syntax error|unterminated)[^\n]*' "$LOG" | head -1 | cut -c1-90)"
  elif [ -n "$extfail" ]; then
    v=control-failure
    why="CREATE EXTENSION$extfail failed here although the preflight created it; the trigger did not reach the defect"
  elif [ "${ex:-0}" = 162 ]; then
    v=detected; why="SIGPROT(34) -- a capability violation${last:+; $last}"
  elif [ "${ex:-0}" = 139 ] || [ "${ex:-0}" = 138 ]; then
    v=port-defect
    why="$([ "${ex}" = 139 ] && echo 'SIGSEGV(11)' || echo 'SIGBUS(10)') -- a port defect, not a detection${last:+; $last}"
  elif [ "${ex:-0}" = 124 ]; then
    v=other; why="timed out; not a measurement"
  elif ! grep -q 'backend>' "$LOG"; then
    v=BADRUN; why="no stand-alone backend prompt -- nothing executed"
  else
    v=silent; why="ran to the prompt with no fault${last:+; first message: $last}"
  fi
  printf '%-52s %-16s %s\n' "$tag" "$v" "${why:0:60}"
  printf '%s\tcheribsd-revocation\t%s\t%s\n' "$tag" "$v" "$why" >> "$TSV"
done

gv=$(awk -F'\t' -v g="$GATE" 'NR>1 && substr($1,1,2)==g {print $3}' "$TSV" | head -1)
if [ "${gv:-}" != detected ]; then
  echo >&2
  echo "MECHANISM GATE: case $GATE is ${gv:-absent}, not detected. Every silent row" >&2
  echo "in this run is unqualified; the matrix is written but inputs.json records" >&2
  echo "the gate as failed, and the rows should not be published as negatives." >&2
fi
scored=$(awk -F'\t' 'NR>1 && $3!="control-failure" && $3!="BADRUN" && $3!="other" && $3!="not-applicable" && $3!="not-runnable"' "$TSV" | wc -l)
cat > "$OUT/inputs.json" <<JSON
{
  "arm": "cheribsd-revocation",
  "kernel": "$KERN",
  "revocation_default": "$REVD",
  "revocation_every_free_default": "$REVF",
  "pg_bin": "$PGBIN",
  "base_cluster": "$BASE",
  "started_utc": "$STAMP",
  "extensions_wanted": "$(echo $WANT | tr '\n' ' ')",
  "extensions_available": "$(echo $AVAIL | sed 's/^ //')",
  "mechanism_gate": {
    "case": "$GATE",
    "verdict": "${gv:-absent}",
    "passed": $([ "${gv:-}" = detected ] && echo true || echo false),
    "kind": "a corpus case required to be detected, which is weaker than a purpose-built control: it says the mechanism reported on something in this configuration, not that it would have reported on each silent case"
  },
  "only": "${ONLY:-all}",
  "cases": $n,
  "scored": $scored
}
JSON
echo
echo '=== summary ==='
awk -F'\t' 'NR>1{c[$3]++} END{for(k in c) printf "  %-16s %d\n", k, c[k]}' "$TSV"
echo "scored $scored of $n"
echo "results: $OUT"
