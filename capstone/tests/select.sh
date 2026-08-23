# One selection rule for every suite: which cases inside it should run.
#
#   CAPSTONE_ONLY=uninit_negative,linear_double   substring match on case names
#   OPT_LEVELS='-O0'                              most suites already honour this
#
# Why it matters: the expensive suites are expensive because they are N probes
# times three optimisation levels. intra-domain-mrev is 8 probes x 3 = 24 boots
# and 1130 s; one probe at one level is about 47 s. A feature usually touches
# one of them.
#
# A pattern that matches no case in the suite is an ERROR. A quietly smaller run
# is the worst outcome available here: it looks exactly like a pass, and the
# whole reason to name a case is to be sure that case ran.
#
# Source it; do not execute it.

CAPSTONE_SELECT_SEEN=""

capstone_selected() {  # $1 = case name -- true if it should run
  CAPSTONE_SELECT_SEEN="$CAPSTONE_SELECT_SEEN $1"
  [ -z "${CAPSTONE_ONLY:-}" ] && return 0
  local want
  for want in ${CAPSTONE_ONLY//,/ }; do
    case "$1" in *"$want"*) return 0 ;; esac
  done
  return 1
}

# Call once before the suite reports. Non-zero if a pattern matched nothing.
capstone_select_verify() {
  [ -z "${CAPSTONE_ONLY:-}" ] && return 0
  local want bad=0
  for want in ${CAPSTONE_ONLY//,/ }; do
    case "$CAPSTONE_SELECT_SEEN" in
      *"$want"*) ;;
      *) echo "CAPSTONE_ONLY names '$want', which matched no case in this suite" >&2
         bad=1 ;;
    esac
  done
  [ "$bad" -eq 0 ] || echo "  cases available:$CAPSTONE_SELECT_SEEN" >&2
  return "$bad"
}

# Did selection actually narrow the run? Suites print this so a log can never be
# mistaken for a full run.
capstone_select_banner() {  # $1 = suite name
  [ -n "${CAPSTONE_ONLY:-}" ] && echo "$1: CAPSTONE_ONLY=$CAPSTONE_ONLY -- PARTIAL RUN"
  [ -n "${OPT_LEVELS:-}" ] && [ "${OPT_LEVELS}" != "-O0 -O1 -O2" ] &&
    echo "$1: OPT_LEVELS=$OPT_LEVELS -- PARTIAL RUN"
  return 0
}
