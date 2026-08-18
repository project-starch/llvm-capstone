#!/usr/bin/env bash
# MANDATORY pre-commit / pre-push scan: personal names and secrets.
#
# Deliberately a SCRIPT and not a subagent. This is a release gate on a permanent,
# absolute repository rule -- it must be deterministic and reproducible, and it must fail
# the same way every time. An agent that is usually right is the wrong tool for a rule
# that has already been violated once (2026-07-25: a name in a commit SUBJECT, which had
# to be amended and force-pushed after the fact).
#
#   usage:
#     bash capstone/tests/precommit-scan.sh                    # scan staged diff
#     bash capstone/tests/precommit-scan.sh --msg <file>       # also scan a commit message
#     bash capstone/tests/precommit-scan.sh --range A..B       # scan a commit range
#
# Exit 0 = clean. Exit 1 = BLOCKED, do not commit or push.
#
# WHY THE NAME LIST IS NOT IN THIS FILE. Hardcoding the collaborator/PI names here would
# itself write those names into a committed file -- exactly the thing the rule forbids.
# So the denylist lives OUTSIDE the repo, one name or pattern per line:
#
#     ~/.claude-c/secrets/name-denylist.txt      (mode 600, never committed)
#
# If that file is absent the script still runs its name-independent heuristics and warns
# that the exact-name check was skipped. Keep the file populated.
set -uo pipefail

DENYLIST="${CAPSTONE_NAME_DENYLIST:-$HOME/.claude-c/secrets/name-denylist.txt}"
MSG_FILE=""
RANGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --msg)   MSG_FILE="$2"; shift 2 ;;
    --range) RANGE="$2";    shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "not a git repo" >&2; exit 2; }
cd "$REPO_ROOT" || exit 2

# ---- gather the text under scrutiny -----------------------------------------------
TMP=$(mktemp) ; trap 'rm -f "$TMP"' EXIT
# THE SCANNER DOES NOT SCAN ITSELF. Its own detector patterns necessarily contain the
# very strings it looks for ('ngrok', 'token', 'user@host'), so including this file makes
# every commit that touches it fail on its own regexes. Excluding it is not a weakened
# pattern -- a detector's rule list is not a violation. Announced below, never silent.
SELF='capstone/tests/precommit-scan.sh'
if [[ -n "$RANGE" ]]; then
  git log --format='%H%n%an%n%ae%n%s%n%b' "$RANGE" >> "$TMP" 2>/dev/null
  git diff "$RANGE" -- . ":(exclude)$SELF"         >> "$TMP" 2>/dev/null
else
  git diff --cached -- . ":(exclude)$SELF"         >> "$TMP" 2>/dev/null
  # ALSO the unstaged and the UNTRACKED. Without these the scan is only as good as the
  # user's staging discipline, and it silently was not: running it BEFORE `git add` scanned
  # the commit message and nothing else, then printed CLEAN. A brand-new file -- exactly the
  # kind that carries a fresh name or a pasted console URL -- appears in NO diff until it is
  # staged, so the most dangerous content was the least likely to be seen. Caught 2026-08-18
  # when a new plans/ document passed a scan that had never read a byte of it.
  git diff -- . ":(exclude)$SELF"                  >> "$TMP" 2>/dev/null
  while IFS= read -r -d '' f; do
    [[ "$f" == "$SELF" ]] && continue
    printf '=== untracked: %s ===\n' "$f" >> "$TMP"
    cat -- "$f" >> "$TMP" 2>/dev/null
  done < <(git ls-files --others --exclude-standard -z 2>/dev/null)
fi
if git diff --cached --name-only 2>/dev/null | grep -qxF "$SELF"; then
  echo "NOTE: $SELF is staged and was EXCLUDED from its own scan. Review it by eye."
  echo
fi
[[ -n "$MSG_FILE" && -f "$MSG_FILE" ]] && cat "$MSG_FILE" >> "$TMP"

if [[ ! -s "$TMP" ]]; then
  # A gate on an absolute rule must not pass silently when it inspected nothing. This
  # printed "nothing staged to scan" and exited 0 for every message-only commit
  # (git commit --allow-empty, used heavily for findings), so those commit MESSAGES were
  # never scanned for names -- the exact content most likely to name a person. Warn loudly
  # and say what to do; still exit 0 so it cannot block a legitimately empty scan.
  echo "precommit-scan: WARNING -- nothing to scan (no staged diff, no --msg file)."
  echo "  If this is a message-only commit, the MESSAGE WAS NOT SCANNED."
  echo "  Scan it explicitly:  bash capstone/tests/precommit-scan.sh --msg <msgfile>"
  exit 0
fi

FAIL=0
hit() {  # $1 = label, $2 = matches
  FAIL=1
  echo "=========================================================="
  echo "BLOCKED: $1"
  echo "=========================================================="
  echo "$2" | head -25
  echo
}

# ---- 1. exact names from the out-of-repo denylist ---------------------------------
if [[ -f "$DENYLIST" ]]; then
  PATTERNS=$(grep -vE '^\s*(#|$)' "$DENYLIST" || true)
  if [[ -n "$PATTERNS" ]]; then
    M=$(printf '%s\n' "$PATTERNS" | while IFS= read -r p; do
          [[ -z "$p" ]] && continue
          grep -inF -- "$p" "$TMP" | sed "s/^/  [name: ${p}] /" || true
        done)
    [[ -n "$M" ]] && hit "a personal name from the denylist appears in the commit message or staged diff" "$M"
  fi
else
  echo "WARNING: no denylist at $DENYLIST -- exact-name check SKIPPED."
  echo "         Create it (mode 600, outside the repo), one name per line."
  echo
fi

# ---- 2. name-independent heuristics ------------------------------------------------
# Email addresses. Excludes dependency/citation URLs, which are legitimate.
M=$(grep -inE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' "$TMP" \
    | grep -viE 'https?://|\.gitmodules|noreply|example\.(com|org)' || true)
[[ -n "$M" ]] && hit "an email address appears in the commit message or staged diff" "$M"

# Attribution trailers that this project forbids outright.
M=$(grep -inE '^\+?\s*(Co-Authored-By|Signed-off-by|Reported-by|Reviewed-by|Tested-by):' "$TMP" || true)
[[ -n "$M" ]] && hit "an attribution trailer is present (forbidden by project rules)" "$M"

# Personal build hostnames, e.g. root@<name>.
# `user@host` / `root@<name>` are documented METASYNTACTIC placeholders in this repo's own
# rule text, exactly like <FPGA-CONSOLE-URL>. Excluding the placeholder forms is not a
# weakened pattern; a real hostname still trips this.
M=$(grep -inE '\b(root|user|build)@[A-Za-z][A-Za-z0-9._-]{2,}' "$TMP" \
    | grep -viE 'root@localhost|@example|@buildroot|@qemu|\b(user|root|build)@(host|hostname|<)|root@<name>' || true)
[[ -n "$M" ]] && hit "a personal build hostname (user@host) appears" "$M"

# Attribution phrasing that usually precedes a real name.
# CASE-SENSITIVE on purpose: the whole signal is the CAPITAL letter that starts a name.
# With grep -i, [A-Z] also matches lowercase and this check silently matches everything
# (it fired on "Task for the external collaborator", which is the correct ROLE wording).
M=$(grep -nE "\b([Tt]hanks to|[Pp]er|[Aa]sked by|[Aa]ccording to|[Ss]uggested by|[Tt]ask for|[Aa]ssigned to|[Mm]sg (to|from)|[Mm]essage (to|from)) +[A-Z][a-z]+('s)?" "$TMP" \
    | grep -vE '\b[Pp]er +(The|A|Our|Its|This|Each|Run|Entry|Rung|Domain|Global|Byte|Call|Cycle|RTL|QEMU|SQLite)\b' || true)
[[ -n "$M" ]] && hit "attribution phrasing followed by a capitalised word -- check it is a ROLE, not a name" "$M"

# ---- 3. secrets: the FPGA console URL / token --------------------------------------
M=$(grep -inE 'ngrok|trycloudflare|loca\.lt|serveo|\btoken\s*[=:]\s*\S|X-Auth|Authorization:' "$TMP" || true)
[[ -n "$M" ]] && hit "a token or tunnel URL appears -- the FPGA console URL/token must NEVER be committed" "$M"

# Any URL sitting next to fpga/board/console wording.
M=$(grep -inE '(fpga|board|console)[^\n]{0,40}https?://' "$TMP" \
    | grep -viF '<FPGA-CONSOLE-URL>' || true)
[[ -n "$M" ]] && hit "a board/FPGA URL appears -- use the placeholder <FPGA-CONSOLE-URL>" "$M"

# If the real URL is available locally, check for it verbatim. Never printed.
URL_FILE="$HOME/.claude-c/secrets/fpga-console-url"
if [[ -f "$URL_FILE" ]]; then
  HOSTPART=$(sed -E 's#^https?://##; s#/.*##' "$URL_FILE" | tr -d '[:space:]')
  if [[ -n "$HOSTPART" ]] && grep -qiF -- "$HOSTPART" "$TMP"; then
    hit "the actual FPGA console host appears in the staged content (value withheld)" \
        "  <redacted -- matched the host in $URL_FILE>"
  fi
fi

# ---- 4. files that must never be committed -----------------------------------------
if [[ -z "$RANGE" ]]; then
  M=$(git diff --cached --name-only | grep -iE '_DEBUG_CHECKPOINT\.md$|session[-_]notes|\.uart\.txt$' || true)
  [[ -n "$M" ]] && hit "debug/report files are staged (never commit these)" "$M"
fi

echo "=========================================================="
if [[ $FAIL -eq 0 ]]; then
  echo "precommit-scan: CLEAN"
  exit 0
fi
echo "precommit-scan: BLOCKED -- fix the above before committing or pushing."
echo "If a hit is a false positive (a role word, a dependency URL, a citation),"
echo "confirm it by eye and re-run; do not weaken the patterns to make it pass."
exit 1
