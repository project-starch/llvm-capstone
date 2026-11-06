# Invoke as: sh semantics.sh ./child
# Uses only files within the caller's cwd grant and the exact shell image.
set -e
child=$1
fail() { echo "HUSH FAIL: $1"; exit 1; }
test "$($child args 'a b' '' '*')" = '[a b]
[]
[*]' || fail arguments
test "$(printf 'a\nb\n' | cat)" = 'a
b' || fail pipeline
test "$($child emit 100000 | $child count)" = 100000 || fail blocking-pipe
test "$(printf 'one\ntwo\n\n')" = 'one
two' || fail substitution-newlines
outer=parent
test "$(outer=child; printf '%s' "$outer")" = child || fail subshell-value
test "$outer" = parent || fail subshell-isolation
f() { printf '%s:%s' "$1" "$outer"; }
test "$(f 'a b' | cat)" = 'a b:parent' || fail function-serialization
test "$(CAPSTONE_HUSH_TEST=value $child env CAPSTONE_HUSH_TEST)" = value || fail environment
test "${CAPSTONE_HUSH_TEST-unset}" = unset || fail environment-restore
set +e
$child exit 17
status=$?
set -e
test "$status" = 17 || fail exit-status
set +e
capstone_hush_missing_command 2> hush-test.err
status=$?
set -e
test "$status" = 127 || fail missing-status
false | true
if true | false; then fail pipeline-status; fi
printf first > hush-test.out
test "$(cat hush-test.out)" = first || fail redirect
printf second 3> hush-test.fd 1>&3
test "$(cat hush-test.fd)" = second || fail descriptor-redirect
test "$($child fds)" = '0 1 2 ' || fail descriptor-leak
test "$(cat <<EOF
hello $((2+3))
EOF
)" = 'hello 5' || fail heredoc-expansion
test "$(cat <<'EOF'
literal $outer
EOF
)" = 'literal $outer' || fail heredoc-quoted
test "$(for n in a b c; do printf '%s' "$n"; done)" = abc || fail loop
case "$outer" in par*) :;; *) fail case;; esac
set +e
$child exit 23 & first=$!
$child exit 24 & second=$!
$child exit 0
wait "$first"
first_status=$?
wait "$second"
second_status=$?
set -e
test "$first_status" = 23 || fail background-status-first
test "$second_status" = 24 || fail background-status-second
echo 'HUSH PASS: shell semantics'
