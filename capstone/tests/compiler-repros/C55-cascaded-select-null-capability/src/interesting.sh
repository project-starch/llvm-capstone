#!/usr/bin/env bash
# Interesting = llc aborts on the LiveVariables assertion; any other outcome is not this bug.
out=$(timeout 120 ${LLC:-llc} "$1" -o /dev/null 2>&1)
grep -q 'getVarInfo: not a virtual register' <<<"$out"
