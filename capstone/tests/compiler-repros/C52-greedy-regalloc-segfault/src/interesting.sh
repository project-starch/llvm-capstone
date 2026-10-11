#!/usr/bin/env bash
# Interesting = llc dies by SIGSEGV inside the Greedy register allocator. Both
# conditions: a reduction that drifts to another crash, or to a clean exit, is not this bug.
out=$(timeout 120 ${LLC:-llc} -O1 "$1" -o /dev/null 2>&1)
rc=$?
[[ $rc -eq 139 ]] && grep -q "Running pass 'Greedy Register Allocator'" <<<"$out"
