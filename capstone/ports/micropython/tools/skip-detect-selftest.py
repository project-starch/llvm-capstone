#!/usr/bin/env python3
"""Negative test for is_target_skip in run-resumable-suite.py. Needs no board and no QEMU.

A target skip and a wrong answer are both a returned word with the raised bit set, and the only
thing that separates them is the captured output. If this predicate widens by accident, a genuine
failure disappears into the skip column, which is the one direction a test report must never move.
"""
import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("suite", HERE / "run-resumable-suite.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)

TRACE = b"Traceback (most recent call last):\n  File \"<stdin>\", line 6, in <module>\nSystemExit:"
CASES = [
    (True,  b"SKIP\n" + TRACE,                      "the plain convention"),
    (True,  b"SKIP-TOO-LARGE\n" + TRACE,            "the memory convention"),
    (True,  b"SKIP\n" + TRACE + b"\n",              "a trailing newline"),
    (False, b"SKIP\n",                              "SKIP printed without raising"),
    (False, b"SKIPPED\n" + TRACE,                   "a word that merely starts with SKIP"),
    (False, b"some output\nSKIP\n" + TRACE,         "SKIP after real output"),
    (False, b"SKIP\n" + TRACE + b" 1",              "SystemExit with an exit code"),
    (False, b"SKIP\nTraceback (most recent call last):\nValueError: x", "a different exception"),
    (False, b"",                                    "no output at all"),
    (False, None,                                   "nothing captured"),
]

fail = 0
for want, output, label in CASES:
    got = suite.is_target_skip(output)
    if bool(got) == want:
        print(f"  ok    {label:<38s} {got}")
    else:
        print(f"  FAIL  {label:<38s} got {got}, wanted {want}")
        fail = 1

print("skip-detect-selftest: " + ("every case behaved" if not fail else "FAILED"))
sys.exit(fail)
