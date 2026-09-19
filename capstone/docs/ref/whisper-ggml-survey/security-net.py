#!/usr/bin/env python3
"""The security-fix survey the ggml work never did.

The earlier pass searched commit subjects for TEMPORAL words -- use-after-free,
dangling, double free. That is the right net for the corpus's question and the
wrong net for finding out whether ggml has a security history at all: fixes that
arrive from a report are described as "heap-buffer-overflow", "OOB read",
"underflow", or just "Merge pull request from GHSA-...".

So: cast a MEMORY-SAFETY net, then classify each hit as spatial or temporal, and
say which layer it lives in. A spatial bug is not a corpus case -- the corpus is
about temporal safety -- but it has to be seen before it can be ruled out.
"""
import os
import re
import subprocess

REPOS = {
    "llama.cpp": "/tmp/claude-1000/corpus-research/llamacpp.git",
    "whisper.cpp": "/tmp/claude-1000/corpus-research/whispercpp.git",
}

# Deliberately wide: anything that smells like a memory-safety or security fix.
NET = [
    "CVE-", "GHSA-", "security", "overflow", "underflow", "out of bounds",
    "out-of-bounds", "OOB", "heap-buffer", "buffer overflow", "sanitizer",
    "asan", "use.after.free", "use-after-free", "double free", "double-free",
    "dangling", "uninitialized", "uninit", "null deref", "null pointer",
    "integer overflow", "bounds check", "validate", "malicious", "untrusted",
]

TEMPORAL = re.compile(
    r"use.after.free|use-after-free|double.free|double-free|dangling|"
    r"after.*(free|destroy|release)|stale|lifetime", re.I)
SPATIAL = re.compile(
    r"overflow|underflow|out.of.bounds|oob|bounds|heap-buffer|index", re.I)

for name, repo in REPOS.items():
    args = ["log", "--format=%x01%h%x02%s", "-i"]
    for term in NET:
        args += [f"--grep={term}"]
    out = subprocess.run(["git", "-C", repo, *args],
                         capture_output=True, text=True).stdout
    rows = []
    for rec in out.split("\x01"):
        if not rec.strip():
            continue
        h, s = (rec.split("\x02") + [""])[:2]
        rows.append((h.strip(), s.strip()))

    print("=" * 78)
    print(f"{name}: {len(rows)} commits match a memory-safety / security net")
    print("=" * 78)
    temporal, spatial, other = [], [], []
    for h, s in rows:
        files = [f for f in subprocess.run(
            ["git", "-C", repo, "show", "--format=", "--name-only", h],
            capture_output=True, text=True).stdout.split("\n") if f]
        bucket = temporal if TEMPORAL.search(s) else (
            spatial if SPATIAL.search(s) else other)
        bucket.append((h, s, files))

    for label, rows2 in (("TEMPORAL (what the corpus needs)", temporal),
                         ("SPATIAL (bounds; not a corpus case)", spatial),
                         ("other / unclear", other)):
        print(f"\n--- {label}: {len(rows2)} ---")
        for h, s, files in rows2[:24]:
            ggml = [f for f in files if f.startswith("ggml/")]
            tag = "  <-- touches ggml/" if ggml else ""
            print(f"    {h}  {s[:84]}{tag}")
            if ggml:
                print(f"          {ggml[:4]}")
    print()
