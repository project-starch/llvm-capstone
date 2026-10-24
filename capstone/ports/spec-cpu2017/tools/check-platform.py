#!/usr/bin/env python3
"""Check a platform directory against manifest.lock.json's platform section; print what matched.
usage: check-platform.py MANIFEST PLATFORM_DIR"""
import hashlib, json, os, sys

m = json.load(open(sys.argv[1]))["platform"]; root = sys.argv[2]
bad = 0
for rel, want in m["files"].items():
    path = os.path.join(root, rel)
    got = hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.isfile(path) else "missing"
    print(f"{'ok' if got == want else 'MISMATCH'} {rel} {got}")
    bad += got != want
sys.exit(1 if bad else 0)
