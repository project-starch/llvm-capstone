#!/usr/bin/env python3
"""Relink postgres-lt0 (the recorded link and strip commands, t3-new6/postgres-lt0-dl/image/commands.json[2:]) with
nbtinsert.o replaced, into t3-cw/postgres-<arm>/image. arm ctrl = the unmodified source recompiled (must reproduce the
record), arm diag = the DIAG-BTU patch. The SDK is the record's own (read only)."""
import getpass, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
H = os.environ.get("CAPSTONE_HDD", f"/mnt/hdd/{getpass.getuser()}/capstone"); BASE = f"{H}/t3-new6/postgres-lt0"
OBJ = f"{BASE}/source/domain/postgresql-17.5/src/backend/access/nbtree/nbtinsert.o"
arm = sys.argv[1]; new_obj = f"{H}/pgdiag/nbtree/nbtinsert.{arm}.o"
img = Path(f"{H}/t3-cw/postgres-{arm}/image")
if img.exists(): sys.exit(f"{img} exists")
img.mkdir(parents=True)
cmds = json.load(open(f"{BASE}-dl/image/commands.json"))[2:]
assert sum(w == OBJ for w in cmds[0]) == 1, "nbtinsert.o must appear exactly once in the link"
for c in cmds:
    c = [new_obj if w == OBJ else w.replace(f"{BASE}/image/postgres.dom", str(img / "postgres.dom")) for w in c]
    subprocess.run(["taskset", "-c", "16-31,48-63", *c], check=True)
dom = img / "postgres.dom"
subprocess.run([f"{H}/t3-freeze/llvm/bin/llvm-objcopy", "-O", "binary", dom, img / "loaded.bin"], check=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
print(arm, "nbtinsert", sha(new_obj)[:16], "image", sha(dom)[:16], "loaded", sha(img / "loaded.bin")[:16])
