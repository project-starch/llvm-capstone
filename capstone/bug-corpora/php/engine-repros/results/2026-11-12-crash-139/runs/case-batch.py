#!/usr/bin/env python3
"""case-batch.py IMAGE OUTDIR REPS [--order stock,block,object,exact] [--script NAME=PATH ...] [--no-case]

One guest boot of the PHP image through the port's run-virtual.py: per arm the sanity control, then REPS runs of
CRASH-139's trigger (the corpus file, byte for byte), then REPS runs of each extra script. Every run is its own
capstone-vexec process with the arm's PHP_CAP_MODE in its own environment, exactly the argv run-realphp.run_arm
builds. Nothing is judged here: the record is run-virtual.py's."""
import argparse, importlib.util, json, os, subprocess, sys
from pathlib import Path

R = Path(os.environ["CAPSTONE_REPO_ROOT"]) / "capstone"
spec = importlib.util.spec_from_file_location("rr", R / "ports/php/app/run-realphp.py")
rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
CASE = "41_9a98904ddd0e_crash-139"
MODES = {"stock": 0, "block": 4, "object": 6, "exact": 7}

p = argparse.ArgumentParser()
p.add_argument("image", type=Path); p.add_argument("out", type=Path); p.add_argument("reps", type=int)
p.add_argument("--order", default="stock,block,object,exact")
p.add_argument("--script", action="append", default=[])
p.add_argument("--no-case", action="store_true")
p.add_argument("--kit", type=Path, default=Path(os.environ["KIT"]))
p.add_argument("--run-timeout", type=int, default=40)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
stage = a.out / "stage"; stage.mkdir(exist_ok=True)
(stage / rr.SANITY_SCRIPT).write_text(rr.SANITY_TEXT)
scripts = []
if not a.no_case:
    (stage / f"{CASE}.php").write_bytes((R / "bug-corpora/php/engine-repros" / CASE / "trigger.php").read_bytes())
    scripts.append(("case", f"{CASE}.php"))
for word in a.script:
    name, path = word.split("=", 1)
    (stage / Path(path).name).write_bytes(Path(path).read_bytes())
    scripts.append((name, Path(path).name))
img = a.image.name
runs = []
for arm in a.order.split(","):
    mode = MODES[arm]
    runs.append((f"{arm}__sanity", f"PHP_CAP_MODE={mode} {img} -d memory_limit=512M /mnt/vm/{rr.SANITY_SCRIPT}"))
    for k in range(1, a.reps + 1):
        for name, fname in scripts:
            runs.append((f"{arm}__{name}__r{k}", f"PHP_CAP_MODE={mode} {img} -d memory_limit=512M /mnt/vm/{fname}"))
cmd = [sys.executable, str(R / "ports/php/app/run-virtual.py"),
       "--qemu", str(a.kit / "qemu-system-riscv64"), "--images", str(a.kit / "images"), "--adapter", str(a.kit / "adapter"),
       "--image", str(a.image), "--record", str(a.out / "run.json"), "--serial-log", str(a.out / "serial.log"),
       "--timeout", "3000", "--run-timeout", str(a.run_timeout)]
for f in sorted(stage.iterdir()):
    cmd += ["--extra", str(f)]
for name, argv in runs:
    cmd += ["--run", f"{name}={argv}"]
(a.out / "runs.json").write_text(json.dumps({"image": str(a.image), "image_sha256": rr.sha256_file(a.image),
                                             "runs": runs}, indent=1) + "\n")
sys.exit(subprocess.call(cmd))
