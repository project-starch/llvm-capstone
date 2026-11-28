#!/usr/bin/env python3
"""assemble-record.py: write the files of results/2026-11-12-crash-139 that are derived from this work directory's
run records. Launcher mapping lines (CAPSTONE_VM_ADD_OK) and kernel console lines are removed from the stored
transcripts and counted, as the earlier records did; exits and every other line are kept; a work-directory prefix
in a path is removed. Refuses to write a file in which a home path survives."""
import json, os, re, sys
from pathlib import Path
W = Path(os.environ["W"]); A = os.environ["A"]; home = os.path.expanduser("~")
OUT = Path(sys.argv[1])
kernel = re.compile(r"^\[\s*\d+\.\d+\]")

def clean_text(s):
    for prefix in (str(W) + "/", A + "/", home + "/crenew-step4/corpus/", home + "/crenew-step4/", home + "/"):
        s = s.replace(prefix, "")
    return s

def write(path, text):
    assert home not in text and os.path.basename(home) not in text, f"a home path or user name survives in {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)

def record(src, dst, serial):
    d = json.load(open(src)); removed = {}
    for name, lines in d["output"].items():
        keep = [clean_text(s) for s in lines if not s.startswith("CAPSTONE_VM_ADD_OK") and not kernel.match(s)]
        removed[name] = len(lines) - len(keep); d["output"][name] = keep
    d["serial_log"] = serial
    d["filtered"] = {"what": "CAPSTONE_VM_ADD_OK launcher mapping lines and kernel console lines removed from output; "
                             "exits and every other line kept", "removed_per_run": removed}
    runs = json.load(open(Path(src).parent / "runs.json"))
    d["image_sha256"] = runs["image_sha256"]
    d["argv"] = {n: a for n, a in runs["runs"]}
    write(dst, json.dumps(d, indent=1) + "\n")
    return d

BOOTS = [  # label, source dir, destination, image description, platform
    ("measured-boot1", "runs/measured-boot1", "runs/measured-boot1.json", "the measured image (run19's)", "the branch's QEMU"),
    ("measured-boot2", "runs/measured-boot2", "runs/measured-boot2.json", "the measured image (run19's)", "the branch's QEMU"),
    ("dev-control-boot1", "runs/dev-control-boot1", "runs/dev-control-boot1.json", "the image built from unmodified dev (run19's control-dev)", "dev's QEMU"),
    ("witness-boot1", "runs/witness-boot1", "probes/witness-boot1.json", "probe image: witness", "the branch's QEMU"),
    ("forced-boot1", "runs/forced-boot1", "probes/forced-boot1.json", "probe image: forced", "the branch's QEMU"),
    ("proposed-boot1", "runs/proposed-boot1", "probes/proposed-boot1.json", "probe image: proposed", "the branch's QEMU"),
    ("proposed-forced-boot1", "runs/proposed-forced-boot1", "probes/proposed-forced-boot1.json", "probe image: proposed, forced", "the branch's QEMU"),
]
rows = [("boot", "image", "image_sha256", "platform", "arm", "script", "run", "exit", "php_output", "issues", "retires",
         "refcount_printed_by_script", "witness_getters", "witness_member_refcount", "witness_foreign_frees",
         "foreign_free_cursor_minus_code_base", "fault", "fail_line")]
for label, src, dst, what, platform in BOOTS:
    d = record(W / src / "run.json", OUT / dst, f"{label}-serial.log (not archived)")
    assert d["booted"] and d["completed"] and not d["panic"], label
    for name, lines in d["output"].items():
        arm, script, *rep = name.split("__")
        php = [l for l in lines if not re.match(r"(ZEND_CAP|PHP_VIRTUAL|CAPSTONE_VM|capstone-exec|\[CAPSTONE\])", l)]
        text = "\n".join(lines)
        m = re.search(r"issues=(\d+) retires=(\d+)", text)
        rc = re.search(r"refcount\((\d+)\)", text)
        we = re.search(r"ZEND_CAP_WITNESS exit reclaims=\d+ foreign=(\d+) first_foreign=\S+ getters=(\d+) member_refcount=(\d+)", text)
        wf = re.search(r"ZEND_CAP_WITNESS foreign-free cursor=(0x[0-9a-f]+) .*getters=(\d+) member_refcount=(\d+)", text)
        fl = re.search(r"domain fault cause=(\d+) pc=(0x[0-9a-f]+) .*?code=(0x[0-9a-f]+)-", text)
        getters = (we.group(2) if we else wf.group(2) if wf else "")
        refc = (we.group(3) if we else wf.group(3) if wf else "")
        if getters == "0":
            refc = ""        # the sanity script calls no __get: the field was never read
        foreign = we.group(1) if we else ("1" if wf else "")
        off = hex(int(wf.group(1), 16) - int(fl.group(3), 16)) if wf and fl else ""
        fault = f"cause={fl.group(1)} pc-code_base={int(fl.group(2), 16) - int(fl.group(3), 16):#x}" if fl else ""
        zf = re.search(r"ZEND_CAP_FAIL code=(\d+) (free|header) of a pointer", text)
        fail = f"ZEND_CAP_FAIL code={zf.group(1)} ({zf.group(2)})" if zf else ""
        rows.append((label, what, d["image_sha256"], platform, arm, script, rep[0] if rep else "", str(d["exits"].get(name)),
                     " | ".join(php), m.group(1) if m else "", m.group(2) if m else "", rc.group(1) if rc else "",
                     getters, refc, foreign, off, fault, fail))
write(OUT / "runs/matrix.tsv", "\n".join("\t".join(r) for r in rows) + "\n")
print(f"{len(rows) - 1} runs in matrix.tsv")
