#!/usr/bin/env python3
"""s23-instances.py: S-23's data-face instances in one table, read from each run's raw console. Each capture is scoped to
its own boot, decoded from the driver's [uart] events after the boot's own load_image. Every field is extracted by pattern
from that text, none written by hand, and a field a capture lacks reads "?". Exit 1 if any capture has no MTVL."""
import ast, os, re, sys, pathlib
HDD = os.environ.get("MNT_HDD", "/mnt/hdd/" + os.environ.get("USER", ""))   # the captures' disk
RUNS = [  # (instance, bitstream, module, program/step, capture)
    ("pb10", "bf3db3175", "3005ed52", "ffmpeg pool, Sublet arm", HDD + "/capstone/t3-run/pb10/raw/guest/driver.log"),
    ("pb10-fix6b", "bf3db3175", "d71f207a", "ffmpeg pool, Sublet arm", HDD + "/capstone/t3-run-fix6/pb10-fix6b/raw/guest/driver.log"),
    ("treeA step 4", "8522d72fe (tree A)", "39d8492d", "Tier 1 vm-suite: sparse-churn", "/tmp/capstone/tier1/run3rev7A/driver.log"),
    ("sql0-treeA", "8522d72fe (tree A)", "39d8492d", "postgres --single, session preflight-00", HDD + "/capstone/t3-run-treeA/sql0-treeA/raw/guest/driver.log"),
]
def console(path):
    t = open(path, errors="replace").read(); k = t.rfind("load_image images/"); t = t[k:]
    return "".join(ast.literal_eval(m) for m in re.findall(r"\[fpga\] \[uart\] ('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")", t)).replace("\r", "")
def f(p, s, g=1):
    m = re.search(p, s); return m.group(g) if m else "?"
bad = 0
print("| instance | bitstream | module | program | FRTY before | slot 7 | CPHT | MTVL | Oops epc | ra | refused insn | reg | seconds | step started |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, bit, mod, prog, path in RUNS:
    c = console(path); mt = f(r"MTVL:([0-9A-F]{8})", c)
    if mt == "?": bad += 1
    i = c.find("MTVL:"); pre = c[:i]
    frty = re.findall(r"FRTY:([0-9A-F]{8})", pre)
    slot7 = f(r"CPID:00001507\s*CPBA:([0-9A-F]{8})\s*CPEN:([0-9A-F]{8})", c, 0)
    slot7 = "[" + "-".join(re.findall(r"CP(?:BA|EN):([0-9A-F]{8})", slot7)) + ")" if slot7 != "?" else "?"
    code = f(r"Code: [0-9a-f ]*\(([0-9a-f]{4})\) ([0-9a-f]{4})", c, 0)
    insn = "?" if code == "?" else (lambda m: m.group(2) + m.group(1))(re.search(r"\(([0-9a-f]{4})\) ([0-9a-f]{4})", code))
    reg = f(r"\b(a[0-7]) : (ffffffd83c2[0-9a-f]{5})", c, 0)
    secs = f(r"\[\s*([0-9.]+)\] Oops - load access fault", c)
    step = f(r"(@@STEP-BEGIN [^\n]+|VM_BEGIN:[^\n]+)\n(?:(?!@@STEP-BEGIN|VM_BEGIN)[\s\S])*MTVL:", c)
    step = re.findall(r"@@STEP-BEGIN [^\n]+|VM_BEGIN:[^\n]+", pre)[-1:] or ["?"]
    print(f"| {name} | {bit} | {mod} | {prog} | {'FRTY:' + frty[-1] if frty else 'none'} | {slot7} | {f(r'CPHT:([0-9A-F]{8})', c)} | {mt} | "
          f"{f(r'epc : ([a-z_0-9+/x]+)', c)} | {f(r' ra : ([a-z_0-9+/x]+)', c)} | {insn} | {reg} | {secs} | {step[0]} |")
sys.exit(1 if bad else 0)
