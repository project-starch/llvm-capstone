#!/usr/bin/env python3
"""b4-check-t1c.py <spec.txt> <run.py> <contract.dom> <twin result.json> <board.txt>: judge B's check boot 3 (PREREG
Addendum 17), b4-t1c: run.py@69fb6cae3db4's Tier 1 SCRIPT with the module's resolve_dirty=1 (the S-24 fix), then a
clock check and a spin pair.

b4-check-t1b.py's method: run.py's OWN analysis block, lifted verbatim from <run.py> (the text between
`lines = [s.strip()` and `result = {`), executed over the step text's lines from `@@B4-STAGE t1c` up to and including
the Tier 1 script's VIRTUAL_RUNTIME_DONE. Nothing after that line (the @@T1C- clock and spin lines) reaches the analysis.

Graded (exit 0 only if every one holds):
- R  read-back, for resolve_dirty and resolve_pte each. Exactly one `@@PARAM <name>=Y` in <board.txt> (boardvm's sysfs
     write, before the step), and exactly one `@@T1C-RD` and one `@@T1C-RD2` line in the step text, each reading Y for
     both. Absent, or anything but Y: READBACK FAIL, exit 3 -- the boot did not run the configuration it is a record
     of. This is checked first, and also when boardvm refused the run and there is no step text.
- P1 every Tier 1 cell equals the twin F0r's (t1-F0r, resolve_dirty=1), except cleanup (false on the board by
     construction: its kernel cannot unload a module) and spin_entered (excluded, reported: t1b's open cell).
- P2 S-24's prediction for this boot: vm exits 1 with `VM_CONTRACT_FAIL:execute through mapping PCC` (F0r's own
     status and failure point, and vp2 r13's resolve_dirty-on reading), and no S-24 bound line (`resolved N times with
     no change`) in the step text. P1 carries the rest of it: the four cells t1b's S-24 fault moved
     (nested_denial_site, vm_denial_sites, linux_pages_released, linux_demand_faults) read F0r's values. Any other vm
     reading reopens t1b's attribution.
- P3 the board's domain-fault cause sequence is t1b's minus its leading S-24 cause 15: [25,25,28,25,15,13,13,28,25].
Reported, not graded:
- the spin pair (`@@T1C-SPIN wait=5|15 rc= ready=`) and the reading it selects (table in Addendum 17);
- the clock check: wall time (date +%s%N, and /proc/uptime) around one ok launch beside its launch_ns/elapsed_ns.
Exit 0 as predicted; 1 off the prediction; 2 no data; 3 read-back failed. Needs CAPSTONE_LLVM_BIN (llvm-nm)."""
import json, os, re, subprocess, sys
from pathlib import Path

if len(sys.argv) != 6:
    sys.exit(__doc__)
tp, runpy, app, twin, board = (Path(x) for x in sys.argv[1:])
PARAMS = ("resolve_dirty", "resolve_pte")
btext = board.read_text(errors="replace").replace("\r", "") if board.is_file() else ""
board_rb = {n: re.findall(rf"^@@PARAM {n}=(\S+)\s*$", btext, re.M) for n in PARAMS}
if not tp.is_file():
    if board.is_file() and any(v != ["Y"] for v in board_rb.values()):
        print(f"READBACK FAIL: no step text, and {board} reads {board_rb} (boardvm refuses a run whose parameters do "
              f"not read back): the boot did not run resolve_dirty=Y resolve_pte=Y; no reading"); sys.exit(3)
    print(f"NO DATA: {tp} does not exist"); sys.exit(2)
src = runpy.read_text()
i = src.index("    lines = [s.strip() for s in out.decode(errors='replace').splitlines()]")
j = src.index("    result = {'status'", i)
block = "\n".join(l[4:] if l.startswith("    ") else l for l in src[i:j].splitlines()[1:])
text = tp.read_text(errors="replace").replace("\r", "")
k = text.find("@@B4-STAGE t1c")
if k < 0:
    print(f"NO DATA: no @@B4-STAGE t1c marker in {tp}"); sys.exit(2)
every = [s.strip() for s in text[k:].splitlines()]
if "VIRTUAL_RUNTIME_DONE" not in every:
    print(f"NO DATA: no VIRTUAL_RUNTIME_DONE in {tp}: the Tier 1 script did not finish"); sys.exit(2)
lines = every[:every.index("VIRTUAL_RUNTIME_DONE") + 1]
after = every[len(lines):]
if not any(s.startswith("VM_EXIT:") for s in lines):
    print(f"NO DATA: no VM_EXIT line in {tp}"); sys.exit(2)

# R: the read-back of both parameters, three times over, all Y.
rb = {}
for n in PARAMS:
    rb[f"@@PARAM {n} (board.txt)"] = board_rb[n]
    for tag in ("@@T1C-RD", "@@T1C-RD2"):
        rb[f"{tag} {n}"] = [m for s in every if s.startswith(tag + " ") for m in re.findall(rf"\b{n}=(\S+)", s)]
rb_ok = all(v == ["Y"] for v in rb.values())
for n, v in rb.items():
    print(f"  {'OK ' if v == ['Y'] else 'BAD'} read-back {n:20} {v}")

class A:
    pass
a = A()
a.skip_recycling = False; a.legacy_application = None; a.perl = None; a.perl_smoke = None; a.php = None
a.exact_bounds = False; a.application = app
ns = {"lines": lines, "a": a, "re": re, "os": os, "subprocess": subprocess, "Path": Path}
exec("tests = {}\n" + block, ns)
tests = ns["tests"]
ref = json.loads(twin.read_text())["tests"]
want = dict(ref); want["cleanup"] = False
EXCLUDED = ("spin_entered",)
off = sorted(n for n in set(tests) | set(want) if n not in EXCLUDED and tests.get(n) != want.get(n))
for n in sorted(set(tests) | set(want)):
    tag = "EXC" if n in EXCLUDED else ("OK " if n not in off else "OFF")
    print(f"  {tag} {n:28} board {tests.get(n)} twin {ref.get(n)}"
          + ("   (cleanup: false by construction)" if n == "cleanup" else "")
          + ("   (excluded from the prediction; reported)" if n in EXCLUDED else ""))

vm_exit = [s for s in lines if s.startswith("VM_EXIT:vm:")]
p2 = vm_exit == ["VM_EXIT:vm:1"] and lines.count("VM_CONTRACT_FAIL:execute through mapping PCC") == 1
bound = [s for s in lines if re.search(r"resolved \d+ times with no change", s)]
p2 = p2 and not bound
print(f"  {'OK ' if p2 else 'OFF'} P2 (S-24) vm: {vm_exit}, execute-through-PCC lines "
      f"{lines.count('VM_CONTRACT_FAIL:execute through mapping PCC')}, S-24 bound lines {len(bound)}")
causes = [int(m[1]) for s in lines if (m := re.match(r"capstone-exec: domain fault cause=(\d+) ", s))]
P3 = [25, 25, 28, 25, 15, 13, 13, 28, 25]
p3 = causes == P3
for s_ in bound:
    print(f"      S-24 bound line (resolve_pte=Y prints its PTE): {s_[:260]}")
print(f"  {'OK ' if p3 else 'OFF'} P3 fault causes {causes} (predicted {P3})")

# Reported: the spin pair and the clock.
spin = {}
for s in after:
    m = re.match(r"@@T1C-SPIN wait=(\d+) rc=(\d+) ready=(\d+)$", s)
    if m:
        spin[int(m[1])] = (int(m[2]), int(m[3]))
READ = {(False, True): "A: slow start -- the 5 s wait is shorter than a spin launch on this board; spin works on B",
        (True, True): "B: not a fixed start-up cost -- both waits see readiness; t1b's miss needs another account",
        (False, False): "C: no readiness within 15 s -- slower still (check the clock line) or spin does not reach "
                        "readiness on B; open, next: a directed spin run",
        (True, False): "D: inconsistent -- readiness at 5 s but not 15 s; nondeterminism, read the outputs"}
if set(spin) == {5, 15}:
    r5, r15 = spin[5][1] == 1, spin[15][1] == 1
    print(f"  REP spin pair: wait 5 s rc={spin[5][0]} ready={spin[5][1]}; wait 15 s rc={spin[15][0]} ready={spin[15][1]}"
          f" -> reading {READ[(r5, r15)]}")
else:
    print(f"  REP spin pair: NO DATA (lines for waits {sorted(spin)})")
print(f"  REP Tier 1 spin_entered (its own 5 s wait): {tests.get('spin_entered')}")
clk = [s for s in after if s.startswith("@@T1C-CLOCK ")]
if len(clk) == 1:
    f = dict(re.findall(r"(\w+)=(\S+)", clk[0]))
    try:
        wall = (int(f["w1"]) - int(f["w0"])) / 1e9
    except (KeyError, ValueError):
        wall = None
    try:
        up = float(f["u1"]) - float(f["u0"])
    except (KeyError, ValueError):
        up = None
    el = int(f["elapsed_ns"]) / 1e9 if "elapsed_ns" in f else None
    ln = int(f["launch_ns"]) / 1e9 if "launch_ns" in f else None
    print(f"  REP clock: ok rc={f.get('rc')} wall(date) {wall} s, wall(uptime) {up} s, launcher elapsed {el} s, "
          f"launch {ln} s" + (f"; elapsed/uptime {el / up:.3f}" if el and up else ""))
else:
    print(f"  REP clock: NO DATA ({len(clk)} @@T1C-CLOCK lines)")

ok = not off and p2 and p3
print(f"B4-T1C {sum(bool(v) for v in tests.values())} of {len(tests)} true; twin F0r {sum(ref.values())} of {len(ref)}; "
      + ("READBACK FAIL: the boot did not run resolve_dirty=Y resolve_pte=Y; no reading" if not rb_ok else
         "AS PREDICTED (every cell equals F0r's but cleanup false and spin_entered excluded; P2, P3 hold)" if ok else
         f"OFF THE PREDICTION: cells {off}; P2 {'ok' if p2 else 'OFF'}; P3 {'ok' if p3 else 'OFF'}"))
sys.exit(3 if not rb_ok else 0 if ok else 1)
