#!/usr/bin/env python3
"""One infra retry around a bare runner: SUP_REAL_RUNNER=<run_sup_bare*.py> python3 infra-retry.py.

The rule (board-run skill, "JTAG errors | infra -- retry"; precedent board-36a641e0b.result-lines.txt:7-9, where a JTAG
load_image timeout "waiting for busy to go low" was re-run once, same image): an attempt that exits non-zero WITHOUT
having started the image -- no "] continue" line and no reading -- produced no reading, so it is run ONCE more with the
same image and environment. Anything else is returned as it is: a run that started and then failed is a result, and a
second non-start is a missing reading that the comparators report.

Output: attempt 1 streams to stdout unchanged; a retry is announced by an "INFRA-RETRY" line, then attempt 2 streams.
Attempt 1's run directory moves to <SUP_OUT>.infra-fail-1 and its output is kept in <SUP_OUT>.infra-fail-1.log, both
outside the runs-*.log names the comparators glob."""
import os, re, shutil, signal, subprocess, sys, time

real = os.environ.get("SUP_REAL_RUNNER")
if not real or not os.path.isfile(real):
    sys.exit(f"infra-retry: SUP_REAL_RUNNER not set or missing: {real!r}")
child = None

def term(sig, frm):
    if child and child.poll() is None:
        child.terminate()
        try: child.wait(timeout=60)
        except subprocess.TimeoutExpired: child.kill()
    sys.exit(128 + sig)

signal.signal(signal.SIGTERM, term)

def attempt():
    global child
    child = subprocess.Popen([sys.executable, real], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             errors="replace")
    lines = []
    for line in child.stdout:
        sys.stdout.write(line); sys.stdout.flush(); lines.append(line)
    return child.wait(), "".join(lines)

rc, out = attempt()
started = re.search(r"\] continue\s*$", out, re.M) or re.search(r"reading +\d+: 0x", out)
if rc == 0 or started:
    sys.exit(rc)
why = "busy-timeout" if "waiting for busy to go low" in out else "no-start"
sup_out = os.environ.get("SUP_OUT", "")
if sup_out and os.path.isdir(sup_out):
    shutil.move(sup_out, sup_out + ".infra-fail-1")
if sup_out:
    with open(sup_out + ".infra-fail-1.log", "w") as f:
        f.write(out)
print(f"INFRA-RETRY {time.strftime('%H:%M:%S')}: attempt 1 rc={rc} never started the image ({why}); "
      f"one retry, same image {os.environ.get('SUP_IMG', '?')}", flush=True)
time.sleep(8)
rc2, _ = attempt()
sys.exit(rc2)
