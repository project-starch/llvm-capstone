#!/usr/bin/env python3
"""The kit's slice verdicts on run directories built from stopat-run.sh's stopped windows.

usage: stopat-kit-test.py OUT PLATFORM NM   (OUT = stopat-run.sh's OUT)

Each case is a tools/profile.sh-shaped directory (stop-at.json, capprof/, invocations/, result.json)
made from a boot of stopat-run.sh, judged by tools/stop-at.py finish and, where it applies, read by
tools/prof-report.py. The unmodified copies must give SLICE and a report; each tampered copy must
give FAIL (and prof-report must refuse it): instruction bytes inside the stop function that are not
the image's (a plain slice fed by another static program at the same link address, the review's
in-memory reproduction), foreign bytes in a domain's text, and a start site whose stop never came.
Exit 1 unless every case gives its verdict."""
import json, os, shutil, subprocess, sys

src, PL, NM = sys.argv[1:4]
TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
root = os.path.join(src, "kit-test")
shutil.rmtree(root, ignore_errors=True)
ok = True


def make(name, boot, spec, exe, window, kind):
    out = os.path.join(root, name)
    os.makedirs(os.path.join(out, "invocations"))
    shutil.copytree(os.path.join(src, boot, "capprof"), os.path.join(out, "capprof"))
    shutil.copy(os.path.join(src, spec + ".json"), os.path.join(out, "stop-at.json"))
    sha = subprocess.run(["sha256sum", os.path.join(src, exe)], capture_output=True, text=True).stdout[:64]
    json.dump({"route": "guest-plain" if kind == "plain" else "guest", "status": 1, "cwd": src, "argv": [exe],
               "exe": os.path.join(src, exe), "exe_sha256": sha, "detail": {"window": window, "guest": None}},
              open(os.path.join(out, "invocations", "0000.json"), "w"))
    json.dump({"verdict": "FAIL", "reasons": ["runcpu exited 1"]}, open(os.path.join(out, "result.json"), "w"))
    open(os.path.join(out, "module-sections.txt"), "w").close()
    open(os.path.join(out, "image-records.txt"), "w").close()
    return out


def tamper_window(out, window, pick, edit):
    p = os.path.join(out, "capprof", f"window-{window}.tsv")
    lines = open(p).read().splitlines()
    i = next(k for k, l in enumerate(lines) if not l.startswith("#") and pick(l.split("\t")))
    f = lines[i].split("\t")
    lines[i] = "\t".join(edit(f))
    open(p, "w").write("\n".join(lines) + "\n")


def verdict(out):
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "stop-at.py"), "finish", out], capture_output=True, text=True)
    return json.load(open(os.path.join(out, "result.json")))["verdict"], (r.stdout + r.stderr).strip()


def report(out, kind):
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "prof-report.py"), "--out", out, "--platform", PL,
                        "--kind", kind, "--nm", NM], capture_output=True, text=True)
    return r.returncode, (r.stderr.strip().splitlines() or [""])[-1]


def expect(name, got, want, detail):
    global ok
    ok &= got == want
    print(f"{'PASS' if got == want else 'FAIL'} {name}: {got} (want {want}) {detail}")


plain = json.load(open(os.path.join(src, "plain-stop.json")))["sites"]["stop"]
site = plain["link"]

out = make("plain", "plain-stop", "plain-stop", "stopat-plain", 1, "plain")
v, why = verdict(out); expect("plain slice, unmodified", v, "SLICE", why)
rc, why = report(out, "plain"); expect("plain slice, prof-report", rc, 0, why)

# The review's reproduction: another mapping fed the stop, and what it ran inside the stop function
# is not the image's code. Here: stopat_f's second instruction recorded with other bytes.
out = make("plain-foreign", "plain-stop", "plain-stop", "stopat-plain", 1, "plain")
tamper_window(out, 1, lambda f: int(f[0], 16) == site + 4, lambda f: f[:2] + [f"{int(f[2], 16) ^ 0x100:08x}"] + f[3:])
v, why = verdict(out); expect("plain slice with foreign bytes inside stopat_f", v, "FAIL", why)
open(os.path.join(out, "slice.json"), "w").write(json.dumps(json.load(open(os.path.join(root, "plain", "slice.json")))))
rc, why = report(out, "plain"); expect("plain slice with foreign bytes inside stopat_f, prof-report", rc, 1, why)

out = make("plain-start-unreached", "plain-start", "plain-start", "stopat-plain", 1, "plain")
os.remove(os.path.join(out, "capprof", "stopped"))
json.dump({"verdict": "PASS", "reasons": []}, open(os.path.join(out, "result.json"), "w"))
v, why = verdict(out); expect("start site, stop never reached, runcpu PASS", v, "FAIL", why)

out = make("cap", "cap-stop", "cap-stop", "stopat-cap", 1, "capstone")
v, why = verdict(out); expect("domain slice, unmodified", v, "SLICE", why)
rc, why = report(out, "capstone"); expect("domain slice, prof-report", rc, 0, why)

cap = json.load(open(os.path.join(src, "cap-stop.json")))
stopped = dict(w.split("=", 1) for w in open(os.path.join(src, "cap-stop", "capprof", "stopped")).read().split())
base = int(stopped["base"], 16)
out = make("cap-foreign", "cap-stop", "cap-stop", "stopat-cap", 1, "capstone")
tamper_window(out, 1, lambda f: base <= int(f[0], 16) < base + cap["text"]["size"] and int(f[0], 16) != base + cap["sites"]["stop"]["link"] - cap["low"],
              lambda f: f[:2] + [f"{int(f[2], 16) ^ 0x100:08x}"] + f[3:])
v, why = verdict(out); expect("domain slice with foreign bytes in the text", v, "FAIL", why)
print("ALL PASS" if ok else "SOME FAIL")
sys.exit(0 if ok else 1)
