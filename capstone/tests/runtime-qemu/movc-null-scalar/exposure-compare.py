#!/usr/bin/env python3
"""Compare the two arms exposure.sh ran: MOVC keeping an integer source (off) and nulling it (on).

usage: exposure-compare.py <OUT>

Prints every nightly suite whose verdict differs, BEEBS' and linear-uninit-corpus' failure lists,
the bare HostCall wire probes one at a time, the delegated runtime probes and libc-test test by
test (their runners' --report files), and the switch's notice count in each arm: any notice with
the switch off, or none with it on, means the arm did not measure what it claims; ON logs that show
a boot but no notice are listed as unverified. The movc probe in each arm's guest is the control
that the guest ran with the switch it claims; if it did not pass, the comparison is refused. A
missing artifact is an ERROR, never a match.
Exit 0 with the arms identical, 1 if they differ, 2 if the comparison could not be made.
"""
import collections, json, pathlib, re, sys

OUT = pathlib.Path(sys.argv[1])
NOTICE = "MOVC-NULL-SCALAR first non-zero source nulled"
errors, differs = [], False


def nightly(arm):
    console = OUT / f"nightly-{arm}.console"
    m = re.findall(r"full report: (\S+)", console.read_text(errors="replace")) if console.exists() else []
    if not m:
        errors.append(f"{console}: no nightly report (missing, or the run did not finish)")
        return None, {}
    report = pathlib.Path(m[-1])
    suites = {}
    for line in report.read_text().splitlines():
        mm = re.match(r"\| (\S+) \| (\S+) \|", line)
        if mm and mm.group(1) != "suite" and not mm.group(1).startswith("-"):
            suites[mm.group(1)] = mm.group(2)
    return report.parent, suites


def failures(logdir, name, pattern):
    log = logdir / f"{name}.log" if logdir else None
    if not log or not log.exists():
        return None
    return sorted(set(re.findall(pattern, log.read_text(errors="replace"))))


def report(kind, arm):
    """A runner's --report file, or an ERROR."""
    p = OUT / f"{kind}-{arm}.json"
    if not p.exists():
        errors.append(f"missing {p}")
        return None
    return json.loads(p.read_text())["results"]


def libc(arm):
    res = report("libc", arm)
    return None if res is None else {name: verdict for name, (verdict, _) in res.items()}


def probes(arm):
    res = report("probes", arm)
    return None if res is None else {name: row["verdict"] for name, row in res.items()}


def notices(arm):
    """(notices, boots, logs with a boot but no notice). A runner that keeps only QEMU's stdout
    loses the notice, which is on stderr, so such a log is unverified rather than wrong."""
    count, boots, silent = 0, 0, []
    for p in (OUT / arm).rglob("*"):
        if p.is_file() and p.suffix in (".log", ".txt", ".out", ".console"):
            text = p.read_text(errors="replace")
            n, b = text.count(NOTICE), len(re.findall(r"OpenSBI v\d", text))
            count, boots = count + n, boots + b
            if b and not n:
                silent.append(p.relative_to(OUT))
    return count, boots, silent


(doff, off), (don, on) = nightly("off"), nightly("on")
if off and on:
    diff = {k: (off.get(k), on.get(k)) for k in sorted(set(off) | set(on)) if off.get(k) != on.get(k)}
    differs |= bool(diff)
    print(f"nightly: {len(on)} suites; differing: {diff or 'none'}")
    for name, pat in (("beebs", r"FAIL run-beebs-\S+"), ("linear-uninit-corpus", r"FAIL\s+\S+  \(rc=\S+;")):
        a, b = failures(doff, name, pat), failures(don, name, pat)
        if a is None or b is None:
            errors.append(f"{name}: log missing in one arm")
            continue
        differs |= a != b
        print(f"  {name} failures {'identical' if a == b else 'DIFFER'}: off {len(a)}, on {len(b)}"
              + ("" if a == b else f"\n    only off: {sorted(set(a) - set(b))}\n    only on:  {sorted(set(b) - set(a))}"))

    both = sorted(k for k in on if k in off and not off[k].startswith("PASS") and not on[k].startswith("PASS")
                  and not k.startswith("lit"))
    if both:
        print(f"  not passing in either arm (whatever they would have run after the failure was not"
              f" measured; read their logs): {both}")

def hostcall(arm):
    """probe -> 'ok' / 'halted' / 'failed rc=N', from the one-probe-at-a-time runs."""
    res = {}
    for f in sorted(OUT.glob(f"hostcall-{arm}-run-hostcall-*.txt")):
        text = f.read_text(errors="replace")
        name = f.name[len(f"hostcall-{arm}-run-hostcall-"):-len("-probe.txt")]
        rc = re.findall(r"^rc=(\d+)", text, re.M)
        halted = "domain halted" in text or "Traceback" in text
        res[name] = "halted" if halted else ("ok" if rc and rc[-1] == "0" else f"failed rc={rc[-1] if rc else '?'}")
    return res
hoff, hon = hostcall("off"), hostcall("on")
if not hoff or not hon:
    errors.append("no one-at-a-time hostcall probe results in one arm")
else:
    diff = {k: (hoff.get(k), hon.get(k)) for k in sorted(set(hoff) | set(hon)) if hoff.get(k) != hon.get(k)}
    differs |= bool(diff)
    print(f"hostcall probes one at a time: {len(hon)}; ok off {sum(v == 'ok' for v in hoff.values())},"
          f" on {sum(v == 'ok' for v in hon.values())}; differing: {diff or 'none'}")
    both = sorted(k for k in hon if hoff.get(k) != "ok" and hon[k] != "ok")
    if both:
        print(f"  not ok in either arm: {both}")

poff, pon = probes("off"), probes("on")
if poff is not None and pon is not None:
    for arm, res in (("off", poff), ("on", pon)):
        if res.get("movc") != "PASS":
            errors.append(f"the movc control did not pass in the {arm} arm ({res.get('movc')}): its guest "
                          "did not run with the switch the arm claims")
    diff = {k: (poff.get(k), pon.get(k)) for k in sorted(set(poff) | set(pon))
            if poff.get(k) != pon.get(k) and k != "movc"}
    differs |= bool(diff)
    print(f"delegated runtime probes: {len(pon)}; pass off {sum(v == 'PASS' for v in poff.values())},"
          f" on {sum(v == 'PASS' for v in pon.values())}; differing (movc aside): {diff or 'none'}")

loff, lon = libc("off"), libc("on")
if loff is not None and lon is not None:
    diff = {k: (loff.get(k), lon.get(k)) for k in sorted(set(loff) | set(lon)) if loff.get(k) != lon.get(k)}
    differs |= bool(diff)
    print(f"libc-test off: {dict(sorted(collections.Counter(loff.values()).items()))}")
    print(f"libc-test on:  {dict(sorted(collections.Counter(lon.values()).items()))}")
    print(f"libc-test differing: {diff or 'none'}")

(noff, boff, _), (non, bon, silent) = notices("off"), notices("on")
print(f"switch notices: off {noff} in {boff} boots, on {non} in {bon} boots")
if silent:
    print(f"  ON logs with a boot and no notice (unverified; their runner may drop QEMU's stderr): "
          f"{len(silent)}: {', '.join(map(str, silent[:8]))}{' ...' if len(silent) > 8 else ''}")
if noff:
    errors.append("the switch printed a notice in the OFF arm: that arm was not QEMU's default")
if non == 0:
    errors.append("no notice in the ON arm at all: it did not run with the switch")

for e in errors:
    print("ERROR", e)
sys.exit(2 if errors else (1 if differs else 0))
