#!/usr/bin/env python3
"""rev7-judge.py BOARD_TXT: the module-39d8492d Tier 1 boot's readings beyond the rungs, as JUDGE lines and a last
PASS/FAIL line. Reads only the console lines the stage scripts print (KMSG- copies excluded).
- @@PARAM read-backs: diag_emerg=Y, node_flat_records=4096, supervisor_slots=1, each once (insmod-time).
- the counters: deltas across vm-contract must equal the sums over its two CAPSTONE_VM_STATS lines: resume_sfences
  == sum(faults), resume_satp_refreshes == sum(steps - 1); resume_satp_changed and busy_frame_refusals stay 0 at every
  read. A counter that reads ABSENT, or a missing STATS line, is NO DATA (never a pass).
- sparse-churn: VM_EXIT:sparse-churn:1, a VM_CONTRACT_FAIL:recycling allocation line, and the must-fire "capstone_vm:
  collect skipped: collection is not available on bf3db3175" on the console at least once.
- the refusal must-fire "wrote no event record" 0, beside the banner once (diag_emerg=1 in it).
Exit 0 PASS, 1 FAIL, 2 NO DATA."""
import re, sys
t = open(sys.argv[1], errors="replace").read().replace("\r", "")
lines = [l for l in t.split("\n") if not l.startswith("KMSG-")]
txt = "\n".join(lines)
out, bad, nodata = [], 0, 0
def say(ok, msg):
    global bad
    out.append(("OK   " if ok else "FAIL ") + msg); bad += not ok
def param(name, when=None):
    pat = rf"^@@PARAM {name}=(\S+?)" + (rf":{when}$" if when else r"$")
    return re.findall(pat, txt, re.M)
for n, want in (("diag_emerg", "Y"), ("node_flat_records", "4096"), ("supervisor_slots", "1")):
    v = param(n); say(v == [want], f"@@PARAM {n} read back {v} (want [{want!r}])")
cnt = {}
for w in ("before-contract", "after-contract", "before-churn", "after-churn"):
    for n in ("resume_sfences", "resume_satp_refreshes", "resume_satp_changed", "busy_frame_refusals"):
        v = param(n, w)
        if len(v) != 1 or not v[0].isdigit():
            out.append(f"NO DATA counter {n}:{w} read {v}"); nodata += 1
        else:
            cnt[(n, w)] = int(v[0])
stats = re.findall(r"^CAPSTONE_VM_STATS .*?\bsteps=(\d+) faults=(\d+)", txt, re.M)
seg = txt[txt.find("@@PARAM resume_sfences=") if "@@PARAM resume_sfences=" in txt else 0:]
seg = seg[:seg.find(":after-contract")] if ":after-contract" in seg else seg
cstats = re.findall(r"^CAPSTONE_VM_STATS .*?\bsteps=(\d+) faults=(\d+)", seg, re.M)
if len(cstats) != 2:
    out.append(f"NO DATA: {len(cstats)} CAPSTONE_VM_STATS lines inside vm-contract (want 2: ok, stale)"); nodata += 1
elif not nodata:
    f = sum(int(x[1]) for x in cstats); s = sum(int(x[0]) - 1 for x in cstats)
    d1 = cnt[("resume_sfences", "after-contract")] - cnt[("resume_sfences", "before-contract")]
    d2 = cnt[("resume_satp_refreshes", "after-contract")] - cnt[("resume_satp_refreshes", "before-contract")]
    say(d1 == f and f > 0, f"resume_sfences delta {d1} == sum(faults) {f} > 0")
    say(d2 == s, f"resume_satp_refreshes delta {d2} == sum(steps - 1) {s}")
    for n in ("resume_satp_changed", "busy_frame_refusals"):
        say(all(cnt[(n, w)] == 0 for w in ("before-contract", "after-contract", "before-churn", "after-churn")),
            f"{n} 0 at every read: {[cnt[(n, w)] for w in ('before-contract', 'after-contract', 'before-churn', 'after-churn')]}")
ex = re.findall(r"^VM_EXIT:sparse-churn:(\d+)$", txt, re.M)
say(ex == ["1"], f"VM_EXIT:sparse-churn {ex} (want ['1'])")
say("VM_CONTRACT_FAIL:recycling allocation" in txt, "VM_CONTRACT_FAIL:recycling allocation present")
ck = len(re.findall(r"capstone_vm: collect skipped: collection is not available on bf3db3175", txt))
say(ck >= 1, f"must-fire 'collect skipped' on the console {ck} (want >= 1)")
ban = re.findall(r"capstone_vm: loaded, diag [a-z0-9-]+, supervisor_slots=(\d+), diag_emerg=(\d)", txt)
ref = len(re.findall(r"capstone_vm: CSRUNV action \d+ wrote no event record", txt))
say(len(ban) == 1 and ban[0] == ("1", "1") and ref == 0, f"banner {ban} (want one, slots 1, diag_emerg 1); refusal lines {ref} (want 0)")
print("\n".join(out))
if nodata: print("JUDGE NO DATA"); sys.exit(2)
print("JUDGE PASS" if not bad else f"JUDGE FAIL ({bad})"); sys.exit(1 if bad else 0)
