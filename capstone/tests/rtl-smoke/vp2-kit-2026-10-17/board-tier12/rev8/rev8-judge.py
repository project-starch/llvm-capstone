#!/usr/bin/env python3
"""rev8-judge.py BOARD_TXT: module 39d8492d's rev8 Tier 1 boot, read beyond the rungs, as JUDGE lines and a last
PASS/FAIL line. It reads only the console lines the stage scripts print (KMSG- copies excluded). The predictions are the
compiler lane's QEMU readings on the twin module a5dec6a32c49b3d3 (kit qkit7-rev2, node_flat_records=4096).
- @@PARAM read-backs: diag_emerg=Y, node_flat_records=4096, supervisor_slots=1, each once.
- rev7's open check, made exact by the read between `ok` and `stale`: across `ok`, resume_sfences rises by `ok`'s own
  STATS faults (> 0) and resume_satp_refreshes by its steps - 1 (QEMU: 426 of faults=426, and 546 of steps=547). The
  board's own STATS values are used, not QEMU's: on silicon rev7's `ok` read steps=511. resume_satp_changed and
  busy_frame_refusals read 0 at every one of the five reads.
- dense-churn, the stopgap's control: the must-fire "collect skipped" exactly once, with "retired 1" and "skips 1";
  then `domain fault cause=30` (the table exhausted, RISCV_EXCP_INSUF_RESOURCES) at the dense loop's CDERIVE, image VA
  0x24238 (pc & 0xffff == 0x4238, the image base being 64 KiB-aligned); the last progress line derived=3000;
  VM_EXIT:dense-churn:139.
- the banner once (slots 1, diag_emerg 1), and 0 "wrote no event record" refusal lines.
A counter that reads ABSENT, or a missing STATS line, is NO DATA, never a pass. Exit 0 PASS, 1 FAIL, 2 NO DATA."""
import re, sys
t = open(sys.argv[1], errors="replace").read().replace("\r", "")
txt = "\n".join(l for l in t.split("\n") if not l.startswith("KMSG-"))
out, bad, nodata = [], 0, 0
def say(ok, msg):
    global bad
    out.append(("OK   " if ok else "FAIL ") + msg); bad += not ok
def param(name, when=None):
    return re.findall(rf"^@@PARAM {name}=(\S+?)" + (rf":{when}$" if when else r"$"), txt, re.M)
for n, want in (("diag_emerg", "Y"), ("node_flat_records", "4096"), ("supervisor_slots", "1")):
    v = param(n); say(v == [want], f"@@PARAM {n} read back {v} (want [{want!r}])")
W = ("before-contract", "after-ok", "after-contract", "before-churn", "after-churn")
cnt = {}
for w in W:
    for n in ("resume_sfences", "resume_satp_refreshes", "resume_satp_changed", "busy_frame_refusals"):
        v = param(n, w)
        if len(v) != 1 or not v[0].isdigit(): out.append(f"NO DATA counter {n}:{w} read {v}"); nodata += 1
        else: cnt[(n, w)] = int(v[0])
a = txt.find(":before-contract\n"); b = txt.find(":after-ok\n")
ok_stats = re.findall(r"^CAPSTONE_VM_STATS .*?\bsteps=(\d+) faults=(\d+)", txt[a:b], re.M) if 0 <= a < b else []
if len(ok_stats) != 1:
    out.append(f"NO DATA: {len(ok_stats)} CAPSTONE_VM_STATS lines between before-contract and after-ok (want 1, `ok`'s)"); nodata += 1
elif not nodata:
    steps, faults = int(ok_stats[0][0]), int(ok_stats[0][1])
    d1 = cnt[("resume_sfences", "after-ok")] - cnt[("resume_sfences", "before-contract")]
    d2 = cnt[("resume_satp_refreshes", "after-ok")] - cnt[("resume_satp_refreshes", "before-contract")]
    say(d1 == faults and faults > 0, f"resume_sfences delta across ok {d1} == ok's faults {faults} > 0")
    say(d2 == steps - 1, f"resume_satp_refreshes delta across ok {d2} == ok's steps - 1 {steps - 1}")
    for n in ("resume_satp_changed", "busy_frame_refusals"):
        say(all(cnt[(n, w)] == 0 for w in W), f"{n} 0 at every read: {[cnt[(n, w)] for w in W]}")
ck = re.findall(r"capstone_vm: collect skipped: collection is not available on bf3db3175 \(Q-27, S-entry\); retired (\d+) available (\d+), skips (\d+)", txt)
say(len(ck) == 1 and ck[0][0] == "1" and ck[0][2] == "1", f"collect skipped {ck} (want once, retired 1, skips 1; QEMU available 256)")
fl = re.findall(r"domain fault cause=(\d+) pc=(0x[0-9a-f]+)", txt[txt.find("VM_BEGIN:dense-churn"):] if "VM_BEGIN:dense-churn" in txt else "")
say(len(fl) == 1 and fl[0][0] == "30" and int(fl[0][1], 16) & 0xffff == 0x4238, f"dense-churn fault {fl} (want one, cause 30, pc & 0xffff == 0x4238)")
pr = [int(x) for x in re.findall(r"VIRTUAL_DENSE_PROGRESS derived=(\d+)", txt)]
say(pr[-1:] == [3000], f"last progress derived={pr[-1:]} of {len(pr)} lines (want 3000)")
ex = re.findall(r"^VM_EXIT:dense-churn:(\d+)$", txt, re.M)
say(ex == ["139"], f"VM_EXIT:dense-churn {ex} (want ['139'])")
ban = re.findall(r"capstone_vm: loaded, diag [a-z0-9-]+, supervisor_slots=(\d+), diag_emerg=(\d)", txt)
ref = len(re.findall(r"capstone_vm: CSRUNV action \d+ wrote no event record", txt))
say(len(ban) == 1 and ban[0] == ("1", "1") and ref == 0, f"banner {ban} (want one, slots 1, diag_emerg 1); refusal lines {ref} (want 0)")
print("\n".join(out))
if nodata: print("JUDGE NO DATA"); sys.exit(2)
print("JUDGE PASS" if not bad else f"JUDGE FAIL ({bad})"); sys.exit(1 if bad else 0)
