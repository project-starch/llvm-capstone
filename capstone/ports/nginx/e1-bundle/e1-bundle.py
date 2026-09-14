#!/usr/bin/env python3
"""E1 bundle: the S1/S2 cells run on the FPGA, in the paper's METHODS.md record shape.

usage: e1-bundle.py <cells.tsv> <out_dir> <boot_dir>...   (boot dirs: board-b78-r<rep>b<boot>)
One runs.jsonl record per attempted cell and repetition; points.csv is the full planned matrix
including cells not attempted; missing values are null with a reason, never zero.
"""
import sys, re, json, csv, os, hashlib, pathlib, shutil, datetime

CELLS = sys.argv[1]; OUT = pathlib.Path(sys.argv[2]); BOOTS = sys.argv[3:]
STUDY = {"p1":"S1","p2":"S1","p3":"S1","p4":"S1","p5":"S1","p6":"S1","p7":"S2",
         "s1":"S1","s2":"S1","s3":"S1","s4":"S1","s5":"S1","s6":"S1","s7":"S2","s8":"S2","s9":"S2",
         "subpool":"S2","pgsub":"S2","pghier":"S2","s10":"S1","s11":"S1"}
CASE = {1:"object-write-live", 2:"pool-destroyed-untouched", 3:"object-read-after-destroy",
        4:"same-address-reuse", 5:"reused-read", 6:"reused-free", 7:"nested-authority-after-ancestor-release",
        8:"stale-handle-vs-new-owner", 9:"stale-authority-offered-back",
        10:"diagnostic-reloaded-pointer-type", 11:"diagnostic-type-and-touch"}
# The RTL's capability causes (capstone-ariane riscv_pkg.sv:349-356). The emulator numbers the same
# faults from a different base (capstone-qemu, spec base 24), so a board cause is named here, not compared by number.
RTL_CAUSE = {25: "UNEXPECTED_OPERAND_TYPE", 26: "INVALID_CAPABILITY", 27: "UNEXPECTED_CAPABILITY_TYPE",
             28: "INSUFFICIENT_CAPABILITY_PERMISSION", 29: "CAPABILITY_OUT_OF_BOUND", 30: "ILLEGAL_OPERAND_VALUE",
             31: "INSUFFICIENT_SYSTEM_RESOURCES", 24: "DEBUG_REQUEST", 2: "ILLEGAL_INSTRUCTION", 3: "BREAKPOINT"}
rows = [l.rstrip("\n").split("\t") for l in open(CELLS) if l.strip()]
cells = {}
for r in rows:
    name, arm, stop, want, boot, va, ent, h, rc, line = (r + [""]*10)[:10]
    cells[name] = dict(name=name, arm=arm, stop=stop, want=want, boot=int(boot), va=va, entry=ent,
                       sha16=h, qemu_rc=rc, qemu_line=line)

def dom_file(name):
    return {"subpool":"ngx-subpool.dom","pgsub":"pg_subpool_test.dom","pghier":"pg_hierarchy.dom"}.get(name, f"ngx-uaf-{name}.dom")

OUT.mkdir(parents=True, exist_ok=True); (OUT/"raw").mkdir(exist_ok=True); (OUT/"analysis").mkdir(exist_ok=True)
records = []; attempted = set(); boots_meta = []
for bd in BOOTS:
    bd = pathlib.Path(bd); m = re.search(r"r(\d+)b(\d+)$", bd.name)
    rep, boot = int(m.group(1)), int(m.group(2))
    log = (bd/"driver.log").read_bytes().decode("latin1")
    # The driver frames the UART in chunks ("...'\n[fpga] [uart] '..."), so a domain's line can straddle two
    # chunks and a number can be cut in half; and the monitor's share markers (ECSA:00000004 ...) land
    # mid-token. Join the seams, unescape the newlines, delete the markers, THEN read marks.
    log = re.sub(r"'\n\[fpga\] \[uart\] '", "", log)
    log = log.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "")
    log = re.sub(r"[A-Z0-9]{4}:[0-9A-F]{8}\n?", "", log)
    i = log.rfind("monitor load_image"); seg = log[i:] if i >= 0 else log
    fw = (bd/"fw.sha").read_text().strip() if (bd/"fw.sha").exists() else None
    control = re.findall(r"RESULT k800 retval=(-?\d+)", seg)
    # The console chunks the UART, so "OpenSBI v" can straddle two lines; count the kernel's own
    # banner too and take the larger (r1b4 read 0 by the first pattern, 1 by the second).
    banners = max(len(re.findall(r"OpenSBI v", seg)), len(re.findall(r"Linux version", seg)))
    boot_id = f"sw78-r{rep}b{boot}"
    boots_meta.append(dict(boot_id=boot_id, fw_sha256=fw, control=control, boot_banners=banners))
    void = not control or control[0] != "4"
    for raw in (bd/"boot.txt", bd/"driver.log", bd/"watchdog.log", bd/"log"):
        if raw.exists(): shutil.copy(raw, OUT/"raw"/f"{boot_id}-{raw.name}")
    arms = re.split(r"\[stages\] --> TEST \d+/\d+\s+", seg)[1:]
    reached = {}
    for a in arms:
        label = a.split("\n", 1)[0].strip()
        for name, c in cells.items():
            if dom_file(name) in label:
                reached[name] = a
    for name, c in cells.items():
        if c["boot"] != boot: continue
        attempted.add((name, rep))
        a = reached.get(name)
        rec = dict(study_id=STUDY[name], manifest_id="E1-S1S2-fpga-2026-09-14", run_id=boot_id,
                   target="fpga", evidence_type="functional",
                   arm="custom-plain" if c["arm"] == "plain" else "custom-sublet",
                   series=("nginx-uaf" if name[0] in "ps" and name[1:].isdigit() else name),
                   case_id=(CASE.get(int(c["stop"])) if c["stop"].isdigit() else name),
                   parameters=dict(stop=(int(c["stop"]) if c["stop"].isdigit() else None), entry_va=c["va"],
                                   image_sha256_16=c["sha16"], allocator=("nginx-pool" if name.startswith(("p","s","sub")) else "postgresql-mmgr")),
                   repetition=rep, boot_id=boot_id, status=None, reason=None,
                   oracle=dict(expected=c["want"], observed=None, qemu_rc=c["qemu_rc"], trap_log=None),
                   metrics=dict(), raw_files=[f"raw/{boot_id}-boot.txt", f"raw/{boot_id}-driver.log"])
        if void:
            rec["status"], rec["reason"] = "invalid-run", "the boot's control did not return 4; the boot is VOID"
        elif a is None:
            rec["status"], rec["reason"] = "invalid-run", "arm not reached (an earlier arm ended the boot)"
        else:
            m = re.search(r"ngx retval = (\d+)", a); pg = re.search(r"__CAPSTONE_PG_([A-Z]+)_(GOOD|BAD|FAILED)__", a)
            wed = "NO RETURN" in a
            tl = re.search(r"TRAP LOG \{seen,mcause\[6:0\]\}\s+(0x[0-9a-f]+)", a)
            rec["oracle"]["trap_log"] = tl.group(1) if tl else None
            if m:
                mark = int(m.group(1)) & 0xFFFFFF; rec["oracle"]["observed"] = f"{mark:06X}"
                if c["want"] == "FAULT":
                    rec["status"], rec["reason"] = "unsafe-success", f"the invalid operation returned mark {mark:06X} instead of faulting"
                elif f"{mark:06X}" == c["want"].upper():
                    rec["oracle"]["match"] = True
                    rec["status"], rec["reason"] = "completed", "mark equals the emulator's"
                else:
                    # A returned mark that differs from the emulator's is a MEASUREMENT, not a broken run:
                    # the cell completed and reported; the disagreement is the finding (2026-09-14 r1b2 s7:
                    # the nested handle's type after the ancestor's revoke reads 2 on the RTL, 7 on the emulator).
                    rec["oracle"]["match"] = False
                    rec["status"], rec["reason"] = "completed", f"mark differs from the emulator's: expected {c['want']} observed {mark:06X} (hardware-vs-emulator difference, to be read against the RTL)"
            elif pg:
                rec["oracle"]["observed"] = f"{pg.group(1)} {pg.group(2)}"
                rec["status"], rec["reason"] = ("completed", "every claim held") if pg.group(2) == "GOOD" else ("unsafe-success", f"the domain reported {pg.group(2)}")
            elif wed:
                if c["want"] == "FAULT":
                    cause = (int(tl.group(1), 16) & 0x7F) if tl else None
                    seen = bool(int(tl.group(1), 16) & 0x80) if tl else False
                    if tl and seen:
                        rec["oracle"]["mcause"] = cause; rec["oracle"]["mcause_name"] = RTL_CAUSE.get(cause, "?")
                        if c["name"] == "s9":
                            # the pinned emulator (c128-qemu-merge deb7d75756) asserts on this image (helper_csrevoke,
                            # rs1_v->tag); the helper lane's merged tip acaa44c228 (PR #4, REVOKE raises instead of
                            # aborting) halts it at image+0x3D2C, the same `revoke t0` the FPGA's mepc names, with its
                            # cause 24 NOT_CAP (the handle untagged by its ldc, Q-11) -- 2026-09-15 00:5x, built apart
                            rec["oracle"]["emulator"] = dict(pinned="asserts (helper_csrevoke rs1_v->tag), no run record",
                                                             merged_tip_acaa44c228="enforced-fault cause 24 NOT_CAP at image+0x3D2C (the same site)")
                        rec["status"], rec["reason"] = "enforced-fault", f"the domain wedged with mcause {cause} ({RTL_CAUSE.get(cause, '?')}) latched in the trap log (M-1: a domain fault is a wedge on this RTL; the emulator names the same fault by its own cause number)"
                    else:
                        rec["status"], rec["reason"] = "timeout", "the domain did not return and no trap cause was latched"
                else:
                    mepc = re.search(r"trap mepc = (0x[0-9a-f]+)", a)
                    off = (int(mepc.group(1), 16) & 0xFFFFF) if mepc else None
                    if name in ("pgsub", "pghier") and tl and off == 0x44:
                        rec["oracle"]["mepc"] = mepc.group(1); rec["oracle"]["mcause"] = (int(tl.group(1), 16) & 0x7F)
                        rec["status"], rec["reason"] = "unsupported", ("implementation-unavailable: the PostgreSQL test images' entry glue "
                            "(my_first_domain/start.S) opens with `delin gp`, and the image declares no globals boundary (no .capstone_gp_initdesc section), "
                            "so the board monitor delivers no gp at all; the domain traps at image+0x44 with cause 25 (operand is not a capability) "
                            "before any scenario runs; passes on the emulator (ISSUES M-8, mechanism corrected from source 2026-09-15)")
                    else:
                        rec["oracle"]["mepc"] = mepc.group(1) if mepc else None
                        rec["status"], rec["reason"] = "timeout", "a returning cell did not return" + (f" (trap log {tl.group(1)}, mepc {mepc.group(1)})" if (tl and mepc) else "")
            else:
                rec["status"], rec["reason"] = "invalid-run", "the arm ran but printed neither a mark nor a marker (not-reached vs wrong-answer: the guest did not report)"
        records.append(rec)

with open(OUT/"runs.jsonl", "w") as f:
    for r in records: f.write(json.dumps(r) + "\n")
with open(OUT/"points.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["study_id","case_id","arm","series","stop","entry_va","image_sha256_16","repetition","planned_boot","attempted","qemu_rc","expected"])
    for name, c in cells.items():
        for rep in (1,2,3):
            w.writerow([STUDY[name], CASE.get(int(c["stop"])) if c["stop"].isdigit() else name,
                        "custom-plain" if c["arm"]=="plain" else "custom-sublet",
                        "nginx-uaf" if name[0] in "ps" and name[1:].isdigit() else name,
                        c["stop"], c["va"], c["sha16"], rep, c["boot"], (name,rep) in attempted, c["qemu_rc"], c["want"]])
manifest = dict(study="S1,S2", manifest_id="E1-S1S2-fpga-2026-09-14", date=datetime.date.today().isoformat(),
                target="fpga", evidence_type="functional",
                sources=dict(probe_branch="s2/3-manager-hierarchy@d5954cc0e7e7 (llvm-capstone)", built_in="a detached worktree with the DOMAIN_BASE_VA knob (build-nginx-domain.sh, domain-build.sh)",
                             monitor="4274268", module="d04bd83 (buildroot #3)", bitstream="caplifive_r30r31_1bfff7776.bit"),
                hardware=dict(core="CVA6 Capstone, Genesys2", clock="25 MHz constraint, timing not closed (WNS -12.425 ns)", node_table="65,536 entries, 65,532 usable, no reclamation",
                              domain_trap_vector="none (M-1): a capability fault inside a domain wedges the core; FAULT cells are read from the trap log by the debug mux",
                              cause_numbering="RTL riscv_pkg.sv: 25 UNEXPECTED_OPERAND_TYPE, 26 INVALID_CAPABILITY, 27 UNEXPECTED_CAPABILITY_TYPE, 28 INSUFFICIENT_PERMISSION, 29 OUT_OF_BOUND, 30 ILLEGAL_OPERAND_VALUE; the emulator's numbers differ (spec base), so causes are compared by name"),
                schedule=dict(repetitions=3, boots=3, order="returning cells in cells.tsv order, the boot's one FAULT cell last", control="k800 = 4 first"),
                boots=boots_meta, cells={n: dict(image=dom_file(n), sha256_16=c["sha16"], entry_va=c["va"], expected=c["want"], qemu_rc=c["qemu_rc"]) for n, c in cells.items()})
(OUT/"manifest.json").write_text(json.dumps(manifest, indent=2))
# summary
by = {}
for r in records: by.setdefault(r["parameters"]["image_sha256_16"] + " " + r["case_id"], []).append(r)
lines = ["# E1: the S1/S2 cells on the FPGA (summary, generated)", "", f"boots: {[b['boot_id'] for b in boots_meta]}", "",
         "| cell | arm | expected | repetitions (status: observed) |", "|---|---|---|---|"]
for name, c in cells.items():
    rs = [r for r in records if r["parameters"]["image_sha256_16"] == c["sha16"]]
    lines.append(f"| {name} ({c['stop']}) | {c['arm']} | {c['want']} | " + "; ".join(f"r{r['repetition']} {r['status']}: {r['oracle']['observed'] or r['oracle']['trap_log'] or '-'}" for r in sorted(rs, key=lambda r: r['repetition'])) + " |")
lines += ["", "Not closable here: tab:safety's 'retained stale pointer after node reuse' row waits on M1 (no node is reused on this bitstream).",
          "FAULT cells on this RTL are wedges read from the trap log (M-1); their pc is not recoverable from the mux, only the cause."]
(OUT/"summary.md").write_text("\n".join(lines) + "\n")
print(f"records: {len(records)}; attempted cells: {len(attempted)}; boots: {len(boots_meta)}; out: {OUT}")
for r in records: print(f"  r{r['repetition']} {r['case_id']:40} {r['arm']:13} {r['status']:15} {r['oracle']['observed'] or r['oracle']['trap_log'] or '-'}")
